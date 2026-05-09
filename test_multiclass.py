"""
六分类虚假新闻检测测试与推理效率脚本

支持两种模式:
1. 常规测试:
   - 提供 checkpoint，输出二分类/六分类指标
2. 推理效率 profiling:
   - 可提供或不提供 checkpoint
   - 不提供 checkpoint 时，使用随机初始化权重，仅用于测推理开销

输出指标:
  - 二分类/六分类指标（有权重且未开启 profile_only 时）
  - Params total/trainable
  - 单样本 forward latency
  - batch latency / E2E latency
  - samples / sec
  - CUDA 显存峰值
  - 可选单样本 forward MACs / FLOPs
"""

import os
import json
import time
import math
import argparse
import statistics

OPTIONAL_IMPORT_ERROR = None
try:
    import numpy as np
    from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from transformers import BertTokenizer, CLIPProcessor

    from models.vimoe_v2 import Vimoe_V2
    from data.multiclass_dataset import MultiClassDataset, NUM_CLASSES
except ModuleNotFoundError as exc:
    OPTIONAL_IMPORT_ERROR = exc
    np = None
    classification_report = None
    accuracy_score = None
    f1_score = None
    confusion_matrix = None
    torch = None
    nn = None
    DataLoader = None
    Subset = None
    BertTokenizer = None
    CLIPProcessor = None
    Vimoe_V2 = None
    MultiClassDataset = None
    NUM_CLASSES = 6

GT_size = 224
word_token_length = 197
image_token_length = 197

token_uncased = None
clip_processor = None


def to_device(x, device):
    return x.to(device)


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def collect_peak_memory(device):
    if device.type != "cuda":
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
    return {
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "cuda_max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    }


def percentile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values = sorted(float(v) for v in values)
    idx = (len(values) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return values[lower]
    weight = idx - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(values):
    values = [float(v) for v in values]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def human_count(value):
    if value is None:
        return "n/a"
    abs_value = abs(float(value))
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs_value >= scale:
            return f"{value / scale:.3f}{unit}"
    return f"{value:.3f}"


def collate_fn_english(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    image_aug = [i[0][2] for i in data]
    labels = [i[0][3] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]

    token_data = token_uncased.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=word_token_length,
        return_tensors="pt",
        return_length=True,
    )
    clip_inputs = clip_processor(
        text=sents,
        images=image,
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
        return_length=True,
    )

    input_ids = token_data["input_ids"]
    attention_mask = token_data["attention_mask"]
    token_type_ids = token_data["token_type_ids"]

    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    return (
        (input_ids, attention_mask, token_type_ids),
        (image, image_aug, labels, category, sents),
        clip_inputs,
        GT_path,
    )


def move_batch_to_device(items, device):
    texts, others, clip_inputs, GT_path = items
    input_ids, attention_mask, token_type_ids = texts
    image, image_aug, labels, category, sents = others
    return {
        "input_ids": to_device(input_ids, device),
        "attention_mask": to_device(attention_mask, device),
        "token_type_ids": to_device(token_type_ids, device),
        "image": to_device(image, device),
        "labels": to_device(labels, device),
        "clip_inputs": clip_inputs.to(device),
    }


def run_model(model, batch):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        image=batch["image"],
        clip_inputs=batch["clip_inputs"],
    )


def maybe_load_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        print("No checkpoint provided. Using randomly initialized weights for profiling.")
        return False
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in state_dict if k not in filtered]
    if skipped:
        print(f"Skipped {len(skipped)} keys due to shape mismatch: {skipped}")
    model.load_state_dict(filtered, strict=False)
    return True


def evaluate(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for items in loader:
            batch = move_batch_to_device(items, device)
            mix_output, image_only_output, text_only_output, loss_int = run_model(model, batch)
            _, preds = torch.max(mix_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_labels = (all_labels > 0).astype(int)
    binary_preds = (all_preds > 0).astype(int)
    binary_acc = accuracy_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds, average="macro", zero_division=0)
    binary_report = classification_report(
        binary_labels, binary_preds,
        target_names=["Real", "Fake"], digits=4, zero_division=0
    )
    binary_cm = confusion_matrix(binary_labels, binary_preds)

    multi_acc = accuracy_score(all_labels, all_preds)
    multi_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    multi_report = classification_report(
        all_labels, all_preds, digits=4, zero_division=0
    )
    multi_cm = confusion_matrix(all_labels, all_preds)

    print("=" * 60)
    print("  Binary Classification (Real vs Fake)")
    print("=" * 60)
    print(f"  ACC:      {binary_acc:.4f}")
    print(f"  Macro-F1: {binary_f1:.4f}")
    print()
    print(binary_report)
    print("Confusion Matrix:")
    print(binary_cm)

    print()
    print("=" * 60)
    print(f"  {NUM_CLASSES}-class Classification")
    print("=" * 60)
    print(f"  ACC:      {multi_acc:.4f}")
    print(f"  Macro-F1: {multi_f1:.4f}")
    print()
    print(multi_report)
    print("Confusion Matrix:")
    print(multi_cm)

    return {
        "binary_acc": binary_acc,
        "binary_f1": binary_f1,
        "multi_acc": multi_acc,
        "multi_f1": multi_f1,
    }


def maybe_profile_flops(model, sample_batch):
    clip_tensor_keys = [k for k, v in sample_batch["clip_inputs"].items() if torch.is_tensor(v)]

    class ForwardWrapper(nn.Module):
        def __init__(self, wrapped_model, clip_keys):
            super().__init__()
            self.wrapped_model = wrapped_model
            self.clip_keys = list(clip_keys)

        def forward(self, input_ids, attention_mask, token_type_ids, image, *clip_values):
            clip_inputs = {k: v for k, v in zip(self.clip_keys, clip_values)}
            output = self.wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image=image,
                clip_inputs=clip_inputs,
            )
            return output[0]

    wrapper = ForwardWrapper(model, clip_tensor_keys).eval()
    inputs = (
        sample_batch["input_ids"],
        sample_batch["attention_mask"],
        sample_batch["token_type_ids"],
        sample_batch["image"],
        *[sample_batch["clip_inputs"][k] for k in clip_tensor_keys],
    )

    report = {
        "backend": None,
        "forward_macs": None,
        "forward_flops": None,
        "error": None,
    }

    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(wrapper, inputs=inputs, verbose=False)
        report["backend"] = "thop"
        report["forward_macs"] = float(macs)
        report["forward_flops"] = float(macs) * 2.0
        return report
    except Exception as exc:
        report["error"] = f"thop failed: {exc}"

    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(wrapper, inputs)
        flops = float(analysis.total())
        report["backend"] = "fvcore"
        report["forward_flops"] = flops
        report["forward_macs"] = flops / 2.0
        report["error"] = None
    except Exception as exc:
        report["error"] = f"{report['error']}; fvcore failed: {exc}"

    return report


def profile_single_sample_latency(loader, model, device):
    model.eval()
    latencies = []
    samples_per_sec = []

    with torch.no_grad():
        for items in loader:
            batch = move_batch_to_device(items, device)
            sync_if_needed(device)
            start = time.perf_counter()
            _ = run_model(model, batch)
            sync_if_needed(device)
            latency = time.perf_counter() - start
            latencies.append(latency)
            if latency > 0:
                samples_per_sec.append(1.0 / latency)

    return {
        "forward_latency_s": {
            "mean": summarize(latencies)["mean"],
            "p95": summarize(latencies)["p95"],
        },
        "samples_per_s": {
            "mean": summarize(samples_per_sec)["mean"],
            "p95": summarize(samples_per_sec)["p95"],
        },
        "ttft_s": None,
        "tpot_s": None,
        "note": "N/A for ViMoE V2 because it is a classifier and does not use autoregressive generate().",
    }


def profile_batch_throughput(loader, model, device, profile_batches=0):
    model.eval()
    forward_batch_latencies = []
    e2e_batch_latencies = []
    sample_latencies = []
    total_samples = 0
    total_e2e_time = 0.0

    iterator = iter(loader)
    batch_idx = 0
    with torch.no_grad():
        while True:
            if profile_batches > 0 and batch_idx >= profile_batches:
                break
            e2e_start = time.perf_counter()
            try:
                items = next(iterator)
            except StopIteration:
                break

            batch = move_batch_to_device(items, device)
            batch_size = int(batch["labels"].shape[0])

            sync_if_needed(device)
            forward_start = time.perf_counter()
            _ = run_model(model, batch)
            sync_if_needed(device)

            forward_latency = time.perf_counter() - forward_start
            e2e_latency = time.perf_counter() - e2e_start

            forward_batch_latencies.append(forward_latency)
            e2e_batch_latencies.append(e2e_latency)
            sample_latencies.append(forward_latency / batch_size)
            total_samples += batch_size
            total_e2e_time += e2e_latency
            batch_idx += 1

    return {
        "num_batches": len(forward_batch_latencies),
        "num_samples": total_samples,
        "forward_batch_latency_s": summarize(forward_batch_latencies),
        "e2e_batch_latency_s": summarize(e2e_batch_latencies),
        "forward_sample_latency_mean_s": summarize(sample_latencies)["mean"],
        "eval_samples_per_sec": (total_samples / total_e2e_time) if total_e2e_time > 0 else None,
    }


def build_profile_report(model, sample_batch, single_loader, test_loader, args, device):
    report = {
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "forward_macs": None,
        "forward_flops": None,
        "flops_backend": None,
        "flops_error": None,
    }
    if args.profile_flops:
        flops_report = maybe_profile_flops(model, sample_batch)
        report["forward_macs"] = flops_report["forward_macs"]
        report["forward_flops"] = flops_report["forward_flops"]
        report["flops_backend"] = flops_report["backend"]
        report["flops_error"] = flops_report["error"]

    report["single_sample_inference"] = profile_single_sample_latency(single_loader, model, device)
    report["throughput"] = profile_batch_throughput(
        test_loader, model, device, profile_batches=args.profile_batches
    )
    report["memory_peak"] = collect_peak_memory(device)
    return report


def print_profile_report(profile):
    print()
    print("=" * 80)
    print("1. Compute Cost & Parameters")
    print("=" * 80)
    print(f"Params (total):      {human_count(profile['params_total'])} ({profile['params_total']})")
    print(f"Params (trainable):  {human_count(profile['params_trainable'])} ({profile['params_trainable']})")
    print(f"Forward MACs:        {human_count(profile['forward_macs'])} ({profile['forward_macs']})")
    print(f"Forward FLOPs:       {human_count(profile['forward_flops'])} ({profile['forward_flops']})")
    if profile["flops_backend"]:
        print(f"Static analysis backend: {profile['flops_backend']}")
    if profile["flops_error"]:
        print(f"Static analysis note:    {profile['flops_error']}")

    single = profile["single_sample_inference"]
    print()
    print("=" * 80)
    print("2. Single-sample Inference Latency")
    print("=" * 80)
    print(
        f"Forward latency (s): mean={single['forward_latency_s']['mean']}  "
        f"p95={single['forward_latency_s']['p95']}"
    )
    print(
        f"Samples / sec:       mean={single['samples_per_s']['mean']}  "
        f"p95={single['samples_per_s']['p95']}"
    )
    print(f"TTFT / TPOT:         {single['note']}")

    throughput = profile["throughput"]
    forward_batch = throughput["forward_batch_latency_s"]
    e2e_batch = throughput["e2e_batch_latency_s"]
    print()
    print("=" * 80)
    print("3. Throughput & Batch Latency")
    print("=" * 80)
    print(
        f"Forward batch latency(s): mean={forward_batch['mean']}  p50={forward_batch['p50']}  "
        f"p90={forward_batch['p90']}  p95={forward_batch['p95']}"
    )
    print(
        f"E2E batch latency(s):     mean={e2e_batch['mean']}  p50={e2e_batch['p50']}  "
        f"p90={e2e_batch['p90']}  p95={e2e_batch['p95']}"
    )
    print(f"Forward sample latency mean(s): {throughput['forward_sample_latency_mean_s']}")
    print(f"Eval samples / sec:             {throughput['eval_samples_per_sec']}")

    memory = profile["memory_peak"]
    print()
    print("=" * 80)
    print("4. VRAM Memory Peak")
    print("=" * 80)
    print(f"CUDA Max Memory Allocated (MB): {memory['cuda_max_memory_allocated_mb']}")
    print(f"CUDA Max Memory Reserved  (MB): {memory['cuda_max_memory_reserved_mb']}")


def main(args):
    if OPTIONAL_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Missing runtime dependency while loading test_multiclass.py. "
            "Please install the project inference dependencies before running the script."
        ) from OPTIONAL_IMPORT_ERROR

    global token_uncased, clip_processor
    token_uncased = BertTokenizer.from_pretrained(args.bert_path)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)

    test_dataset = MultiClassDataset(
        ann_file=args.test_file,
        root_dir=args.test_image_root,
        image_size=GT_size,
        is_train=False,
        max_words=args.max_words,
    )
    if args.limit_samples > 0:
        test_dataset = Subset(test_dataset, range(min(args.limit_samples, len(test_dataset))))
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty after applying the current file paths / sample limit.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_english,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    single_dataset = Subset(test_dataset, range(min(args.single_samples, len(test_dataset))))
    single_loader = DataLoader(
        single_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_english,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Building ViMoE V2 model with {NUM_CLASSES} classes")
    model = Vimoe_V2(
        dataset=args.dataset_name,
        text_token_len=197,
        image_token_len=197,
        is_use_bce=False,
        batch_size=args.batch_size,
        thresh=0.5,
        num_classes=6,
        bert_path=args.bert_path,
        clip_path=args.clip_path,
        mae_path=args.mae_path,
    )

    weights_loaded = maybe_load_checkpoint(model, args.checkpoint)
    if not weights_loaded:
        args.profile_efficiency = True
        args.profile_only = True

    device = torch.device(args.device)
    model = model.to(device)

    profile = None
    if args.profile_efficiency:
        model.eval()
        with torch.no_grad():
            for batch_idx, items in enumerate(test_loader):
                if batch_idx >= args.warmup_batches:
                    break
                batch = move_batch_to_device(items, device)
                sync_if_needed(device)
                _ = run_model(model, batch)
                sync_if_needed(device)

        reset_peak_memory(device)
        sample_batch = move_batch_to_device(next(iter(single_loader)), device)
        profile = build_profile_report(model, sample_batch, single_loader, test_loader, args, device)
        print_profile_report(profile)

    metrics = None
    if weights_loaded and not args.profile_only:
        metrics = evaluate(test_loader, model, device)
        print()
        print("=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  Binary  ACC={metrics['binary_acc']:.4f}  Macro-F1={metrics['binary_f1']:.4f}")
        print(f"  Multi   ACC={metrics['multi_acc']:.4f}  Macro-F1={metrics['multi_f1']:.4f}")

    if args.output_json:
        payload = {
            "checkpoint_loaded": weights_loaded,
            "checkpoint": args.checkpoint,
            "metrics": metrics,
            "profile": profile,
        }
        parent = os.path.dirname(args.output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMoE V2 六分类测试")
    parser.add_argument("-test_file", type=str, required=True,
                        help="测试集 JSONL 路径")
    parser.add_argument("-test_image_root", type=str, required=True,
                        help="测试集图片根目录")
    parser.add_argument("-checkpoint", type=str, default="",
                        help="模型权重路径；留空则使用随机初始化权重做 profiling")
    parser.add_argument("-dataset_name", type=str, default="Twitter")
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-max_words", type=int, default=512)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("-profile_efficiency", action="store_true",
                        help="输出推理效率指标")
    parser.add_argument("-profile_only", action="store_true",
                        help="只做效率 profiling，不输出分类指标")
    parser.add_argument("-profile_flops", action="store_true",
                        help="尝试统计单样本 forward MACs/FLOPs")
    parser.add_argument("-warmup_batches", type=int, default=2,
                        help="正式计时前的预热 batch 数")
    parser.add_argument("-profile_batches", type=int, default=0,
                        help="限制参与 batch profiling 的 batch 数，0 表示全量")
    parser.add_argument("-single_samples", type=int, default=16,
                        help="单样本延迟测试用样本数")
    parser.add_argument("-limit_samples", type=int, default=0,
                        help="限制参与测试/测速的总样本数，0 表示全量")
    parser.add_argument("-output_json", type=str, default="",
                        help="可选 JSON 输出路径")
    parser.add_argument("-bert_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/bert-base-uncased")
    parser.add_argument("-clip_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/clip-vit-base-patch16")
    parser.add_argument("-mae_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/mae_pretrain_vit_base.pth")
    args = parser.parse_args()
    main(args)
