#!/usr/bin/env python3
"""
Inference efficiency profiler for the ViMoE V2 multiclass pipeline.

This script follows the same path arguments used by train_multiclass.py and
profiles inference on the validation split (or a 90/10 split of the training
set if no validation file is provided).

Reported metrics:
1. Compute Cost & Parameters
   - Params (total / trainable)
   - Forward MACs / FLOPs for a single-sample forward pass
2. Single-sample inference latency
   - Forward latency mean / P95
   - Single-sample throughput
3. Throughput & batch latency
   - Forward batch latency
   - End-to-end batch latency
   - Mean per-sample latency
   - Eval samples / sec
4. CUDA peak memory
   - Max allocated / reserved

TTFT / TPOT are intentionally reported as N/A because ViMoE V2 is a
non-autoregressive classifier and does not use model.generate().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence


GT_SIZE = 224
WORD_TOKEN_LENGTH = 197
IMAGE_TOKEN_LENGTH = 197

token_uncased = None
clip_processor = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViMoE V2 六分类推理效率测试")
    parser.add_argument(
        "-train_file",
        type=str,
        default="/data1/hsiri/AMG/datasets/trianDBthinkGeminiCOT3_260107.jsonl",
    )
    parser.add_argument(
        "-val_file",
        type=str,
        default="/data1/hsiri/AMG/datasets/val.json",
        help="验证集 JSONL 路径，为空则从训练集切分 10%%",
    )
    parser.add_argument(
        "-train_image_root",
        type=str,
        default="/data1/hsiri/AMG/datasets/AMG_MEDIA/train_imagesN",
        help="训练集图片根目录",
    )
    parser.add_argument(
        "-val_image_root",
        type=str,
        default="/data1/hsiri/AMG/datasets/AMG_MEDIA/val_imagesN",
        help="验证集图片根目录，为空则与训练集相同",
    )
    parser.add_argument(
        "-dataset_name",
        type=str,
        default="Twitter",
        help="英文数据用 Twitter/gossip/politi，中文数据用其他名称",
    )
    parser.add_argument("-output_dir", type=str, default="./checkpoints/multiclass")
    parser.add_argument(
        "-checkpoint",
        type=str,
        default="",
        help="预训练权重路径（可加载二分类权重，strict=False）",
    )
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-finetune", type=int, default=0, help="保留训练脚本兼容参数，推理中不使用")
    parser.add_argument("-val_only", action="store_true", help="保留训练脚本兼容参数，推理中不使用")
    parser.add_argument("-int_lr", type=float, default=1e-4, help="保留训练脚本兼容参数，推理中不使用")
    parser.add_argument("-int_beta", type=float, default=0.7, help="保留训练脚本兼容参数，推理中不使用")
    parser.add_argument("-agr_threshold", type=float, default=0.3)
    parser.add_argument("-sem_threshold", type=float, default=0.3)
    parser.add_argument("-max_words", type=int, default=512, help="文本最大字数截断")
    parser.add_argument(
        "-bert_path",
        type=str,
        default="/map-vepfs/liniuniu/hesirui/bert-base-uncased",
    )
    parser.add_argument(
        "-clip_path",
        type=str,
        default="/map-vepfs/liniuniu/hesirui/clip-vit-base-patch16",
    )
    parser.add_argument(
        "-mae_path",
        type=str,
        default="/map-vepfs/liniuniu/hesirui/mae_pretrain_vit_base.pth",
    )
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("-warmup_batches", type=int, default=2, help="正式计时前的预热 batch 数")
    parser.add_argument("-single_samples", type=int, default=16, help="单样本延迟测试用样本数")
    parser.add_argument("-limit_samples", type=int, default=0, help="限制参与 profiling 的总样本数")
    parser.add_argument(
        "-profile_flops",
        action="store_true",
        help="尝试使用 thop/fvcore 统计单样本 forward MACs/FLOPs",
    )
    parser.add_argument("-seed", type=int, default=2024)
    parser.add_argument("-output_json", type=str, default="", help="可选 JSON 输出路径")
    return parser.parse_args()


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": len(clean),
        "mean": float(statistics.mean(clean)),
        "std": float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
    }


def human_count(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(float(value))
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs_value >= scale:
            return f"{value / scale:.3f}{unit}"
    return f"{value:.3f}"


def set_seed(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def to_device(x: Any, device: Any) -> Any:
    return x.to(device)


def sync_if_needed(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def reset_peak_memory(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.reset_peak_memory_stats(device)


def collect_peak_memory(torch_module: Any, device: Any) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
    return {
        "cuda_max_memory_allocated_mb": torch_module.cuda.max_memory_allocated(device) / (1024 ** 2),
        "cuda_max_memory_reserved_mb": torch_module.cuda.max_memory_reserved(device) / (1024 ** 2),
    }


def collate_fn_english(data: Sequence[Any]) -> Any:
    sents = [item[0][0] for item in data]
    image = [item[0][1] for item in data]
    image_aug = [item[0][2] for item in data]
    labels = [item[0][3] for item in data]
    category = [0 for _ in data]
    gt_path = [item[1] for item in data]

    token_data = token_uncased.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=WORD_TOKEN_LENGTH,
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

    import torch

    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    return (
        (input_ids, attention_mask, token_type_ids),
        (image, image_aug, labels, category, sents),
        clip_inputs,
        gt_path,
    )


def build_eval_dataset(args: argparse.Namespace, dataset_cls: Any, random_split_fn: Any, torch_module: Any) -> Any:
    train_dataset = dataset_cls(
        ann_file=args.train_file,
        root_dir=args.train_image_root,
        image_size=GT_SIZE,
        is_train=True,
        max_words=args.max_words,
    )

    val_image_root = args.val_image_root if args.val_image_root else args.train_image_root
    if args.val_file and os.path.exists(args.val_file):
        eval_dataset = dataset_cls(
            ann_file=args.val_file,
            root_dir=val_image_root,
            image_size=GT_SIZE,
            is_train=False,
            max_words=args.max_words,
        )
    else:
        total = len(train_dataset)
        val_size = max(1, int(total * 0.1))
        train_size = total - val_size
        _, eval_dataset = random_split_fn(
            train_dataset,
            [train_size, val_size],
            generator=torch_module.Generator().manual_seed(args.seed),
        )
    return eval_dataset


def move_batch_to_device(items: Any, device: Any) -> Dict[str, Any]:
    texts, others, clip_inputs, _ = items
    input_ids, attention_mask, token_type_ids = texts
    image, _, labels, _, _ = others
    return {
        "input_ids": to_device(input_ids, device),
        "attention_mask": to_device(attention_mask, device),
        "token_type_ids": to_device(token_type_ids, device),
        "image": to_device(image, device),
        "labels": to_device(labels, device),
        "clip_inputs": clip_inputs.to(device),
    }


def forward_model(model: Any, batch: Dict[str, Any]) -> Any:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        image=batch["image"],
        clip_inputs=batch["clip_inputs"],
    )


def build_model(args: argparse.Namespace, model_cls: Any) -> Any:
    return model_cls(
        dataset=args.dataset_name,
        text_token_len=WORD_TOKEN_LENGTH,
        image_token_len=IMAGE_TOKEN_LENGTH,
        is_use_bce=False,
        batch_size=args.batch_size,
        thresh=0.5,
        agr_threshold=args.agr_threshold,
        sem_threshold=args.sem_threshold,
        warmup_epochs=0,
        num_classes=6,
        bert_path=args.bert_path,
        clip_path=args.clip_path,
        mae_path=args.mae_path,
    )


def load_checkpoint(torch_module: Any, model: Any, checkpoint_path: str) -> None:
    assert checkpoint_path and os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    state_dict = torch_module.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in state_dict if k not in filtered]
    if skipped:
        print(f"Skipped {len(skipped)} keys due to shape mismatch: {skipped}")
    model.load_state_dict(filtered, strict=False)


def warmup_model(model: Any, loader: Any, args: argparse.Namespace, torch_module: Any, device: Any) -> None:
    if args.warmup_batches <= 0:
        return
    model.eval()
    with torch_module.inference_mode():
        for batch_idx, items in enumerate(loader):
            if batch_idx >= args.warmup_batches:
                break
            batch = move_batch_to_device(items, device)
            sync_if_needed(torch_module, device)
            _ = forward_model(model, batch)
            sync_if_needed(torch_module, device)


def profile_single_sample_latency(
    model: Any,
    loader: Any,
    torch_module: Any,
    device: Any,
) -> Dict[str, Any]:
    latencies: List[float] = []
    throughputs: List[float] = []

    model.eval()
    with torch_module.inference_mode():
        for items in loader:
            batch = move_batch_to_device(items, device)
            sync_if_needed(torch_module, device)
            start = time.perf_counter()
            _ = forward_model(model, batch)
            sync_if_needed(torch_module, device)
            latency = time.perf_counter() - start
            latencies.append(latency)
            if latency > 0:
                throughputs.append(1.0 / latency)

    return {
        "forward_latency_s": {
            "mean": summarize(latencies)["mean"],
            "p95": summarize(latencies)["p95"],
        },
        "samples_per_s": {
            "mean": summarize(throughputs)["mean"],
            "p95": summarize(throughputs)["p95"],
        },
        "autoregressive_metrics": {
            "ttft_s": None,
            "tpot_s": None,
            "generated_tokens_per_s": None,
            "reason": "N/A: ViMoE V2 is a non-autoregressive classifier and does not use model.generate().",
        },
    }


def profile_batch_throughput(
    model: Any,
    loader: Any,
    torch_module: Any,
    device: Any,
) -> Dict[str, Any]:
    forward_batch_latencies: List[float] = []
    e2e_batch_latencies: List[float] = []
    sample_latencies: List[float] = []
    total_samples = 0
    total_e2e_time = 0.0

    model.eval()
    iterator = iter(loader)
    with torch_module.inference_mode():
        while True:
            e2e_start = time.perf_counter()
            try:
                items = next(iterator)
            except StopIteration:
                break

            batch = move_batch_to_device(items, device)
            batch_size = int(batch["labels"].shape[0])

            sync_if_needed(torch_module, device)
            forward_start = time.perf_counter()
            _ = forward_model(model, batch)
            sync_if_needed(torch_module, device)

            forward_latency = time.perf_counter() - forward_start
            e2e_latency = time.perf_counter() - e2e_start

            forward_batch_latencies.append(forward_latency)
            e2e_batch_latencies.append(e2e_latency)
            sample_latencies.append(forward_latency / batch_size)
            total_samples += batch_size
            total_e2e_time += e2e_latency

    return {
        "num_batches": len(forward_batch_latencies),
        "num_samples": total_samples,
        "forward_batch_latency_s": summarize(forward_batch_latencies),
        "e2e_batch_latency_s": summarize(e2e_batch_latencies),
        "forward_sample_latency_mean_s": summarize(sample_latencies)["mean"],
        "eval_samples_per_sec": (total_samples / total_e2e_time) if total_e2e_time > 0 else None,
    }


def maybe_profile_flops(model: Any, sample_batch: Dict[str, Any], torch_nn_module: Any) -> Dict[str, Any]:
    clip_tensor_keys = [key for key, value in sample_batch["clip_inputs"].items() if hasattr(value, "shape")]

    class ForwardWrapper(torch_nn_module.Module):
        def __init__(self, wrapped_model: Any, keys: Sequence[str]) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.keys = list(keys)

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any, image: Any, *clip_values: Any) -> Any:
            clip_inputs = {key: value for key, value in zip(self.keys, clip_values)}
            return self.wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image=image,
                clip_inputs=clip_inputs,
            )[0]

    wrapper = ForwardWrapper(model, clip_tensor_keys).eval()
    inputs = (
        sample_batch["input_ids"],
        sample_batch["attention_mask"],
        sample_batch["token_type_ids"],
        sample_batch["image"],
        *[sample_batch["clip_inputs"][key] for key in clip_tensor_keys],
    )
    report: Dict[str, Any] = {
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


def build_static_cost_report(model: Any, sample_batch: Dict[str, Any], args: argparse.Namespace, torch_nn_module: Any) -> Dict[str, Any]:
    params_total = sum(param.numel() for param in model.parameters())
    params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    report: Dict[str, Any] = {
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "forward_macs": None,
        "forward_flops": None,
        "backend": None,
        "error": None,
    }
    if args.profile_flops:
        report.update(maybe_profile_flops(model, sample_batch, torch_nn_module))
    return report


def print_human_readable(report: Dict[str, Any]) -> None:
    static_cost = report["static_cost"]
    single_latency = report["single_sample_inference"]
    throughput = report["throughput"]
    memory = report["memory_peak"]

    print("=" * 80)
    print("1. Compute Cost & Parameters")
    print("=" * 80)
    print(f"Params (total):      {human_count(static_cost['params_total'])} ({static_cost['params_total']})")
    print(f"Params (trainable):  {human_count(static_cost['params_trainable'])} ({static_cost['params_trainable']})")
    print(f"Forward MACs:        {human_count(static_cost['forward_macs'])} ({static_cost['forward_macs']})")
    print(f"Forward FLOPs:       {human_count(static_cost['forward_flops'])} ({static_cost['forward_flops']})")
    if static_cost.get("backend"):
        print(f"Static analysis backend: {static_cost['backend']}")
    if static_cost.get("error"):
        print(f"Static analysis note:    {static_cost['error']}")

    print()
    print("=" * 80)
    print("2. Single-sample Inference Latency")
    print("=" * 80)
    print(
        "Forward latency (s): "
        f"mean={single_latency['forward_latency_s']['mean']}  "
        f"p95={single_latency['forward_latency_s']['p95']}"
    )
    print(
        "Samples / sec:       "
        f"mean={single_latency['samples_per_s']['mean']}  "
        f"p95={single_latency['samples_per_s']['p95']}"
    )
    print(f"TTFT / TPOT:         {single_latency['autoregressive_metrics']['reason']}")

    print()
    print("=" * 80)
    print("3. Throughput & Batch Latency")
    print("=" * 80)
    forward_batch = throughput["forward_batch_latency_s"]
    e2e_batch = throughput["e2e_batch_latency_s"]
    print(
        "Forward batch latency(s): "
        f"mean={forward_batch['mean']}  p50={forward_batch['p50']}  "
        f"p90={forward_batch['p90']}  p95={forward_batch['p95']}"
    )
    print(
        "E2E batch latency(s):     "
        f"mean={e2e_batch['mean']}  p50={e2e_batch['p50']}  "
        f"p90={e2e_batch['p90']}  p95={e2e_batch['p95']}"
    )
    print(f"Forward sample latency mean(s): {throughput['forward_sample_latency_mean_s']}")
    print(f"Eval samples / sec:             {throughput['eval_samples_per_sec']}")

    print()
    print("=" * 80)
    print("4. VRAM Memory Peak")
    print("=" * 80)
    print(f"CUDA Max Memory Allocated (MB): {memory['cuda_max_memory_allocated_mb']}")
    print(f"CUDA Max Memory Reserved  (MB): {memory['cuda_max_memory_reserved_mb']}")


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, random_split
    from transformers import BertTokenizer, CLIPProcessor

    from data.multiclass_dataset import MultiClassDataset
    from models.vimoe_v2 import Vimoe_V2

    set_seed(torch, args.seed)
    device = torch.device(args.device)

    global token_uncased, clip_processor
    token_uncased = BertTokenizer.from_pretrained(args.bert_path)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)

    eval_dataset = build_eval_dataset(args, MultiClassDataset, random_split, torch)
    if args.limit_samples > 0:
        eval_dataset = Subset(eval_dataset, range(min(args.limit_samples, len(eval_dataset))))
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty after applying the current file paths / sample limit.")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_english,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    single_sample_count = min(args.single_samples, len(eval_dataset))
    single_dataset = Subset(eval_dataset, range(single_sample_count))
    single_loader = DataLoader(
        single_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_english,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    print("Building ViMoE V2 model for inference profiling")
    model = build_model(args, Vimoe_V2)
    load_checkpoint(torch, model, args.checkpoint)
    model = model.to(device)
    model.eval()

    warmup_model(model, eval_loader, args, torch, device)
    reset_peak_memory(torch, device)

    sample_items = next(iter(single_loader))
    sample_batch = move_batch_to_device(sample_items, device)

    static_cost = build_static_cost_report(model, sample_batch, args, nn)
    single_sample_inference = profile_single_sample_latency(model, single_loader, torch, device)
    throughput = profile_batch_throughput(model, eval_loader, torch, device)
    memory_peak = collect_peak_memory(torch, device)

    report = {
        "checkpoint": args.checkpoint,
        "dataset_name": args.dataset_name,
        "device": str(device),
        "num_eval_samples": len(eval_dataset),
        "batch_size": args.batch_size,
        "static_cost": static_cost,
        "single_sample_inference": single_sample_inference,
        "throughput": throughput,
        "memory_peak": memory_peak,
    }

    print_human_readable(report)
    print()
    print(json.dumps(report, indent=2, ensure_ascii=False))

    output_json = args.output_json
    if not output_json:
        os.makedirs(args.output_dir, exist_ok=True)
        output_json = os.path.join(args.output_dir, "inference_profile.json")
    else:
        parent = os.path.dirname(output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {output_json}")


if __name__ == "__main__":
    main()
