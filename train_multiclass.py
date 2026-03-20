"""
六分类虚假新闻检测训练脚本 (基于 ViMoE V2)

类别:
  0: 真新闻
  1: 图片伪造
  2: 实体不一致
  3: 事件不一致
  4: 时间不一致
  5: 无效视觉信息

评估指标:
  - 二分类 (真新闻 vs 假新闻): ACC, Macro-F1
  - 六分类: ACC, Macro-F1, per-class precision/recall/F1

用法示例:
  python train_multiclass.py \
    -train_file /data1/hsiri/AMG/datasets/trianDBthinkGeminiCOT3_260107.jsonl \
    -val_file /data1/hsiri/AMG/datasets/val.json \
    -train_image_root /data1/hsiri/AMG/datasets/AMG_MEDIA/train_imagesN \
    -val_image_root /data1/hsiri/AMG/datasets/AMG_MEDIA/val_imagesN \
    -device cuda:0 \
    -batch_size 16 \
    -epochs 100
"""

import os
import random
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, CLIPProcessor
import pytorch_warmup as warmup

from util import Progbar
from models.vimoe_v2 import Vimoe_V2
from data.multiclass_dataset import MultiClassDataset, NUM_CLASSES

# constants
GT_size = 224
word_token_length = 197
image_token_length = 197

# 延迟初始化，等 args 解析后再加载
token_uncased = None
clip_processor = None

stateful_metrics = [
    "CE_loss", "Int_loss", "mean_acc", "lr",
]


def to_device(x, device):
    return x.to(device)


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


def load_model(model, load_path, strict=False):
    load_net = torch.load(load_path, map_location="cpu")
    model.load_state_dict(load_net, strict=strict)


def main(args):
    print(args)
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # ====================== Init tokenizers from local paths ======================
    global token_uncased, clip_processor
    token_uncased = BertTokenizer.from_pretrained(args.bert_path)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    print(f"BertTokenizer: {args.bert_path}")
    print(f"CLIPProcessor: {args.clip_path}")

    # ====================== Data ======================
    train_dataset = MultiClassDataset(
        ann_file=args.train_file,
        root_dir=args.train_image_root,
        image_size=GT_size,
        is_train=True,
        max_words=args.max_words,
    )

    val_image_root = args.val_image_root if args.val_image_root else args.train_image_root
    if args.val_file and os.path.exists(args.val_file):
        validate_dataset = MultiClassDataset(
            ann_file=args.val_file,
            root_dir=val_image_root,
            image_size=GT_size,
            is_train=False,
            max_words=args.max_words,
        )
    else:
        print("No separate val file, splitting 90/10 from training data")
        total = len(train_dataset)
        val_size = int(total * 0.1)
        train_size = total - val_size
        train_dataset, validate_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    collate_fn = collate_fn_english
    print(f"Using English collate (bert-base-uncased + openai/clip)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    # ====================== Model ======================
    print(f"Building ViMoE V2 model with {NUM_CLASSES} classes")
    model = Vimoe_V2(
        dataset=args.dataset_name,
        text_token_len=197,
        image_token_len=197,
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

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in ckpt.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = [k for k in ckpt if k not in filtered]
        if skipped:
            print(f"Skipped {len(skipped)} keys: {skipped}")
        model.load_state_dict(filtered, strict=False)

    model = model.to(args.device)
    model.train()

    # ====================== Loss & Optimizer ======================
    if hasattr(train_dataset, "class_weights"):
        class_weights = train_dataset.class_weights.to(args.device)
    elif hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "class_weights"):
        class_weights = train_dataset.dataset.class_weights.to(args.device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(args.device)

    optim_params_normal, optim_params_fast, optim_params_extremefast = [], [], []
    finetune_encoders = False
    for k, v in model.named_parameters():
        if v.requires_grad:
            if "image_model" in k or "text_model" in k:
                finetune_encoders = True
                optim_params_normal.append(v)
            elif "interaction" in k:
                optim_params_extremefast.append(v)
            else:
                optim_params_fast.append(v)

    fine_tuning = args.finetune > 0
    print(f"Fine-tuning encoders: {fine_tuning}")
    num_steps = int(len(train_loader) * args.epochs * 1.1)

    if finetune_encoders:
        optimizer = torch.optim.AdamW(
            optim_params_normal, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    optimizer_fast = torch.optim.AdamW(
        optim_params_fast,
        lr=5e-5 if not fine_tuning else 1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    optimizer_extremefast = torch.optim.AdamW(
        optim_params_extremefast,
        lr=args.int_lr if not fine_tuning else 1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    scheduler_fast = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fast, T_max=num_steps)
    warmup_scheduler_fast = warmup.UntunedLinearWarmup(optimizer_fast)
    scheduler_extremefast = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_extremefast, T_max=num_steps
    )
    warmup_scheduler_extremefast = warmup.UntunedLinearWarmup(optimizer_extremefast)

    # ====================== Training ======================
    best_val_acc = 0.0
    best_epoch = 0
    best_report = ""
    best_metrics = {"binary_acc": 0, "binary_f1": 0, "multi_acc": 0, "multi_f1": 0}

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        cost_vector, acc_vector = [], []
        int_beta = args.int_beta

        if args.val_only:
            pass
        else:
            total = len(train_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)

            for i, items in enumerate(train_loader):
                texts, others, clip_inputs, GT_path = items
                input_ids, attention_mask, token_type_ids = texts
                image, image_aug, labels, category, sents = others

                input_ids = to_device(input_ids, args.device)
                attention_mask = to_device(attention_mask, args.device)
                token_type_ids = to_device(token_type_ids, args.device)
                image = to_device(image, args.device)
                image_aug = to_device(image_aug, args.device)
                labels = to_device(labels, args.device)
                clip_inputs = clip_inputs.to(args.device)

                mix_output, image_only_output, text_only_output, loss_int = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    image=image,
                    clip_inputs=clip_inputs,
                )

                loss_CE = criterion(mix_output, labels)
                loss_CE_image = criterion(image_only_output, labels)
                loss_CE_text = criterion(text_only_output, labels)
                loss_single_modal = (loss_CE_text + loss_CE_image) / 2
                loss = loss_CE + loss_single_modal + int_beta * loss_int

                if finetune_encoders:
                    optimizer.zero_grad()
                optimizer_fast.zero_grad()
                optimizer_extremefast.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                if epoch >= 10 and finetune_encoders:
                    optimizer.step()
                optimizer_fast.step()
                optimizer_extremefast.step()

                _, preds = torch.max(mix_output, 1)
                accuracy = (preds == labels).float().mean()

                cost_vector.append(loss.item())
                acc_vector.append(accuracy.item())

                logs = [
                    ("CE_loss", loss_CE.item()),
                    ("Image", loss_CE_image.item()),
                    ("Text", loss_CE_text.item()),
                    ("Int_loss", int_beta * loss_int.item()),
                    ("mean_acc", np.mean(acc_vector)),
                ]
                progbar.add(len(image), values=logs)

                if finetune_encoders:
                    with warmup_scheduler.dampening():
                        scheduler.step()
                with warmup_scheduler_fast.dampening():
                    scheduler_fast.step()
                with warmup_scheduler_extremefast.dampening():
                    scheduler_extremefast.step()

            print(
                "Epoch [%d/%d], Loss: %.4f, Train_Acc: %.4f"
                % (epoch + 1, args.epochs, np.mean(cost_vector), np.mean(acc_vector))
            )

        # ====================== Validation ======================
        metrics, val_report = evaluate(
            validate_loader, model, criterion, args.device
        )
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(val_report)

        val_multi_acc = metrics["multi_acc"]
        if val_multi_acc > best_val_acc:
            best_val_acc = val_multi_acc
            best_epoch = epoch + 1
            best_report = val_report
            best_metrics = metrics
            ckpt_path = os.path.join(
                args.output_dir,
                f"best_ep{epoch+1}_{datetime.datetime.now().strftime('%m%d')}_{int(val_multi_acc*100)}.pkl",
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved to {ckpt_path}")

        print(
            f"Best so far (Epoch {best_epoch}): "
            f"Binary ACC={best_metrics['binary_acc']:.4f} F1={best_metrics['binary_f1']:.4f} | "
            f"Multi ACC={best_metrics['multi_acc']:.4f} F1={best_metrics['multi_f1']:.4f}"
        )

        if args.val_only:
            break

    with open(os.path.join(args.output_dir, "results.log"), "a") as f:
        f.write(f"==================== {datetime.datetime.now()} ====================\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"Binary ACC: {best_metrics['binary_acc']:.4f}  Macro-F1: {best_metrics['binary_f1']:.4f}\n")
        f.write(f"Multi  ACC: {best_metrics['multi_acc']:.4f}  Macro-F1: {best_metrics['multi_f1']:.4f}\n")
        f.write(f"{best_report}\n")
        f.write(f"args: {args}\n\n")


def evaluate(loader, model, criterion, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for items in loader:
            texts, others, clip_inputs, GT_path = items
            input_ids, attention_mask, token_type_ids = texts
            image, image_aug, labels, category, sents = others

            input_ids = to_device(input_ids, device)
            attention_mask = to_device(attention_mask, device)
            token_type_ids = to_device(token_type_ids, device)
            image = to_device(image, device)
            labels = to_device(labels, device)
            clip_inputs = clip_inputs.to(device)

            mix_output, image_only_output, text_only_output, loss_int = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image=image,
                clip_inputs=clip_inputs,
            )

            _, preds = torch.max(mix_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- 六分类指标 ----
    multi_acc = accuracy_score(all_labels, all_preds)
    multi_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    multi_report = classification_report(
        all_labels, all_preds, digits=4, zero_division=0
    )

    # ---- 二分类指标 (0=真新闻, 1-5=假新闻) ----
    binary_labels = (all_labels > 0).astype(int)
    binary_preds = (all_preds > 0).astype(int)
    binary_acc = accuracy_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds, average="macro", zero_division=0)
    binary_report = classification_report(
        binary_labels, binary_preds, digits=4, zero_division=0
    )

    report = (
        "========== 二分类指标 (真新闻 vs 假新闻) ==========\n"
        f"Binary ACC: {binary_acc:.4f}    Binary Macro-F1: {binary_f1:.4f}\n"
        f"{binary_report}\n"
        "========== 六分类指标 ==========\n"
        f"Multi ACC:  {multi_acc:.4f}    Multi Macro-F1:  {multi_f1:.4f}\n"
        f"{multi_report}"
    )

    metrics = {
        "binary_acc": binary_acc,
        "binary_f1": binary_f1,
        "multi_acc": multi_acc,
        "multi_f1": multi_f1,
    }

    model.train()
    return metrics, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMoE V2 六分类训练")
    parser.add_argument("-train_file", type=str,
                        default="/data1/hsiri/AMG/datasets/trianDBthinkGeminiCOT3_260107.jsonl")
    parser.add_argument("-val_file", type=str,
                        default="/data1/hsiri/AMG/datasets/val.json",
                        help="验证集 JSONL 路径，为空则从训练集切分 10%%")
    parser.add_argument("-train_image_root", type=str,
                        default="/data1/hsiri/AMG/datasets/AMG_MEDIA/train_imagesN",
                        help="训练集图片根目录")
    parser.add_argument("-val_image_root", type=str,
                        default="/data1/hsiri/AMG/datasets/AMG_MEDIA/val_imagesN",
                        help="验证集图片根目录，为空则与训练集相同")
    parser.add_argument("-dataset_name", type=str, default="Twitter",
                        help="英文数据用 Twitter/gossip/politi，中文数据用其他名称")
    parser.add_argument("-output_dir", type=str, default="./checkpoints/multiclass")
    parser.add_argument("-checkpoint", type=str, default="",
                        help="预训练权重路径（可加载二分类权重，strict=False）")
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-finetune", type=int, default=0,
                        help="是否微调 BERT 和 MAE 编码器")
    parser.add_argument("-val_only", action="store_true",
                        help="仅评估，不训练")
    parser.add_argument("-int_lr", type=float, default=1e-4)
    parser.add_argument("-int_beta", type=float, default=0.7)
    parser.add_argument("-agr_threshold", type=float, default=0.3)
    parser.add_argument("-sem_threshold", type=float, default=0.3)
    parser.add_argument("-max_words", type=int, default=512,
                        help="文本最大字数截断")
    parser.add_argument("-bert_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/bert-base-uncased")
    parser.add_argument("-clip_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/clip-vit-base-patch16")
    parser.add_argument("-mae_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/mae_pretrain_vit_base.pth")
    args = parser.parse_args()
    main(args)
