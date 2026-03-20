"""
六分类虚假新闻检测 测试脚本

输出指标:
  - 二分类 (真新闻 vs 假新闻): ACC, Macro-F1
  - 六分类: ACC, Macro-F1, per-class precision/recall/F1

用法:
  python test_multiclass.py \
    -test_file /data1/hsiri/AMG/datasets/val.json \
    -test_image_root /data1/hsiri/AMG/datasets/AMG_MEDIA/val_imagesN \
    -checkpoint ./checkpoints/multiclass/best_xxx.pkl \
    -device cuda:0
"""

import os
import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.autograd import Variable
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor

from models.vimoe_v2 import Vimoe_V2
from data.multiclass_dataset import MultiClassDataset, NUM_CLASSES

GT_size = 224
word_token_length = 197
image_token_length = 197

token_uncased = None
clip_processor = None


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


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


def evaluate(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for items in loader:
            texts, others, clip_inputs, GT_path = items
            input_ids, attention_mask, token_type_ids = texts
            image, image_aug, labels, category, sents = others

            input_ids = to_var(input_ids)
            attention_mask = to_var(attention_mask)
            token_type_ids = to_var(token_type_ids)
            image = to_var(image)
            labels = to_var(labels)
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

    # ---- 二分类 (0=真新闻, 1-5=假新闻) ----
    binary_labels = (all_labels > 0).astype(int)
    binary_preds = (all_preds > 0).astype(int)
    binary_acc = accuracy_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds, average="macro", zero_division=0)
    binary_report = classification_report(
        binary_labels, binary_preds,
        target_names=["Real", "Fake"], digits=4, zero_division=0
    )
    binary_cm = confusion_matrix(binary_labels, binary_preds)

    # ---- 六分类 ----
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


def main(args):
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_english,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Building ViMoE V2 model with {NUM_CLASSES} classes")
    model = Vimoe_V2(
        dataset=args.dataset_name,
        text_token_len=word_token_length,
        image_token_len=image_token_length,
        is_use_bce=False,
        batch_size=args.batch_size,
        thresh=0.5,
        num_classes=NUM_CLASSES,
        bert_path=args.bert_path,
        clip_path=args.clip_path,
        mae_path=args.mae_path,
    )

    assert args.checkpoint and os.path.exists(args.checkpoint), \
        f"Checkpoint not found: {args.checkpoint}"
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    # positional buffers 在 forward 中按实际 batch_size 重新计算，跳过 shape 不匹配的
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in state_dict if k not in filtered]
    if skipped:
        print(f"Skipped {len(skipped)} keys due to shape mismatch: {skipped}")
    model.load_state_dict(filtered, strict=False)

    model = model.to(args.device)
    metrics = evaluate(test_loader, model, args.device)

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Binary  ACC={metrics['binary_acc']:.4f}  Macro-F1={metrics['binary_f1']:.4f}")
    print(f"  Multi   ACC={metrics['multi_acc']:.4f}  Macro-F1={metrics['multi_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMoE V2 六分类测试")
    parser.add_argument("-test_file", type=str, required=True,
                        help="测试集 JSONL 路径")
    parser.add_argument("-test_image_root", type=str, required=True,
                        help="测试集图片根目录")
    parser.add_argument("-checkpoint", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("-dataset_name", type=str, default="Twitter")
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-max_words", type=int, default=512)
    parser.add_argument("-bert_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/bert-base-uncased")
    parser.add_argument("-clip_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/clip-vit-base-patch16")
    parser.add_argument("-mae_path", type=str,
                        default="/map-vepfs/liniuniu/hesirui/mae_pretrain_vit_base.pth")
    args = parser.parse_args()
    main(args)
