import json
import os
import random
import copy

import cv2
import numpy as np
import torch
import torch.utils.data as data
import albumentations as A
from PIL import Image

import data.util as util

NUM_CLASSES = 6


class MultiClassDataset(data.Dataset):
    """六分类虚假新闻检测数据集，读取 JSONL 格式。

    JSONL 每行示例:
        {"Id": "xxx", "text": "...", "label": 0}

    图片路径: {root_dir}/{Id}.png
    """

    def __init__(
        self,
        ann_file,
        root_dir,
        image_size=224,
        is_train=True,
        id_field="Id",
        text_field="text",
        label_field="label",
        image_ext=".png",
    ):
        super(MultiClassDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.id_field = id_field
        self.text_field = text_field
        self.label_field = label_field
        self.image_ext = image_ext
        self.not_valid_set = set()

        self.transform_resize = A.Compose(
            [A.Resize(always_apply=True, height=image_size, width=image_size)]
        )
        self.transform_aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.GaussNoise(always_apply=False, p=0.2),
                        A.ISONoise(always_apply=False, p=0.2),
                    ]
                ),
                A.Resize(always_apply=True, height=image_size, width=image_size),
            ]
        )

        self.ann = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.ann.append(item)

        print(f"Loaded {len(self.ann)} samples from {ann_file}")

        class_counts = [0] * NUM_CLASSES
        for item in self.ann:
            label = int(item[self.label_field])
            if 0 <= label < NUM_CLASSES:
                class_counts[label] += 1
        total = sum(class_counts)
        print(f"Class distribution: {dict(enumerate(class_counts))}")

        self.class_weights = torch.tensor(
            [total / max(c, 1) for c in class_counts], dtype=torch.float32
        )
        self.class_weights = self.class_weights / self.class_weights.sum() * NUM_CLASSES

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        find_path = False
        while not find_path:
            item = self.ann[index]
            sample_id = str(item[self.id_field])
            content = str(item[self.text_field])
            label = int(item[self.label_field])

            img_path = os.path.join(self.root_dir, sample_id + self.image_ext)

            if img_path in self.not_valid_set:
                index = random.randint(0, len(self.ann) - 1)
                continue

            if not os.path.exists(img_path):
                index = random.randint(0, len(self.ann) - 1)
                continue

            img_GT = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_GT is None:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    img_GT = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
                except Exception:
                    self.not_valid_set.add(img_path)
                    index = random.randint(0, len(self.ann) - 1)
                    continue

            if img_GT.dtype != np.uint8:
                self.not_valid_set.add(img_path)
                index = random.randint(0, len(self.ann) - 1)
                continue

            H_origin, W_origin = img_GT.shape[:2]
            if H_origin < 10 or W_origin < 10:
                self.not_valid_set.add(img_path)
                index = random.randint(0, len(self.ann) - 1)
                continue

            if img_GT.ndim == 2:
                img_GT = np.expand_dims(img_GT, axis=2)
            if img_GT.shape[2] > 3:
                img_GT = img_GT[:, :, :3]

            img_GT = util.channel_convert(img_GT.shape[2], "RGB", [img_GT])[0]
            find_path = True

        if self.is_train:
            img_GT_aug = self.transform_aug(image=copy.deepcopy(img_GT))["image"]
        else:
            img_GT_aug = self.transform_resize(image=copy.deepcopy(img_GT))["image"]

        img_GT = self.transform_resize(image=copy.deepcopy(img_GT))["image"]

        img_GT = img_GT.astype(np.float32) / 255.0
        img_GT_aug = img_GT_aug.astype(np.float32) / 255.0

        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_GT_aug.shape[2] == 3:
            img_GT_aug = img_GT_aug[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_GT_aug = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT_aug, (2, 0, 1)))
        ).float()

        GT_path = img_path
        return (content, img_GT, img_GT_aug, label, 0), GT_path
