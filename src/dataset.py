import os
import json
import random

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

IMG_SIZE = 480

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_paths(dataset_path: str, split: str):
    """
    Resolve annotation file and image directory from a dataset root.

    Supports two layouts:

    Standard COCO:
        dataset_path/
          annotations/instances_{split}2017.json
          {split}2017/

    Mini / custom:
        dataset_path/
          annotations.json
          images/
    """
    dataset_path = os.path.abspath(dataset_path)

    # --- Standard COCO layout ---
    coco_ann = os.path.join(dataset_path, "annotations", f"instances_{split}2017.json")
    coco_img = os.path.join(dataset_path, f"{split}2017")
    if os.path.isfile(coco_ann) and os.path.isdir(coco_img):
        return coco_ann, coco_img

    # --- Mini / flat layout ---
    mini_ann = os.path.join(dataset_path, "annotations.json")
    mini_img = os.path.join(dataset_path, "images")
    if os.path.isfile(mini_ann) and os.path.isdir(mini_img):
        return mini_ann, mini_img

    raise FileNotFoundError(
        f"Could not find a valid COCO layout under '{dataset_path}'.\n"
        f"  Expected either:\n"
        f"    {coco_ann}  +  {coco_img}/\n"
        f"  or:\n"
        f"    {mini_ann}  +  {mini_img}/"
    )


class CocoDataset(Dataset):
    """
    COCO-format detection dataset.

    Returns per sample:
        image  : float32 tensor (3, H, W), ImageNet-normalised
        target : dict with
                   'boxes'  — (N, 4) float32, normalised [cx, cy, w, h] in [0, 1]
                   'labels' — (N,)   int64,   dense 0-based category index
    """

    def __init__(self, annotation_file: str, image_dir: str, img_size: int = IMG_SIZE):
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        with open(annotation_file) as f:
            coco = json.load(f)

        self.image_dir = image_dir
        self.img_size  = img_size

        self.id_to_image   = {img['id']: img for img in coco['images']}
        self.image_to_anns = {}
        for ann in coco['annotations']:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)

        # Only keep images that have at least one annotation
        self.image_ids = list(self.image_to_anns.keys())

        # Remap sparse COCO category IDs → dense 0-based indices
        cat_ids = sorted({cat['id'] for cat in coco['categories']})
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes   = len(cat_ids)

        print(f"Dataset loaded: {len(self.image_ids)} images, {self.num_classes} categories")
        print(f"  annotations : {annotation_file}")
        print(f"  images      : {image_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id   = self.image_ids[idx]
        image_info = self.id_to_image[image_id]
        anns       = self.image_to_anns[image_id]

        # Load and resize image
        img_path = os.path.join(self.image_dir, image_info['file_name'])
        image    = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Resize longest edge to img_size, preserving aspect ratio
        scale = self.img_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Convert boxes: COCO [x,y,w,h] pixel → DETR [cx,cy,w,h] normalised
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            cx = (x + 0.5 * bw) / orig_w
            cy = (y + 0.5 * bh) / orig_h
            nw = bw / orig_w
            nh = bh / orig_h
            cx, cy, nw, nh = (max(0., min(1., v)) for v in (cx, cy, nw, nh))
            if nw > 0 and nh > 0:
                boxes.append([cx, cy, nw, nh])
                labels.append(self.cat_id_to_idx[ann['category_id']])

        boxes  = torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros((0,), dtype=torch.long)

        return image, {'boxes': boxes, 'labels': labels}


def collate_fn(batch):
    """
    Pad images in a batch to the same spatial size and build a boolean padding mask.

    Returns:
        images  : (B, 3, H_max, W_max)
        masks   : (B, H_max, W_max)  True = padded region (ignored by attention)
        targets : list of dicts, length B
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images, masks = [], []
    for img in images:
        c, h, w = img.shape
        padded = F.pad(img, (0, max_w - w, 0, max_h - h))
        padded_images.append(padded)

        mask = torch.zeros((max_h, max_w), dtype=torch.bool)
        mask[h:, :] = True   # bottom padding rows
        mask[:, w:] = True   # right padding columns
        masks.append(mask)

    return torch.stack(padded_images), torch.stack(masks), targets


def build_dataloaders(
    dataset_path: str,
    split: str = "val",
    batch_size: int = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
):
    """
    Build train and val DataLoaders from a COCO-format dataset.

    Args:
        dataset_path : root directory of the dataset (see _resolve_paths for layout)
        split        : COCO split name — 'val' for val2017, 'train' for train2017
                       (only used for standard COCO layout; ignored for mini layout)
        batch_size   : samples per batch
        train_ratio  : fraction of data used for training (rest goes to val)
        seed         : random seed for the train/val split
        num_workers  : DataLoader worker processes
        img_size     : longest-edge resize target

    Returns:
        train_loader, val_loader, num_classes
    """
    ann_file, img_dir = _resolve_paths(dataset_path, split)
    dataset = CocoDataset(ann_file, img_dir, img_size=img_size)

    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)

    cut        = int(train_ratio * len(indices))
    train_data = Subset(dataset, indices[:cut])
    val_data   = Subset(dataset, indices[cut:])

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"Train: {len(train_data)} images ({len(train_loader)} batches)")
    print(f"Val  : {len(val_data)} images ({len(val_loader)} batches)")

    return train_loader, val_loader, dataset.num_classes
