"""
GTSRB dataset loader (Kaggle CSV format).

Expected layout (Kaggle "GTSRB - German Traffic Sign Recognition Benchmark"):

    <root>/
        Train.csv        # columns: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
        Test.csv         # same columns; Path like "Test/00000.png"
        Train/0/..png ... Train/42/..png
        Test/..png
        Meta/ (unused here)

The loader reads the CSV, optionally crops each image to its ROI (recommended:
it removes the border padding and improves accuracy), resizes to a fixed size,
and normalizes with ImageNet statistics because the perception model is a
MobileNetV3-Small pretrained on ImageNet.

Windows / Spyder note: keep num_workers=0 (the default) unless you run the
training script from a proper `if __name__ == "__main__":` guard, which we do.
"""

import os
from typing import Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

NUM_CLASSES = 43
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve(root: str, rel_path: str) -> str:
    """Join a CSV 'Path' (forward-slash relative) onto root, cross-platform."""
    return os.path.join(root, *rel_path.replace("\\", "/").split("/"))


class GTSRBCsvDataset(Dataset):
    """GTSRB images listed in a Kaggle-style CSV (Path + ClassId columns)."""

    def __init__(
        self,
        root: str,
        csv_file: str,
        transform: Optional[transforms.Compose] = None,
        use_roi: bool = True,
    ):
        self.root = root
        self.transform = transform
        self.use_roi = use_roi

        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(root, csv_file)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

        required = {"Path", "ClassId"}
        if not required.issubset(self.df.columns):
            raise ValueError(
                f"CSV {csv_path} is missing columns {required - set(self.df.columns)}. "
                f"Found: {list(self.df.columns)}"
            )
        self.has_roi = {"Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"}.issubset(self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = _resolve(self.root, str(row["Path"]))
        image = Image.open(img_path).convert("RGB")

        if self.use_roi and self.has_roi:
            box = (int(row["Roi.X1"]), int(row["Roi.Y1"]),
                   int(row["Roi.X2"]), int(row["Roi.Y2"]))
            # guard against degenerate boxes
            if box[2] > box[0] and box[3] > box[1]:
                image = image.crop(box)

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["ClassId"])
        return image, label


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Transforms for training (light augmentation) and evaluation."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(
    root: str,
    img_size: int = 64,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 0,
    use_roi: bool = True,
    train_csv: str = "Train.csv",
    test_csv: str = "Test.csv",
    seed: int = 42,
):
    """Return (train_loader, val_loader, test_loader).

    The validation set is carved out of the training CSV. Note the val split
    reuses the *training* transform's dataset object but with the eval transform,
    so we build two dataset views and split by index for a clean eval transform.
    """
    # Full training set with train transform, and a mirror with eval transform.
    train_full = GTSRBCsvDataset(root, train_csv, build_transforms(img_size, True), use_roi)
    eval_mirror = GTSRBCsvDataset(root, train_csv, build_transforms(img_size, False), use_roi)

    n_total = len(train_full)
    n_val = int(round(val_split * n_total))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=g)

    train_ds = torch.utils.data.Subset(train_full, list(train_idx))
    val_ds = torch.utils.data.Subset(eval_mirror, list(val_idx))
    test_ds = GTSRBCsvDataset(root, test_csv, build_transforms(img_size, False), use_roi)

    common = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
