from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_transforms(augment: bool = True, randaugment: bool = False):
    train_ops = []
    if augment:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if randaugment:
            train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_ops = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def get_cifar100_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int = 2,
    augment: bool = True,
    randaugment: bool = False,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    data_root = Path(data_root).expanduser()
    train_tf, test_tf = build_transforms(augment=augment, randaugment=randaugment)
    train_set = datasets.CIFAR100(
        root=str(data_root), train=True, transform=train_tf, download=download
    )
    test_set = datasets.CIFAR100(
        root=str(data_root), train=False, transform=test_tf, download=download
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
