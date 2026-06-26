from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# Data loading and augmentation.
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
    download: bool = True,
    val_size: int = 5000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root).expanduser()
    train_tf, test_tf = build_transforms(augment=augment, randaugment=randaugment)
    train_full = datasets.CIFAR100(
        root=str(data_root), train=True, transform=train_tf, download=download
    )
    val_full = datasets.CIFAR100(
        root=str(data_root), train=True, transform=test_tf, download=download
    )
    test_set = datasets.CIFAR100(
        root=str(data_root), train=False, transform=test_tf, download=download
    )
    if not 0 < val_size < len(train_full):
        raise ValueError(f"val_size must be between 1 and {len(train_full) - 1}, got {val_size}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
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
    return train_loader, val_loader, test_loader


# Model definitions.
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 100, width: int = 48, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, width, stride=1, dropout=dropout),
            nn.MaxPool2d(2),
            ConvBlock(width, width * 2, stride=1, dropout=dropout),
            nn.MaxPool2d(2),
            ConvBlock(width * 2, width * 4, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(width * 4),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: float = 4.0):
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        hidden_ch = int(round(in_ch * expansion))
        layers = []
        if hidden_ch != in_ch:
            layers.append(ConvBNAct(in_ch, hidden_ch, kernel_size=1))
        layers.extend(
            [
                ConvBNAct(hidden_ch, hidden_ch, kernel_size=3, stride=stride, groups=hidden_ch),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


def make_mv2_stage(
    in_ch: int,
    out_ch: int,
    repeats: int,
    stride: int,
    expansion: float,
) -> nn.Sequential:
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    layers = [InvertedResidual(in_ch, out_ch, stride=stride, expansion=expansion)]
    for _ in range(1, repeats):
        layers.append(InvertedResidual(out_ch, out_ch, stride=1, expansion=expansion))
    return nn.Sequential(*layers)


def make_optional_mv2_stage(
    channels: int,
    repeats: int,
    expansion: float,
) -> nn.Module:
    if repeats < 0:
        raise ValueError(f"repeats must be non-negative, got {repeats}")
    if repeats == 0:
        return nn.Identity()
    return make_mv2_stage(
        channels,
        channels,
        repeats=repeats,
        stride=1,
        expansion=expansion,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 100, base_width: int = 64, dropout: float = 0.0):
        super().__init__()
        self.in_ch = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, blocks=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_width * 8, num_classes),
        )

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def make_encoder(
    embed_dim: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth)


class ConvFeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, grid_size: int, dropout: float):
        super().__init__()
        self.grid_size = grid_size
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        bsz, num_tokens, hidden_dim = x.shape
        expected_tokens = self.grid_size * self.grid_size
        if num_tokens != expected_tokens:
            raise ValueError(
                f"ConvFeedForward expected {expected_tokens} tokens, got {num_tokens}"
            )
        x = x.transpose(1, 2).reshape(bsz, hidden_dim, self.grid_size, self.grid_size)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(self.act(x))
        return self.drop(self.fc2(x))


class ConvFFNTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        grid_size: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = ConvFeedForward(
            embed_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            grid_size=grid_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


def make_convffn_encoder(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    grid_size: int,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            ConvFFNTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, grid_size)
            for _ in range(depth)
        ]
    )


class NoAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the same qkv/projection capacity as attention, but remove token-token interaction.
        value = self.qkv(self.norm1(x)).chunk(3, dim=-1)[2]
        x = x + self.drop(self.proj(value))
        x = x + self.ffn(self.norm2(x))
        return x


def make_no_attention_encoder(
    embed_dim: int, depth: int, mlp_ratio: float, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        *[NoAttentionBlock(embed_dim, mlp_ratio, dropout) for _ in range(depth)]
    )


class TinyViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        pool: str = "cls",
    ):
        super().__init__()
        self.pool = pool
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)
        self.encoder = make_encoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.drop(x)
        x = self.encoder(x)
        if self.pool == "mean":
            x = x[:, 1:].mean(dim=1)
        else:
            x = x[:, 0]
        return self.head(self.norm(x))


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        stem_width: int = 48,
        embed_dim: int = 128,
        patch_size: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        feature_size: int = 16,
        mixer: str = "attention",
    ):
        super().__init__()
        if feature_size % patch_size != 0:
            raise ValueError(f"feature_size={feature_size} must be divisible by patch_size={patch_size}")
        if mixer not in {"attention", "no_attention"}:
            raise ValueError(f"Unknown mixer: {mixer}")
        self.depth = depth
        self.mixer = mixer
        self.stem = nn.Sequential(
            ConvBlock(3, stem_width, stride=1, dropout=dropout),
            nn.MaxPool2d(2),
            ConvBlock(stem_width, stem_width * 2, stride=1, dropout=dropout),
        )
        if depth <= 0:
            self.patch_embed = None
            self.pos_embed = None
            self.drop = None
            self.encoder = None
            self.norm = None
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(stem_width * 2),
                nn.Dropout(dropout),
                nn.Linear(stem_width * 2, num_classes),
            )
        else:
            self.patch_embed = PatchEmbed(feature_size, patch_size, stem_width * 2, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            self.drop = nn.Dropout(dropout)
            if mixer == "attention":
                self.encoder = make_encoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
            else:
                self.encoder = make_no_attention_encoder(embed_dim, depth, mlp_ratio, dropout)
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.depth <= 0:
            return self.head(x)
        x = self.patch_embed(x) + self.pos_embed
        x = self.drop(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 108,
        feature_size: int = 8,
        patch_size: int = 2,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        mixer: str = "attention",
        ffn_type: str = "mlp",
    ):
        super().__init__()
        if feature_size % patch_size != 0:
            raise ValueError(f"feature_size={feature_size} must be divisible by patch_size={patch_size}")
        if mixer not in {"attention", "no_attention"}:
            raise ValueError(f"Unknown mixer: {mixer}")
        if ffn_type not in {"mlp", "conv"}:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        if mixer == "no_attention" and ffn_type != "mlp":
            raise ValueError("ffn_type='conv' is only supported with mixer='attention'")

        self.feature_size = feature_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (feature_size // patch_size) * (feature_size // patch_size)
        self.ffn_type = ffn_type

        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.drop = nn.Dropout(dropout)
        if mixer == "attention" and ffn_type == "mlp":
            self.encoder = make_encoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        elif mixer == "attention":
            self.encoder = make_convffn_encoder(
                embed_dim,
                depth,
                num_heads,
                mlp_ratio,
                dropout,
                grid_size=feature_size // patch_size,
            )
        else:
            self.encoder = make_no_attention_encoder(embed_dim, depth, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.global_to_local = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _unfold(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        bsz, channels, height, width = x.shape
        patch = self.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"Feature map {(height, width)} must be divisible by patch_size={patch}")
        grid_h, grid_w = height // patch, width // patch
        x = x.reshape(bsz, channels, grid_h, patch, grid_w, patch)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        return x.view(bsz * patch * patch, grid_h * grid_w, channels), bsz, height, width

    def _fold(self, tokens: torch.Tensor, bsz: int, height: int, width: int) -> torch.Tensor:
        patch = self.patch_size
        grid_h, grid_w = height // patch, width // patch
        tokens = tokens.view(bsz, patch, patch, grid_h, grid_w, self.embed_dim)
        tokens = tokens.permute(0, 5, 3, 1, 4, 2).contiguous()
        return tokens.view(bsz, self.embed_dim, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_rep(x)
        tokens, bsz, height, width = self._unfold(local_feat)
        tokens = tokens + self.pos_embed
        tokens = self.drop(tokens)
        tokens = self.encoder(tokens)
        global_feat = self._fold(self.norm(tokens), bsz, height, width)
        global_feat = self.global_to_local(global_feat)
        return self.fusion(torch.cat([x, global_feat], dim=1))


class ResNetHybridTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        base_width: int = 27,
        embed_dim: int = 108,
        patch_size: int = 2,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        feature_size: int = 8,
        mixer: str = "attention",
        ffn_type: str = "mlp",
    ):
        super().__init__()

        mid_channels = base_width * 4
        out_channels = base_width * 8
        self.in_ch = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(mid_channels, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels, blocks=2, stride=2)

        self.hybrid_block2 = MobileViTBlock(
            in_channels=base_width * 2,
            embed_dim=base_width * 2,
            feature_size=16,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mixer=mixer,
            ffn_type=ffn_type,
        )
        self.hybrid_block3 = MobileViTBlock(
            in_channels=mid_channels,
            embed_dim=embed_dim,
            feature_size=feature_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mixer=mixer,
            ffn_type=ffn_type,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_channels, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.hybrid_block2(x)
        x = self.layer3(x)
        x = self.hybrid_block3(x)
        x = self.layer4(x)
        cnn_feat = self.avgpool(x).flatten(1)
        return self.head(self.dropout(cnn_feat))


def _as_list(value: int | Sequence[int], length: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * length
    values = list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return [int(v) for v in values]


class MobileCIFARBackbone(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        mode: str = "cnn",
        stem_width: int = 32,
        channels: Sequence[int] = (32, 48, 96, 144),
        stage_repeats: Sequence[int] = (1, 1, 1, 1),
        expansion: float = 4.0,
        local_repeats: Sequence[int] = (0, 0, 0),
        cnn_repeats: Sequence[int] = (3, 3, 4),
        cnn_expansion: float = 6.0,
        embed_dims: Sequence[int] = (96, 144, 192),
        depths: Sequence[int] = (2, 2, 3),
        depth: Optional[int] = None,
        num_heads: Sequence[int] = (3, 4, 4),
        mlp_ratio: float = 2.0,
        patch_size: int = 2,
        dropout: float = 0.1,
        head_dim: int = 576,
        mixer: str = "attention",
        ffn_type: str = "mlp",
    ):
        super().__init__()
        if mode not in {"cnn", "hybrid"}:
            raise ValueError(f"Unknown mobile mode: {mode}")
        channels = _as_list(channels, 4, "channels")
        stage_repeats = _as_list(stage_repeats, 4, "stage_repeats")
        local_repeats = _as_list(local_repeats, 3, "local_repeats")
        cnn_repeats = _as_list(cnn_repeats, 3, "cnn_repeats")
        embed_dims = _as_list(embed_dims, 3, "embed_dims")
        depths = _as_list(depth if depth is not None else depths, 3, "depths")
        num_heads = _as_list(num_heads, 3, "num_heads")

        self.mode = mode
        self.stem = ConvBNAct(3, stem_width, kernel_size=3, stride=1)
        self.stage1 = make_mv2_stage(
            stem_width,
            channels[0],
            repeats=stage_repeats[0],
            stride=1,
            expansion=expansion,
        )
        self.down16 = make_mv2_stage(
            channels[0],
            channels[1],
            repeats=stage_repeats[1],
            stride=2,
            expansion=expansion,
        )
        self.local16 = make_optional_mv2_stage(
            channels[1],
            repeats=local_repeats[0],
            expansion=cnn_expansion,
        )
        self.block16 = self._make_middle_block(
            mode,
            channels[1],
            feature_size=16,
            cnn_repeats=cnn_repeats[0],
            cnn_expansion=cnn_expansion,
            embed_dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            dropout=dropout,
            mixer=mixer,
            ffn_type=ffn_type,
        )
        self.down8 = make_mv2_stage(
            channels[1],
            channels[2],
            repeats=stage_repeats[2],
            stride=2,
            expansion=expansion,
        )
        self.local8 = make_optional_mv2_stage(
            channels[2],
            repeats=local_repeats[1],
            expansion=cnn_expansion,
        )
        self.block8 = self._make_middle_block(
            mode,
            channels[2],
            feature_size=8,
            cnn_repeats=cnn_repeats[1],
            cnn_expansion=cnn_expansion,
            embed_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            dropout=dropout,
            mixer=mixer,
            ffn_type=ffn_type,
        )
        self.down4 = make_mv2_stage(
            channels[2],
            channels[3],
            repeats=stage_repeats[3],
            stride=2,
            expansion=expansion,
        )
        self.local4 = make_optional_mv2_stage(
            channels[3],
            repeats=local_repeats[2],
            expansion=cnn_expansion,
        )
        self.block4 = self._make_middle_block(
            mode,
            channels[3],
            feature_size=4,
            cnn_repeats=cnn_repeats[2],
            cnn_expansion=cnn_expansion,
            embed_dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            dropout=dropout,
            mixer=mixer,
            ffn_type=ffn_type,
        )
        self.head = nn.Sequential(
            ConvBNAct(channels[3], head_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    @staticmethod
    def _make_middle_block(
        mode: str,
        channels: int,
        feature_size: int,
        cnn_repeats: int,
        cnn_expansion: float,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        patch_size: int,
        dropout: float,
        mixer: str,
        ffn_type: str,
    ) -> nn.Module:
        if mode == "cnn":
            return make_mv2_stage(
                channels,
                channels,
                repeats=cnn_repeats,
                stride=1,
                expansion=cnn_expansion,
            )
        return MobileViTBlock(
            in_channels=channels,
            embed_dim=embed_dim,
            feature_size=feature_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mixer=mixer,
            ffn_type=ffn_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down16(x)
        x = self.local16(x)
        x = self.block16(x)
        x = self.down8(x)
        x = self.local8(x)
        x = self.block8(x)
        x = self.down4(x)
        x = self.local4(x)
        x = self.block4(x)
        return self.head(x)


class MobileCNNCIFAR(MobileCIFARBackbone):
    def __init__(self, **kwargs):
        kwargs.pop("mode", None)
        super().__init__(mode="cnn", **kwargs)


class MobileViTCIFAR(MobileCIFARBackbone):
    def __init__(self, **kwargs):
        kwargs.pop("mode", None)
        super().__init__(mode="hybrid", **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", config))
    name = model_cfg.pop("name")
    if name == "small_cnn":
        return SmallCNN(**model_cfg)
    if name == "resnet18_cifar":
        return CIFARResNet18(**model_cfg)
    if name == "tiny_vit":
        return TinyViT(**model_cfg)
    if name == "hybrid":
        return HybridCNNTransformer(**model_cfg)
    if name == "resnet_hybrid":
        return ResNetHybridTransformer(**model_cfg)
    if name == "mobile_cnn":
        return MobileCNNCIFAR(**model_cfg)
    if name == "mobilevit_cifar":
        return MobileViTCIFAR(**model_cfg)
    raise ValueError(f"Unknown model name: {name}")


# Training and evaluation utilities.
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch_idx, (images, targets) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), targets) * batch_size
        total_samples += batch_size
        iterator.set_postfix(loss=total_loss / total_samples, acc=total_acc / total_samples)

    return total_loss / max(total_samples, 1), total_acc / max(total_samples, 1)


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_cli_overrides(config: Dict[str, Any], args):
    if args.experiment is not None:
        config["experiment"] = args.experiment

    model_cfg = config.setdefault("model", {})
    train_cfg = config.setdefault("training", {})
    for arg_name, cfg_name in [
        ("depth", "depth"),
        ("patch_size", "patch_size"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            model_cfg[cfg_name] = value
    if args.mixer is not None:
        model_cfg["mixer"] = args.mixer
    if args.ffn_type is not None:
        model_cfg["ffn_type"] = args.ffn_type

    if args.lr is not None:
        train_cfg["lr"] = args.lr
    return config


def main():
    parser = argparse.ArgumentParser(description="Train ResNet/Transformer models on CIFAR-100.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--experiment", default=None, help="Override experiment name in the config.")
    parser.add_argument("--depth", type=int, default=None, help="Override Transformer depth.")
    parser.add_argument("--patch-size", type=int, default=None, help="Override patch size.")
    parser.add_argument("--mixer", choices=["attention", "no_attention"], default=None, help="Override hybrid token mixer.")
    parser.add_argument("--ffn-type", choices=["mlp", "conv"], default=None, help="Override hybrid Transformer FFN type.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    exp_name = config.get("experiment", args.config.stem)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)

    train_cfg = config.get("training", {})
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 2))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(config).to(device)
    params = count_parameters(model)

    val_size = int(train_cfg.get("val_size", 5000))
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=bool(train_cfg.get("augment", True)),
        randaugment=bool(train_cfg.get("randaugment", True)),
        download=args.download,
        val_size=val_size,
        seed=seed,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.1)))
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-2)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-6)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = Path(args.output_dir) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    metrics_path = run_dir / "metrics.csv"
    best_val_acc = -1.0
    best_epoch = 0
    start_time = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds"],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=args.limit_train_batches,
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                max_batches=args.limit_val_batches,
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": time.time() - epoch_start,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"[{exp_name}] epoch {epoch:03d}/{epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "config": config, "epoch": epoch, "val_acc": val_acc},
                    run_dir / "best.pt",
                )

        torch.save(
            {"model": model.state_dict(), "config": config, "epoch": epochs, "val_acc": val_acc},
            run_dir / "last.pt",
        )

    best_path = run_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        max_batches=args.limit_val_batches,
    )

    summary = {
        "experiment": exp_name,
        "config": str(args.config),
        "model": config.get("model", {}).get("name"),
        "parameters": params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_test_acc": test_acc,
        "epochs": epochs,
        "total_seconds": time.time() - start_time,
        "seed": seed,
        "val_size": val_size,
        "device": str(device),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
