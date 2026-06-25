from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


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
    ):
        super().__init__()
        if feature_size % patch_size != 0:
            raise ValueError(f"feature_size={feature_size} must be divisible by patch_size={patch_size}")
        self.depth = depth
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
            self.encoder = make_encoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", config))
    name = model_cfg.pop("name")
    if name == "small_cnn":
        return SmallCNN(**model_cfg)
    if name == "tiny_vit":
        return TinyViT(**model_cfg)
    if name == "hybrid":
        return HybridCNNTransformer(**model_cfg)
    raise ValueError(f"Unknown model name: {name}")
