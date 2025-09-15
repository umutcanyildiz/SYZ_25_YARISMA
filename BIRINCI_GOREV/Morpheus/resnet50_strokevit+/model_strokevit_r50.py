# model_strokevit_r50.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class StrokeViT_R50(nn.Module):
    """
    ResNet50 (features_only, layer3/stride16) -> 24x24x1024 @ 384x384 input
    + ViT tarzı Transformer Encoder (CLS token, patch=1 benzeri tokenizasyon).
    Öğrenilebilir 2D pozisyon gömme; farklı HxW gelirse interpolasyonla uyarlanır.
    """
    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 384,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: int = 4,
        drop: float = 0.1,
    ):
        super().__init__()
        # ResNet50 backbone (layer3 çıkışı: C=1024, stride=16)
        self.backbone = timm.create_model(
            "resnet50", pretrained=True, features_only=True, out_indices=(2,)
        )
        c_backbone = self.backbone.feature_info.channels()[-1]  # 1024
        self.img_size = img_size
        self.grid_init = img_size // 16  # 384 -> 24

        # Token (C->D) projeksiyon
        self.proj = nn.Linear(c_backbone, d_model)

        # CLS ve pozisyon embedding (öğrenilebilir)
        tokens = self.grid_init * self.grid_init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Encoder (batch_first)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Sınıflandırma başı
        self.head = nn.Linear(d_model, 1 if num_classes == 1 else num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _pos_embed(self, x_tokens, H, W):
        """Pozisyon embed'ini (token kısmını) HxW'ye interpolate et, sonra CLS ekle."""
        B, N, D = x_tokens.shape
        pe = self.pos_embed  # (1, 1+T, D)
        cls_pe = pe[:, :1, :]
        tok_pe = pe[:, 1:, :]  # (1, T, D)
        t0 = int((tok_pe.shape[1]) ** 0.5)
        tok_pe_2d = tok_pe.reshape(1, t0, t0, D).permute(0, 3, 1, 2)  # (1,D,t0,t0)
        tok_pe_2d = F.interpolate(tok_pe_2d, size=(H, W), mode="bicubic", align_corners=False)
        tok_pe = tok_pe_2d.permute(0, 2, 3, 1).reshape(1, H * W, D)   # (1, H*W, D)
        pe_resized = torch.cat([cls_pe, tok_pe], dim=1)               # (1, 1+H*W, D)
        return x_tokens + pe_resized[:, : x_tokens.size(1), :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features: B x 1024 x H x W (genelde H=W=img/16)
        feats = self.backbone(x)[0]
        B, C, H, W = feats.shape

        # Token dizisi: B x (H*W) x C -> proje B x N x D
        x_tok = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_tok = self.proj(x_tok)

        # CLS ekle + pozisyon
        cls = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat([cls, x_tok], dim=1)  # (B, 1+N, D)
        x_tok = self._pos_embed(x_tok, H, W)

        # Transformer + CLS
        x_tok = self.transformer(x_tok)
        cls_out = self.norm(x_tok[:, 0])

        # Logits
        return self.head(cls_out)
