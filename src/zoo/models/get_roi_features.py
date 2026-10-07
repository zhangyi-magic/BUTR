"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .utils import get_activation
import torch.nn as nn
import os
import copy


# from .ops.modules import MSDeformAttn


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            n_levels=4, n_heads=8, n_points=4
    ):
        super().__init__()
        self.normalize_before = normalize_before
        # self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


# transformer
class AxisPermutedEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            n_levels=4, n_heads=8, n_points=4
    ):
        super().__init__()
        self.normalize_before = normalize_before
        # self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, q, k, v, src_mask=None) -> torch.Tensor:
        src = residual = v
        if self.normalize_before:
            src = self.norm1(src)

        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class AxisPermutedEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(AxisPermutedEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, glob_pos_embeds=None) -> torch.Tensor:
        output = src
        B, C, _, _ = glob_pos_embeds.shape
        glob_pos_embeds = glob_pos_embeds.reshape(B, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.reshape(C, -1).permute(1, 0).unsqueeze(0)
        for layer in self.layers:
            q = k = self.with_pos_embed(output, glob_pos_embeds + pos_embed)
            output = layer(q, k, output, src_mask=src_mask)

            output = output.permute(1, 0, 2).contiguous()
            q = k = self.with_pos_embed(output, (glob_pos_embeds + pos_embed).permute(1, 0, 2).contiguous())
            output = layer(q, k, output, src_mask=src_mask)
            output = output.permute(1, 0, 2).contiguous()

        if self.norm is not None:
            output = self.norm(output)

        return output


class WindowProcessor(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=1024, num_layers=1, dropout=0.0, activation="relu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        encoder_layer = AxisPermutedEncoderLayer(
            self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.window_encoder = AxisPermutedEncoder(encoder_layer, self.num_layers)

    @torch.no_grad()
    def _prepare_windows(self, features, mask, n):
        """
        功能不变的窗口划分 + defe_mask 生成（max_pool2d + >0）。
        - features: [B, C, H_feat, W_feat]
        - mask:     [B, 1, H_mask, W_mask]  (defe_feature_filtered)
        - return:
            windows:   [B, n, n, C, win_h, win_w]
            defe_mask: [B, n, n] bool
        """
        B, C, H_feat, W_feat = features.shape
        H_mask, W_mask = mask.shape[-2:]

        # 你的原实现依赖：H_feat % n == 0 & W_feat % n == 0
        assert H_feat % n == 0 and W_feat % n == 0

        win_h = H_feat // n
        win_w = W_feat // n

        # 1) features -> windows (view + permute，不拷贝数据)
        # [B, C, H, W] -> [B, n, n, C, win_h, win_w]
        windows = features.view(B, C, n, win_h, n, win_w).permute(0, 2, 4, 1, 3, 5).contiguous()

        # 2) mask -> defe_mask via max_pool，保持原语义：窗口内只要有任意 >0 就 True
        # 注意：mask 分辨率可能不同于 feature 分辨率，按你原公式推 kernel
        # kernel_h = (H_mask / H_feat) * (H_feat/n) = H_mask / n
        # kernel_w = (W_mask / W_feat) * (W_feat/n) = W_mask / n
        # 但用整数运算要稳：保证可整除，否则退回到自适应池化
        if (H_mask % n == 0) and (W_mask % n == 0):
            kernel_h = H_mask // n
            kernel_w = W_mask // n
            pooled = F.max_pool2d(mask.float(), kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
            # pooled: [B,1,n,n]
        else:
            # 兜底：仍然“窗口内 max>0”语义，用 adaptive_max_pool2d 对齐到 [n,n]
            pooled = F.adaptive_max_pool2d(mask.float(), output_size=(n, n))

        defe_mask = pooled.squeeze(1) > 0  # [B,n,n] bool
        return windows, defe_mask, win_h, win_w

    def _get_rel_embedding(self, window_size):
        h, w = window_size
        coords = self._get_relative_coords(h, w).to(self.rel_pos_encoder[0].weight.device)
        # [h,w,2] -> MLP -> [h,w,C] -> [C,h,w]
        return self.rel_pos_encoder(coords).permute(2, 0, 1)

    @staticmethod
    def _get_relative_coords(h, w):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        # 防止 w/h=1 的除零
        wx = max(w - 1, 1)
        hy = max(h - 1, 1)
        return torch.stack([grid_x / wx, grid_y / hy], dim=-1)

    def _encode_features(self, features, rel_pos_embed, glob_pos_embeds):
        """
        features: [N, C, win_h, win_w]
        glob_pos_embeds: [N, C, win_h, win_w]
        rel_pos_embed: [C, win_h, win_w]
        """
        N, C, h, w = features.shape
        # [N, C, h, w] -> [N, L, C]
        x = features.view(N, C, -1).permute(0, 2, 1).contiguous()
        x = self.window_encoder(x, pos_embed=rel_pos_embed, glob_pos_embeds=glob_pos_embeds)
        # back: [N, L, C] -> [N, C, h, w]
        return x.permute(0, 2, 1).contiguous().view(N, C, h, w)

    def forward(self, backbone_memory, defe_feature_filtered, n, glob_pos_embed):
        """
        高性能并行版（功能不变）：
        - 只处理 defe_mask=True 的窗口
        - 对这些窗口做 window_encoder
        - 将输出以 += 的方式加回对应 patch
        """
        B, C, H, W = backbone_memory.shape
        assert H % n == 0 and W % n == 0, "H and W must be divisible by n"

        # 1) 划窗 + mask
        windows, defe_mask, win_h, win_w = self._prepare_windows(backbone_memory, defe_feature_filtered, n)
        # windows: [B,n,n,C,win_h,win_w]
        # defe_mask: [B,n,n]

        # 2) 拉平所有 valid windows（跨 batch 合并）
        idx = torch.nonzero(defe_mask, as_tuple=False)  # [N_total, 3] => (b,i,j)
        if idx.numel() == 0:
            # 保持你原逻辑的 assert 行为
            raise AssertionError("No valid windows found")

        b_idx = idx[:, 0]
        i_idx = idx[:, 1]
        j_idx = idx[:, 2]

        # 3) gather windows -> [N_total, C, win_h, win_w]
        sel_windows = windows[b_idx, i_idx, j_idx]  # advanced indexing, GPU 并行 gather

        # 4) gather glob_pos_embeds patch（同样并行）
        # glob_pos_embed: 你的上游是 build_2d_sincos_position_embedding(W,H)->permute->view(-1,H,W)
        # 通常是 [C,H,W]
        if glob_pos_embed.dim() == 4:
            # [1,C,H,W] -> [C,H,W]
            glob_pos_embed_ = glob_pos_embed[0]
        else:
            glob_pos_embed_ = glob_pos_embed
        assert glob_pos_embed_.shape == (C, H, W), f"glob_pos_embed shape mismatch: {glob_pos_embed_.shape}"

        pos_windows = glob_pos_embed_.view(C, n, win_h, n, win_w).permute(1, 3, 0, 2, 4).contiguous()
        # [n,n,C,win_h,win_w]
        sel_pos = pos_windows[i_idx, j_idx]  # [N_total,C,win_h,win_w]

        # 5) window encoder（GPU 并行最核心）
        rel_pos_embed = self._get_rel_embedding((win_h, win_w)).to(backbone_memory.device)
        encoded = self._encode_features(sel_windows, rel_pos_embed, sel_pos)  # [N_total,C,win_h,win_w]

        # 6) scatter-add 回原图（等价于逐窗 reconstructed += feats）
        # 用 view 把 reconstructed 改成 windows 视图，再 index_put_ accumulate=True 做 +=
        reconstructed = backbone_memory  # 功能不变：在原特征上加增强
        recon_view = reconstructed.view(B, C, n, win_h, n, win_w).permute(0, 2, 4, 1, 3, 5).contiguous()
        # recon_view: [B,n,n,C,win_h,win_w]
        recon_view.index_put_((b_idx, i_idx, j_idx), encoded, accumulate=True)

        # 还原 [B,C,H,W]
        reconstructed = recon_view.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

        return reconstructed, defe_mask
