#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from collections import OrderedDict
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from surkit.nn.jax.activations import GELU
from surkit.nn.jax.cnn import Conv2d
from surkit.nn.jax.pooling import _pair
from surkit.nn.jax.transformers import AFNO2D

#todo: extract _pair to utils
#todo: apply drop

def trunc_normal(key, shape, mean=0.0, std=0.02, a=-2.0, b=2.0):
    x = jax.random.normal(key, shape)
    lower = 0.0 + a * std
    upper = 0.0 + b * std
    x = jax.numpy.clip(x, lower, upper)
    return x

# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor
#
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None, scale_by_keep=True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    in_features: int
    hidden_features: int = None
    out_features: int = None
    act_layer: nn.Module = GELU
    drop: float = 0.
    def setup(self):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        self.fc1 = nn.Dense(hidden_features, kernel_init=trunc_normal)
        self.act = self.act_layer()
        self.fc2 = nn.Dense(out_features, kernel_init=trunc_normal)
        # self.dropout = nn.Dropout(self.drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        return x


class Block(nn.Module):
    mixing_type: str
    dim: int
    mlp_ratio: float = 4.
    drop: float = 0.
    drop_path: float = 0.
    act_layer: nn.Module = GELU
    norm_layer: nn.Module = nn.LayerNorm
    double_skip: bool = True
    num_blocks: int = 8
    sparsity_threshold: float = 0.01
    hard_thresholding_fraction: float = 1.
    hidden_size_factor: int = 1
    def setup(self):
        self.norm1 = self.norm_layer(self.dim)

        # if mixing_type == 'afno':
        #       self.filter = AFNO2D(hidden_size=hidden_size, num_blocks=fno_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)
        # elif mixing_type == "bfno":
        #     self.filter = BFNO2D(hidden_size=768, num_blocks=8, hard_thresholding_fraction=1)
        # elif mixing_type == "sa":
        #     self.filter = SelfAttention(dim=768, h=14, w=8)
        # if mixing_type == "gfn":
        #     self.filter = GlobalFilter(dim=768, h=14, w=8)
        # elif mixing_type == "ls":
        #     self.filter = AttentionLS(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe=False, nglo=1, dp_rank=2, w=2)

        if self.mixing_type == 'afno':
            self.filter = AFNO2D(self.dim, self.num_blocks, self.sparsity_threshold, self.hard_thresholding_fraction, self.hidden_size_factor)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = self.norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop,
        )
    def __call__(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        # x = self.drop_path(x)
        return x


class PatchEmbed(nn.Module):
    img_size: int = 128
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 768
    def setup(self):
        img_size = _pair(self.img_size)
        patch_size = _pair(self.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.proj = Conv2d(self.in_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x):
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        return x


class AFNONet(nn.Module):
    img_size: int = 128
    patch_size: int = 16
    mixing_type: str = 'afno'
    in_channels: int = 1
    out_channels: int = 1
    embed_dim: int = 768
    depth: int = 12
    mlp_ratio: float = 4.
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    num_blocks: int = 8
    sparsity_threshold: float = 0.01
    hard_thresholding_fraction: float = 1.
    hidden_size_factor: int = 1
    def setup(self):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        img_size = _pair(self.img_size)
        patch_size = _pair(self.patch_size)
        assert (
                img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), f"img_size {img_size} should be divisible by patch_size {patch_size}"

        self.img_size_ = img_size
        self.patch_size_ = patch_size
        self.num_features = self.embed_dim

        norm_layer = partial(nn.LayerNorm)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.param("pos_embed", trunc_normal, (1, num_patches, self.embed_dim))
        # self.pos_drop = nn.Dropout(rate=self.drop_rate)

        self.h = img_size[0] // patch_size[0]
        self.w = img_size[1] // patch_size[1]

        self.blocks = [
            Block(
                mixing_type=self.mixing_type,
                dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=self.drop_path_rate,
                norm_layer=norm_layer,
                double_skip=True,
                num_blocks=self.num_blocks,
                sparsity_threshold=self.sparsity_threshold,
                hard_thresholding_fraction=self.hard_thresholding_fraction,
                hidden_size_factor=self.hidden_size_factor,
            )
            for _ in range(self.depth)
        ]

        self.head = nn.Dense(
            self.out_channels * patch_size[0] * patch_size[1],
            use_bias=False,
            kernel_init=trunc_normal,
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        # x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def __call__(self, x, train=False):
        x = self.forward_features(x)
        x = self.head(x)

        out = jax.numpy.reshape(x, list(x.shape[:-1]) + [self.patch_size_[0], self.patch_size_[1], -1])
        out = jax.numpy.transpose(out, (0, 5, 1, 3, 2, 4))
        out = jax.numpy.reshape(out, list(out.shape[:2]) + [self.img_size_[0], self.img_size_[1]])

        return out