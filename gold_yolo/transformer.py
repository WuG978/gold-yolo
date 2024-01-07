# 2023.09.18-Changed for transformer of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# This file editing from https://github.com/hustvl/TopFormer/blob/main/mmseg/models/backbones/topformer.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.cnn import ConvModule
from .layers import Conv2d_BN, DropPath, h_sigmoid


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 将输入从 in_features 维线性投影到 hidden_features 维
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        # depth-wise 卷积,对每个输入通道分别做卷积,增加非线性
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )
        self.act = act_layer()
        # 将 hidden_features 维的特征投影回 out_features 维
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        # 正则化
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        attn_ratio=4,
        activation=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.num_heads = num_heads  # the number of attention heads
        self.scale = key_dim**-0.5  # scaling factor
        self.key_dim = key_dim  # dimension of head
        self.nh_kd = nh_kd = (
            key_dim * num_heads
        )  # num_head * key_dim = the total dimension of Q,V
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        # print("org shape x: ", get_shape(x))

        qq = (
            self.to_q(x)
            .reshape(B, self.num_heads, self.key_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)  # shape = (B, num_heads, H*W, H*W)
        attn = attn.softmax(dim=-1)  # dim=-1 means applying softmax along the last dim

        xx = torch.matmul(attn, vv)  # shape = (B, num_heads, H*W, d)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class SimpleLinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        print("BCHW: ", x.shape)
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        q = q + 1e-6
        k = k + 1e-6

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)

        q = q / q_norm
        k = k / k_norm

        q, k, v = (
            rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v]
        )

        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b n c -> b c n")
        return x


class FocusedLinearAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        focusing_factor=3,
        kernel_size=5,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            groups=head_dim,
            padding=kernel_size // 2,
        )
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(
            torch.zeros(size=(1, window_size[0] * window_size[1], dim))
        )
        print(
            "Linear Attention window{} f{} kernel{}".format(
                window_size, focusing_factor, kernel_size
            )
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) maskr with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        print("BCHW: ", x.shape, x.device)
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, HW, C]
        B, N, C = get_shape(x)  # B=B, N=HW, C=C
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # qkv [3, B, N, C]
        q, k, v = qkv.unbind(0)  # q, k, v [B, N, C]
        # print(
        #     "\nk shape:",
        #     k.shape,
        #     "position encoding shape:",
        #     self.positional_encoding.shape,
        # )
        # k = k + self.positional_encoding
        focusing_factor = self.focusing_factor  # 3
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6  # q = ReLU(q)
        k = kernel_function(k) + 1e-6  # q = ReLU(k)
        scale = nn.Softplus()(self.scale)
        q = q / scale  # q = ReLU(q) / scale
        k = k / scale  # k = ReLU(k) / scale
        q_norm = q.norm(dim=-1, keepdim=True)  # q_norm = ||ReLU(q)/scale||
        k_norm = k.norm(dim=-1, keepdim=True)  # k_norm = ||ReLU(k)/scale||
        if float(focusing_factor) <= 6:
            q = q**focusing_factor  # q = (ReLU(q) / scale)**3
            k = k**focusing_factor  # k = (ReLU(k) / scale)**3
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        # q = (ReLU(q) / scale)**3 / ||(ReLU(q) / scale)**3|| * ||ReLU(q)/scale||
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        # k = (ReLU(k) / scale)**3 / ||(ReLU(k) / scale)**3|| * ||ReLU(k)/scale||
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (
            rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v]
        )  # q, k, v [B*num_heads, N, C/num_heads]
        print("After processing q, k, v shape: ", q.shape, v.shape, q.shape)
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]  # i = N, j = N, c = C/num_heads, d = C/num_heads
        print("i = ", i, "j = ", j, "c = ", c, "d = ", d)

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (w h) c -> b c w h", b=B, c=self.dim, w=num, h=num)
        print("final x shape: ", x.shape)
        return x


class top_Block(nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_cfg=dict(type="BN2d", requires_grad=True),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            norm_cfg=norm_cfg,
        )
        # self.attn = SimpleLinearAttention(dim, num_heads=num_heads)
        # self.attn = FocusedLinearAttention(
        #     dim,
        #     window_size=(10, 10),
        #     num_heads=num_heads,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     attn_drop=0,
        #     proj_drop=drop,
        #     focusing_factor=3,
        #     kernel_size=5,
        # )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            norm_cfg=norm_cfg,
        )

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(
        self,
        block_num,
        embedding_dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_layer=nn.ReLU6,
    ):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                top_Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_cfg=norm_cfg,
                    act_layer=act_layer,
                )
            )

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode="onnx"):
        super().__init__()
        self.stride = stride
        if pool_mode == "torch":
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == "onnx":
            self.pool = onnx_AdaptiveAvgPool2d

    def forward(self, inputs):
        # Pooling output size is same as the feature map with min resolution
        B, C, H, W = get_shape(inputs[-1])
        # This calculation approach ensures that incomplete pooling windows
        # are taken into account when computing the output size.
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, "pool"):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return torch.cat(out, dim=1)


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type="BN", requires_grad=True),
        activations=None,
        global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg

        if not global_inp:
            global_inp = inp

        self.local_embedding = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.global_embedding = ConvModule(
            global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.global_act = ConvModule(
            global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(
                self.act(global_act), size=(H, W), mode="bilinear", align_corners=False
            )
            global_feat = F.interpolate(
                global_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        out = local_feat * sig_act + global_feat
        return out
