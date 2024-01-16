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

    def forward(self, x):
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

class Attention_Flatten(torch.nn.Module):
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

        # self.dwc = nn.Conv2d(in_channels=int(attn_ratio*key_dim), out_channels=int(attn_ratio*key_dim), kernel_size=5, groups=num_heads, padding=5//2)
        self.dwc = Conv2d_BN(int(attn_ratio*key_dim), int(attn_ratio*key_dim), ks=5, groups=num_heads, pad=5//2, norm_cfg=norm_cfg)

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, nh_kd)))

        self.proj = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        B, C, W, H = get_shape(x)

        qq = (
            self.to_q(x)
            .reshape(B, self.num_heads * self.key_dim, H * W)
            .permute(0, 2, 1)
        )  # [B, H*W, num_heads*key_dim]
        kk = self.to_k(x).reshape(B, self.nu m_heads * self.key_dim, H * W).permute(0, 2, 1)  # [B, num_heads*key_dim, H*W]
        vv = self.to_v(x).reshape(B, self.num_heads * self.d, H * W).permute(0, 2, 1)  # [B, H*W, attn_ratio*key_dim*num_heads]

        focusing_factor = 3
        kernel_function = nn.ReLU()
        qq = kernel_function(qq) + 1e-6
        kk = kernel_function(kk) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = qq / scale
        k = kk / scale
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
        
        # Rearrange Q, K, V separately to accommodate the calculation of multi-head attention
        q, k, v = (
            rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, vv]
        )  # q, k: [B*num_heads, H*W, key_dim]; v: [B*num_heads, H*W, attn_ratio*key_dim]

        # i and j represent the length of the sequence (i.e., the number of attention heads) of Q and K, respectively
        # c is the number of channels within each header (C/num_heads, where C is the number of channels of the original input)
        # d is the number of channels of V, which is the same as c because V is the output of Q and K.
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]
        # attention score/weight computation
        # k.sum(dim=1): sum K in the second dimension is equivalent to averaging the keys for each head
        # then perform a dot product with Q to obtain an attention fraction tensor z
        z = 1 / (W * H + torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)  # [B*num_heads, N]
        # z = torch.ones((B*self.num_heads, H*W), device='cuda', requires_grad=True)
        if i * j * (c + d) > c * d * (i + j):
            # print("kv: ", (j*j*(c+d))/(c*d*(i+j)))
            kv = torch.einsum("b j c, b j d -> b c d", k, v)  # [B*num_heads, key_dim, attn_ratio*key_dim]
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)  # [B*num_heads, H*W, attn_ratio*key_dim]
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)  # [B*num_heads, H*W, H*W]
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)  # [B*num_heads, H*W, attn_ratio*key_dim]

        feature_map = rearrange(v, "b (w h) c -> b c w h", w=W, h=H)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        xx = x.permute(0, 2, 1).reshape(B, self.dh, W, H)
        xx = self.proj(xx)

        return xx
    

class FocusedLinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        proj_drop=0.0,
        focusing_factor=3,
        kernel_size=5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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

    def forward(self, x, mask=None):
        # flatten: [B, C, W, H] -> [B, C, WH]
        # transpose: [B, C, WH] -> [B, WH, C]
        B, C, W, H = x.shape
        N = W * H
        x = x.flatten(2).transpose(1, 2)  # [B, C, W, H] -> [B, N, C]
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # qkv [3, B, N, C]
        q, k, v = qkv.unbind(0)  # q, k, v [B, N, C]
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
        # Rearrange Q, K, V separately to accommodate the calculation of multi-head attention
        q, k, v = (
            rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v]
        )  # q, k, v [B*num_heads, N, C/num_heads]
        
        # i and j represent the length of the sequence (i.e., the number of attention heads) of Q and K, respectively
        # c is the number of channels within each header (C/num_heads, where C is the number of channels of the original input)
        # d is the number of channels of V, which is the same as c because V is the output of Q and K.
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]
        # attention score/weight computation
        # k.sum(dim=1): sum K in the second dimension is equivalent to averaging the keys for each head
        # then perform a dot product with Q to obtain an attention fraction tensor z
        # z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)  # [B*num_heads, N]
        z = 1 / (W * H + torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)

        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)  # [B*num_heads, C/num_heads, C/num_heads]
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)  # [B*num_heads, N, C/num_heads]
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)  # [B*num_heads, N, N]
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)  # [B*num_heads, N, C/num_heads]

        feature_map = rearrange(v, "b (w h) c -> b c w h", w=W, h=H)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (w h) c -> b c w h", b=B, c=self.dim, w=W, h=H)
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

        # self.attn = Attention(
        #     dim,
        #     key_dim=key_dim,
        #     num_heads=num_heads,
        #     attn_ratio=attn_ratio,
        #     activation=act_layer,
        #     norm_cfg=norm_cfg,
        # )
        self.attn = Attention_Flatten(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            norm_cfg=norm_cfg,
        )
        # self.attn = FocusedLinearAttention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=True,
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
