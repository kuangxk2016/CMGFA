# ---------------------------------------------
#  Modified by Xinkai Kuang
# ---------------------------------------------

#import ipdb
import torch
# import torch.nn.functional as F
#
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model

from einops import rearrange
# from functools import partial
from torch import nn, einsum


class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Agg_0(nn.Module):
    def __init__(self, seg_dim):
        super().__init__()
        self.conv = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1) #in_channels, out_channels, kernel_size, stride, padding
        self.norm = nn.LayerNorm(seg_dim)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.act(self.norm(x.reshape(b, c, -1).permute(0, 2, 1)))
        # x = self.act(self.norm(x))

        return x

'''
Hardswish/swish
模型性能会比ReLU性能好
但是计算量更大
'''

class Aggregator(nn.Module):
    def __init__(self, dim, seg=5):
        super().__init__()
        self.dim = dim
        self.seg = seg
        # self.dim_N = dim_N  #用来决定是q还是kv

        seg_dim = self.dim // self.seg
        seg_dim0 = self.dim-seg_dim*(self.seg-1)

        self.norm0 = nn.BatchNorm2d(seg_dim) #SyncBatchNorm
        self.act0 = nn.Hardswish()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(seg_dim)
        self.act1 = nn.Hardswish()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.BatchNorm2d(seg_dim)
        self.act2 = nn.Hardswish()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.BatchNorm2d(seg_dim)
        self.act3 = nn.Hardswish()

        self.agg0 = Agg_0(seg_dim0)


    def forward(self, x, size): #, num_head
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        B, C, H, W = x.shape
        seg_dim = self.dim // self.seg
        seg_dim0 = self.dim-seg_dim*(self.seg-1)
        split_seg = [seg_dim]*(self.seg-1)
        split_seg.append(seg_dim0)
        x = x.split(split_seg, dim=1)

        x_local = x[4]#.reshape(self.dim_N, B//self.dim_N, seg_dim, H, W).permute(1,0,2,3,4).reshape(B//self.dim_N, self.dim_N*seg_dim, H, W)
        x_local = self.agg0(x_local)
        x_local = x_local.permute(0,2,1).reshape(B,seg_dim0,H,W)

        x0 = self.act0(self.norm0(x[0]))
        x1 = self.act1(self.norm1(self.agg1(x[1])))
        x2 = self.act2(self.norm2(self.agg2(x[2])))
        x3 = self.act3(self.norm3(self.agg3(x[3])))

        x = torch.cat([x_local, x0, x1, x2, x3], dim=1)

        # C = C // (self.seg) * (self.seg-1)
        # x = x.reshape(self.dim_N, B//self.dim_N, num_head, C//num_head, H*W).permute(0, 1, 2, 4, 3)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        return x#, x_local


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h}  # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch,
                                 kernel_size=(cur_window, cur_window),
                                 padding=(padding_size, padding_size),
                                 dilation=(dilation, dilation),
                                 groups=cur_head_split * Ch,
                                 )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)

        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):

        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W

        # Convolutional relative position encoding.
        q_img = q  # Shape: [B, h, H*W, Ch].
        v_img = v  # Shape: [B, h, H*W, Ch].

        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)  # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)  # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q_img * conv_v_img

        return EV_hat_img


class CrossAtt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.N = 2 #用来决定是kv还是/  2是kv，默认1是q
        self.kv = nn.Linear(dim, dim * self.N, bias=qkv_bias) #生成K和V
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator_q = Aggregator(dim=dim, seg=5)
        self.aggregator_kv = Aggregator(dim=dim, seg=5)


        trans_dim = dim // 5 * 4
        self.crpe = ConvRelPosEnc(Ch=trans_dim // num_heads, h=num_heads, window={3: 2, 5: 3, 7: 3})

    def forward(self, x, y, size): #x是f*，y是f^lidar
        ###############size可以不需要
        B, C, H, W  = x.shape

        # Q, K, V.
        kv = self.kv(x).reshape(B, H, W, 2, C).permute(3, 0, 1, 2, 4).reshape(2*B, H*W, C)
        q = y.reshape(B, H, W, C).reshape(B, H*W, C)
        kv, x_agg0 = self.aggregator_kv(kv, size, self.num_heads)
        q, q_agg0 = self.aggregator_q(q, size, self.num_heads)
        k, v = kv[0], kv[1]

        # att
        k_softmax = k.softmax(dim=2)   #k
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v) # K*V
        eff_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v) # Q*(K*V)
        crpe = self.crpe(q, v, size=size)  #卷积相对位置编码
        # Merge and reshape.
        x = self.scale * eff_att + crpe
        x = x.transpose(1, 2).reshape(B, H*W, C//5*4)
        x = torch.cat([x, x_agg0+q_agg0], dim=-1)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x




class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x

