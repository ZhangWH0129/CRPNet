import torch
import torch.nn as nn
import torch.nn.functional as nnf
import Module.Config as Config


import numpy as np
from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np



class CA(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(CA, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            CA(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)
class Mlp(nn.Module):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    # print('window_partition(B, H, W, L, C):', x.shape)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape  # (num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k, v = qkv2[1], qkv2[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.MLP = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])

        max_pool = self.MLP(max_pool)
        avg_pool = self.MLP(avg_pool)

        out = max_pool + avg_pool
        out = self.sigmoid(out).view([b, c, 1, 1, 1])
        return out * x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_diy=96):
        super().__init__()
        dim = dim_diy
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim_diy)
        # print('dim', dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_diy)
        mlp_hidden_dim = int(dim_diy * mlp_ratio)
        self.mlp = Mlp(in_features=dim_diy, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv_block = CAB(num_feat=self.dim, compress_ratio=3, squeeze_factor=16)
        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, y, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        y = self.norm1(y)
        x = x.view(B, H, W, T, C)
        y = y.view(B, H, W, T, C)## 1，20，24，20，128

        # Conv_X 通道数不变   1，128，20，24，20
        conv_x = self.conv_block(x.permute(0, 4, 1, 2, 3))
        # to 1 20 24 20 128  /to 1 9600 128
        conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous().view(B, H * W * T, C)
        conv_y = self.conv_block(y.permute(0, 4, 1, 2, 3))
        conv_y = conv_y.permute(0, 2, 3, 4, 1).contiguous().view(B, H * W * T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            shifted_y = torch.roll(y, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_y = y
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows_xy = self.attn(x_windows, y_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C
        #attn_windows_yx = self.attn(x_windows, y_windows, mask=attn_mask)

        # merge windows
        attn_windows_x = attn_windows_xy.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows_x, self.window_size, Hp, Wp, Tp)  # B H' W' L' C
        #attn_windows_y = attn_windows_yx.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        #shifted_y = window_reverse(attn_windows_y, self.window_size, Hp, Wp, Tp)  # B H' W' L' C
        # shifted_y = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            #y = torch.roll(shifted_y, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),dims=(1, 2, 3))
        else:
            x = shifted_x
            #y = shifted_y

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()
            #y = y[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)
        #y = y.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x) + conv_x * 0.01 + conv_y * 0.01
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x






class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 dim_diy=96):
        super().__init__()
        dim = dim_diy
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                dim_diy=dim_diy)
            for i in range(depth)])

    def forward(self, x, y, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        # print('H W T:', H, W, T)
        # print('windows_size:', self.window_size[0], self.window_size[1], self.window_size[2])
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        # print('Hp, Wp, Tp:', Hp, Wp, Tp)
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, y, attn_mask)
        return x, H, W, T, x, H, W, T

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 dim_diy=96):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dim_diy = dim_diy

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            #self.pos_embd = SinPositionalEncoding3D(96).cuda()#SinusoidalPositionEmbedding().cuda()
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(1):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=1 if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               dim_diy=dim_diy)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = dim_diy

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(dim_diy)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, y):
        """Forward function."""
        # x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        # print('The shape of patch nums', Wh, Ww, Wt)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
            y = y.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        for i in range(1):  # num_layers  = len(depths)  = 4
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, y, Wh, Ww, Wt)
            # print('ok')
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{0}')
                # print(norm_layer)
                # norm_layer = nn.LayerNorm(self.dim_diy, eps=1e-05, elementwise_affine=True)
                x_out = norm_layer(x_out)
                # print('***', x_out.shape)
                # print('The num_features:', self.num_features)

                out = x_out.view(-1, H, W, T, self.num_features).permute(0, 4, 1, 2, 3).contiguous()
        return out

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

class LWCA(nn.Module):
    def __init__(self, config, dim_diy):
        super(LWCA, self).__init__()
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dim_diy=dim_diy
                                           )

    def forward(self, x, y):
        moving_fea_cross = self.transformer(x, y)
        return moving_fea_cross


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)




class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out




class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)




class Encoder5(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder5, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(8 * c, 16 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(16 * c)
        )




    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)
        return [out0, out1, out2, out3, out4]
# class CCABlock(nn.Module):
#     def __init__(self, inchannel=8, outchannel=16 ):
#         class CCABlock(nn.Module):
#             def __init__(self, inchannel=8, outchannel=16):


class Spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spacial_attention, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              bias=False
                              )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class CSABlock(nn.Module):
    def __init__(self, inchannel=8, outchannel=16 ):
        super(CSABlock, self).__init__()
        self.SA_F2M = Spacial_attention(kernel_size= 7)
        self.SA_M2F = Spacial_attention(kernel_size= 7)

    def forward(self, fix, move):
        Copy_fix = fix
        Copy_move = move

        SA_fix = self.SA_F2M(Copy_fix)
        SA_move = self.SA_M2F(Copy_move)
        fix = Copy_move * SA_fix + Copy_fix
        move = Copy_fix * SA_move + Copy_move
        return fix, move

class CSA_ADDBlock(nn.Module):
    def __init__(self, inchannel=8, outchannel=16 ):
        super(CSA_ADDBlock, self).__init__()
        self.SA_F2M = Spacial_attention(kernel_size= 7)
        self.SA_M2F = Spacial_attention(kernel_size= 7)
        self.conv1 = nn.Conv3d(inchannel*2,inchannel,1)
        #self.sig = nn.Sigmoid()


    def forward(self, fix, move):
        Copy_fix = fix
        Copy_move = move

        SA_fix = self.SA_F2M(Copy_fix)
        SA_move = self.SA_M2F(Copy_move) #(1,1,10,12,10)
        Cat = torch.cat((fix, move), dim =1)
        Cat = self.conv1(Cat)
        #Cat_Sig = self.sig(Cat)

        #ADD = Copy_move * Cat_Sig + Cat + Copy_move * Cat_Sig

        fix = Copy_move * SA_fix + Copy_fix
        move = Copy_fix * SA_move + Copy_move
        return fix , move , Cat




class FeaFuse(nn.Module):
    def __init__(self, input=16, output=16, r=4):
        super(FeaFuse, self).__init__()
        inter_channels = int(input// r)

        self.local_att = nn.Sequential(
            nn.Conv3d(input, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, input, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            #nn.BatchNorm3d(input),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv3d(input, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, input, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            # nn.BatchNorm3d(input),
        )

        self.conv1 = nn.Conv3d(2 * input, output, 1 )
        self.conv1_2 = nn.Conv3d(input, output, 1 )
        self.conv3 = nn.Conv3d(input, output, 3 ,1, 1)

    def forward(self, Corr , SA):
        Cat = torch.cat((Corr, SA), dim=1)
        Cat = self.conv1(Cat)
        Add = Corr + SA
        att1 = self.local_att(Cat)
        att2 = self.local_att2(Add)
        Cat = att1* Cat + Cat
        Add = att2 * Add + Add
        # Add = self.conv1_2(self.conv3(Add))

        return 0.5*Cat + 0.5*Add






class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x
class CRPNet(nn.Module):
    def __init__(self, inshape=(160, 192, 160), flow_multiplier=1., in_channel=1, channels=16):
        super(CRPNet, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels

        self.encoder_moving = Encoder5(in_channel=in_channel, first_out_channel=c)
        self.encoder_fixed = Encoder5(in_channel=in_channel, first_out_channel=c)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(5):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

        config = Config.get_ZReg_LPBA40_config()
        self.crosstrans5 = LWCA(config, dim_diy=256)
        self.crosstrans4 = LWCA(config, dim_diy=128)
        self.crosstrans3 = LWCA(config, dim_diy=64)
        self.crosstrans2 = LWCA(config, dim_diy=32)
        self.crosstrans1 = LWCA(config, dim_diy=16)

        self.cconv_5_1 = nn.Sequential(
            ConvInsBlock(12 * 4 * c, 16 * c, 3, 1),
            ConvInsBlock(4 * 4 * c, 16 * c, 3, 1)
        )
        self.cconv_5_1_1 = nn.Sequential(
            ConvInsBlock(12 * 4 * c, 16 * c, 3, 1),
            ConvInsBlock(4 * 4 * c, 16 * c, 3, 1)
        )

        self.cconv_4_1 = nn.Sequential(
            ConvInsBlock(6 * 4 * c, 8 * c, 3, 1),
            ConvInsBlock(2 * 4 * c, 8 * c, 3, 1)
        )
        self.cconv_4_1_1 = nn.Sequential(
            ConvInsBlock(6 * 4 * c, 8 * c, 3, 1),
            ConvInsBlock(2 * 4 * c, 8 * c, 3, 1)
        )
        self.cconv_3_1 = nn.Sequential(
            ConvInsBlock(3 * 4 * c, 4 * c, 3, 1),
            ConvInsBlock(1 * 4 * c, 4 * c, 3, 1)
        )
        self.cconv_3_1_1 = nn.Sequential(
            ConvInsBlock(3 * 4 * c, 4 * c, 3, 1),
            ConvInsBlock(1 * 4 * c, 4 * c, 3, 1)
        )
        self.cconv_2_1 = nn.Sequential(
            ConvInsBlock(3 * 2 * c, 2 * c, 3, 1),
            ConvInsBlock(1 * 2 * c, 2 * c, 3, 1)
        )
        self.cconv_2_1_1 = nn.Sequential(
            ConvInsBlock(3 * 2 * c, 2 * c, 3, 1),
            ConvInsBlock(1 * 2 * c, 2 * c, 3, 1)
        )
        self.cconv_1_1 = nn.Sequential(
            ConvInsBlock(1 * 3 * c, 1 * c, 3, 1),
            ConvInsBlock(1 * 1 * c, 1 * c, 3, 1)
        )
        self.cconv_1_Cross = nn.Sequential(
            ConvInsBlock(1 * 2 * c, 1 * c, 3, 1),
            ConvInsBlock(1 * 1 * c, 1 * c, 3, 1)
        )
        self.cconv_5_2 = nn.Sequential(
            ConvInsBlock(8 * 4 * c, 16 * c, 3, 1),
            ConvInsBlock(4 * 4 * c, 16 * c, 3, 1)
        )
        self.cconv_5_2_1 = nn.Sequential(
            ConvInsBlock(8 * 4 * c, 16 * c, 3, 1),
            ConvInsBlock(4 * 4 * c, 16 * c, 3, 1)
        )
        self.cconv_4_2 = nn.Sequential(
            ConvInsBlock(4 * 4 * c, 8 * c, 3, 1),
            ConvInsBlock(2 * 4 * c, 8 * c, 3, 1)
        )
        self.cconv_4_2_1 = nn.Sequential(
            ConvInsBlock(4 * 4 * c, 8 * c, 3, 1),
            ConvInsBlock(2 * 4 * c, 8 * c, 3, 1)
        )
        self.cconv_3_2 = nn.Sequential(
            ConvInsBlock(2 * 4 * c, 4 * c, 3, 1),
            ConvInsBlock(1 * 4 * c, 4 * c, 3, 1)
        )
        self.cconv_3_2_1 = nn.Sequential(
            ConvInsBlock(2 * 4 * c, 4 * c, 3, 1),
            ConvInsBlock(1 * 4 * c, 4 * c, 3, 1)
        )
        self.cconv_2_2 = nn.Sequential(
            ConvInsBlock(2 * 2 * c, 2 * c, 3, 1),
            ConvInsBlock(1 * 2 * c, 2 * c, 3, 1)
        )
        self.cconv_2_2_1 = nn.Sequential(
            ConvInsBlock(2 * 2 * c, 2 * c, 3, 1),
            ConvInsBlock(1 * 2 * c, 2 * c, 3, 1)
        )
        self.cconv_1_2 = nn.Sequential(
            ConvInsBlock(1 * 2 * c, 1 * c, 3, 1),
            ConvInsBlock(1 * 1 * c, 1 * c, 3, 1)
        )
        self.defconv5 = nn.Conv3d(16 * c, 3, 3, 1, 1)
        self.defconv5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv5.weight.shape))
        self.defconv5.bias = nn.Parameter(torch.zeros(self.defconv5.bias.shape))

        self.defconv4 = nn.Conv3d(8 * c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))

        self.defconv3 = nn.Conv3d(4 * c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))

        self.defconv2 = nn.Conv3d(2 * c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))

        self.defconv1 = nn.Conv3d(1 * c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

        self.getSA1 = CSA_ADDBlock(256)
        #self.feafuse1 = MS_CAM_3D(256)
        self.feafuse1 = FeaFuse(256, 256)
        self.getSA2 = CSA_ADDBlock(128)
        #self.feafuse2 = MS_CAM_3D(128)
        self.feafuse2 = FeaFuse(128, 128)
        self.getSA3 = CSA_ADDBlock(64)
        #self.feafuse3 = MS_CAM_3D(64)
        self.feafuse3 = FeaFuse(64, 64)
        self.getSA4 = CSA_ADDBlock(32)
        #self.feafuse4 = MS_CAM_3D(32)
        self.feafuse4 = FeaFuse(32, 32)
        self.getSA5 = CSA_ADDBlock(16)

        self.CrossSA_5_1 = CSABlock()
        self.CrossSA_4_1 = CSABlock()
        self.CrossSA_3_1 = CSABlock()
        self.CrossSA_2_1 = CSABlock()
        self.CrossSA_1_1 = CSABlock()

        # self.cconv_5_2 = nn.Sequential(
        #     ConvInsBlock(4 * 8 * c, 2 * 8 * c, 3, 1),
        #     ConvInsBlock(2 * 8 * c, 16 * c, 3, 1)
        # )
        #
        # self.cconv_4_2 = nn.Sequential(
        #     ConvInsBlock(4 * 4 * c, 2 * 4 * c, 3, 1),
        #     ConvInsBlock(2 * 4 * c, 8 * c, 3, 1)
        # )
        #
        # self.cconv_3_2 = nn.Sequential(
        #     ConvInsBlock(2 * 4 * c, 1 * 4 * c, 3, 1),
        #     ConvInsBlock(1 * 4 * c, 4 * c, 3, 1)
        # )
        #
        # self.cconv_2_2 = nn.Sequential(
        #     ConvInsBlock(2 * 2 * c, 1 * 2 * c, 3, 1),
        #     ConvInsBlock(1 * 2 * c, 2 * c, 3, 1)
        # )
        #
        # self.cconv_1_2 = nn.Sequential(
        #     ConvInsBlock(4 * 1 * c, 1 * 2 * c, 3, 1),
        #     ConvInsBlock(2 * 1 * c, 1 * c, 3, 1)
        # )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

    def forward(self, moving, fixed):
        # encode stage
        M1, M2, M3, M4, M5 = self.encoder_moving(moving)

        F1, F2, F3, F4, F5 = self.encoder_fixed(fixed)


        ### Iter stage 0
        # first block
        C_M5 = self.crosstrans5(F5, M5)
        fix_5 , move_5 , ADD = self.getSA1(F5, M5)
        CopyCross5 = self.cconv_5_2_1(torch.cat((C_M5, ADD), dim=1))
        Cross5 = torch.cat((fix_5, move_5, CopyCross5), dim=1)
        Cross5todef = self.cconv_5_1_1(Cross5) #256 10 12----256 10 12
        flow5_1 = self.defconv5(Cross5todef) #3 10 12
        flow5_1_diform = self.diff[4](flow5_1)  # get diformation flow

        # Second block
        warped = self.warp[4](M5, flow5_1_diform)  # (1,128,20burjiuqudianli,24,20)
        fix_5 , move_5 , ADD = self.getSA1(F5, warped)
        CopyCross5 = self.cconv_5_2(torch.cat((Cross5todef, ADD), dim=1))
        Cross5 = torch.cat((fix_5, move_5, CopyCross5), dim=1)
        Cross5todef = self.cconv_5_1(Cross5)
        flow5_2 = self.defconv5(Cross5todef)
        flow5_2_diform = self.diff[4](flow5_2)
        flow5_2 = flow5_1_diform + flow5_2_diform
        # Third block

        warped = self.warp[4](M5, flow5_2)  # (1,128,20burjiuqudianli,24,20)
        fix_5, move_5, ADD = self.getSA1(F5, warped)
        CopyCross5 = self.cconv_5_2(torch.cat((Cross5todef, ADD), dim=1))
        Cross5 = torch.cat((fix_5, move_5, CopyCross5), dim=1)
        Cross5todef = self.cconv_5_1(Cross5)
        flow5_3 = self.defconv5(Cross5todef)
        flow5_3_diform = self.diff[4](flow5_3)
        flow5 = flow5_2 + flow5_3_diform


        ### Iter stage 1
        # first block
        flow5 = self.ResizeTransformer(flow5)
        warped = self.warp[3](M4, flow5)
        C_M4 = self.crosstrans4(F4, warped)
        fix_4 , move_4 , ADD = self.getSA2(warped, F4)
        CopyCross4 = self.cconv_4_2_1(torch.cat((C_M4, ADD), dim=1))
        Cross4 = torch.cat((fix_4, move_4, CopyCross4), dim=1)
        Cross4todef = self.cconv_4_1_1(Cross4)
        flow4_1 = self.defconv4(Cross4todef)
        flow4_1_diform = self.diff[3](flow4_1)  # get diformation flow
        flow4 = flow5 + flow4_1_diform

        # Second block
        warped = self.warp[3](M4, flow4)  # (1,128,20burjiuqudianli,24,20)
        fix_4, move_4, ADD = self.getSA2(warped, F4)
        CopyCross4 = self.cconv_4_2(torch.cat((Cross4todef, ADD), dim=1))
        Cross4 = torch.cat((fix_4, move_4, CopyCross4), dim=1)
        Cross4todef = self.cconv_4_1(Cross4)
        flow4_2 = self.defconv4(Cross4todef)
        flow4_2_diform = self.diff[3](flow4_2)
        flow4 = flow4_1_diform + flow4_2_diform

        # Third block
        warped = self.warp[3](M4, flow4)  # (1,128,20burjiuqudianli,24,20)
        fix_4, move_4, ADD = self.getSA2(warped, F4)
        CopyCross4 = self.cconv_4_2(torch.cat((Cross4todef, ADD), dim=1))
        Cross4 = torch.cat((fix_4, move_4, CopyCross4), dim=1)
        Cross4todef = self.cconv_4_1(Cross4)
        flow4_3 = self.defconv4(Cross4todef)
        flow4_3_diform = self.diff[3](flow4_3)
        flow4 = flow4 + flow4_3_diform


        ### Iter stage 2
        # first block
        flow4 = self.ResizeTransformer(flow4)
        warped = self.warp[2](M3, flow4)
        C_M3 = self.crosstrans3(F3, warped)
        fix_3, move_3, ADD = self.getSA3(F3, warped)
        CopyCross3 = self.cconv_3_2_1(torch.cat((C_M3, ADD), dim=1))
        Cross3 = torch.cat((fix_3, move_3, CopyCross3), dim=1)
        Cross3todef = self.cconv_3_1_1(Cross3)
        flow3_1 = self.defconv3(Cross3todef)
        flow3_1_diform = self.diff[2](flow3_1)  # get diformation flow
        flow3 = flow4 + flow3_1_diform

        # Second block
        warped = self.warp[2](M3, flow3)
        #C_M3 = self.crosstrans3(F3, warped)
        fix_3, move_3, ADD = self.getSA3(F3, warped)
        CopyCross3 = self.cconv_3_2(torch.cat((Cross3todef, ADD), dim=1))
        Cross3 = torch.cat((fix_3, move_3, CopyCross3), dim=1)
        Cross3todef = self.cconv_3_1(Cross3)
        flow3_2 = self.defconv3(Cross3todef)
        flow3_2_diform = self.diff[2](flow3_2)
        flow3 = flow3 + flow3_2_diform

        # Third block
        warped = self.warp[2](M3, flow3)
        fix_3, move_3, ADD = self.getSA3(F3, warped)
        CopyCross3 = self.cconv_3_2(torch.cat((Cross3todef, ADD), dim=1))
        Cross3 = torch.cat((fix_3, move_3, CopyCross3), dim=1)
        Cross3todef = self.cconv_3_1(Cross3)
        flow3_3 = self.defconv3(Cross3todef)
        flow3_3_diform = self.diff[2](flow3_3)
        flow3 = flow3 + flow3_3_diform

        ### Iter stage 3
        # first block
        flow3 = self.ResizeTransformer(flow3)
        warped = self.warp[1](M2, flow3)
        C_M2 = self.crosstrans2(F2, warped)
        fix_2, move_2, ADD = self.getSA4(F2, warped)
        CopyCross2 = self.cconv_2_2_1(torch.cat((C_M2, ADD), dim=1))
        Cross2 = torch.cat((fix_2, move_2, CopyCross2), dim=1)
        Cross2todef = self.cconv_2_1_1(Cross2)
        flow2_1 = self.defconv2(Cross2todef)
        flow2_1_diform = self.diff[1](flow2_1)  # get diformation flow
        flow2 = flow3 + flow2_1_diform

        # Second block
        warped = self.warp[1](M2, flow2)  # (1,128,20burjiuqudianli,24,20)
        fix_2, move_2, ADD = self.getSA4(F2, warped)
        CopyCross2 = self.cconv_2_2(torch.cat((Cross2todef, ADD), dim=1))
        Cross2 = torch.cat((fix_2, move_2, CopyCross2), dim=1)
        Cross2todef = self.cconv_2_1(Cross2)
        flow2_2 = self.defconv2(Cross2todef)
        flow2_2_diform = self.diff[1](flow2_2)
        flow2 = flow2 + flow2_2_diform

        #     # Third block
        warped = self.warp[1](M2, flow2)  # (1,128,20burjiuqudianli,24,20)
        fix_2, move_2, ADD = self.getSA4(F2, warped)
        CopyCross2 = self.cconv_2_2(torch.cat((Cross2todef, ADD), dim=1))
        Cross2 = torch.cat((fix_2, move_2, CopyCross2), dim=1)
        Cross2todef = self.cconv_2_1(Cross2)
        flow2_3 = self.defconv2(Cross2todef)
        flow2_3_diform = self.diff[1](flow2_3)
        flow2 = flow2 + flow2_3_diform

        ### Iter stage 4
        flow2 = self.ResizeTransformer(flow2)
        warped = self.warp[0](M1, flow2)
        Cross = self.cconv_1_Cross(torch.cat((warped, F1), dim=1))
        Cross1 = torch.cat((F1, warped, Cross), dim=1)

        # min_val = Cross1.min()
        # max_val = Cross1.max()
        # u = torch.unique(Cross1)
        # print(f"张量的最小值: {min_val.item()}")
        # print(f"张量的最大值: {max_val.item()}")
        # print(f"张量的值: ", u)

        Cross1todef = self.cconv_1_1(Cross1)
        flow1_1 = self.defconv1(Cross1todef)

        # min_val = flow1_1.min()
        # max_val = flow1_1.max()
        # u = torch.unique(flow1_1)
        # print(f"张量的最小值: {min_val.item()}")
        # print(f"张量的最大值: {max_val.item()}")
        # print(f"张量的值: ", u)

        flow1_1_diform = self.diff[0](flow1_1)
        flow1 = flow2 + flow1_1_diform

        # min_val = flow1.min()
        # max_val = flow1.max()
        # u = torch.unique(flow1)
        # print(f"张量的最小值: {min_val.item()}")
        # print(f"张量的最大值: {max_val.item()}")
        # print(f"张量的值: ", u)

        # warped = self.warp[0](M1, flow1)
        # fix_1, move_1, ADD = self.getSA4(F1, warped)
        # CopyCross1 = self.cconv_1_2(torch.cat((Cross1todef, ADD), dim=1))
        # Cross1 = torch.cat((fix_1, move_1, CopyCross1), dim=1)
        # Cross1todef = self.cconv_1_1(Cross1)
        # # Cross1 = torch.cat((warped, F1), dim=1)
        # # Corss1todef = self.cconv_1_1(Cross1)
        # flow1_1 = self.defconv1(Cross1todef)
        # flow1_1_diform = self.diff[0](flow1_1)
        # flow1 = flow1 + flow1_1_diform

        # min_val = flow1.min()
        # max_val = flow1.max()
        # u = torch.unique(flow1)
        # print(f"张量的最小值: {min_val.item()}")
        # print(f"张量的最大值: {max_val.item()}")
        # print(f"张量的值: ", u)
        wrapped = self.warp[0](moving, flow1)

        return wrapped, flow1
