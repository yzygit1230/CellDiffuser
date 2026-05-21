import torch.nn as nn
import math
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch.nn import functional as F
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


    
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x
        
    
class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)
  
class MambaNet(nn.Module):

    def __init__(self, num_classes=1000, pos_pos=0,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(MambaNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.pos_pos = pos_pos
        self.inplanes = 64
        self.dilation = 1
        self.da_act = 'relu'
        self.gated = True
        self.mlp_ratio = 2

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1p = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1p = norm_layer(self.inplanes)
        self.cap = ChannelAttention(self.inplanes)
        self.sap = SpatialAttention()
        self.maxpoolp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        dpr1 = torch.linspace(0, 0.2, 4)[0].item()
        depth1 = 3
        dpr2 = torch.linspace(0, 0.2, 4)[1].item()
        depth2 = 3
        dpr3 = torch.linspace(0, 0.2, 4)[2].item()
        depth3 = 7
        dpr4 = torch.linspace(0, 0.2, 4)[3].item()
        depth4 = 7
        self.mambalayer_i1 = MambaVisionLayer(dim = 64, depth = depth1, num_heads = 2, window_size = 8, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = True, drop = 0, attn_drop = 0., drop_path=dpr1, downsample= True,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth1//2+1, depth1)) if depth1%2!=0 else list(range(depth1//2, depth1)),)
        self.mambalayer_i2 = MambaVisionLayer(dim = 128, depth = depth2, num_heads = 4, window_size = 8, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = True, drop = 0, attn_drop = 0., drop_path=dpr2, downsample= True,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth2//2+1, depth2)) if depth2%2!=0 else list(range(depth2//2, depth2)),)
        self.mambalayer_i3 = MambaVisionLayer(dim = 256, depth = depth3, num_heads = 8, window_size = 14, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = False, drop = 0, attn_drop = 0., drop_path=dpr3, downsample= True,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth3//2+1, depth3)) if depth3%2!=0 else list(range(depth3//2, depth3)),)
        self.mambalayer_i4 = MambaVisionLayer(dim = 512, depth = depth4, num_heads = 16, window_size = 7, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = False, drop = 0, attn_drop = 0., drop_path=dpr4, downsample= False,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth4//2+1, depth4)) if depth4%2!=0 else list(range(depth4//2, depth4)),)
        
        self.mambalayer_p1 = MambaVisionLayer(dim = 64, depth = depth1, num_heads = 2, window_size = 8, mlp_ratio = 4, qkv_bias = True,
                                qk_scale = None, conv = True, drop = 0, attn_drop = 0., drop_path=dpr1, downsample= True,
                                layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth1//2+1, depth1)) if depth1%2!=0 else list(range(depth1//2, depth1)),)
        self.mambalayer_p2 = MambaVisionLayer(dim = 128, depth = depth2, num_heads = 4, window_size = 8, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = True, drop = 0, attn_drop = 0., drop_path=dpr2, downsample= True,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth2//2+1, depth2)) if depth2%2!=0 else list(range(depth2//2, depth2)),)
        self.mambalayer_p3 = MambaVisionLayer(dim = 256, depth = depth3, num_heads = 8, window_size = 14, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = False, drop = 0, attn_drop = 0., drop_path=dpr3, downsample= True,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth3//2+1, depth3)) if depth3%2!=0 else list(range(depth3//2, depth3)),)
        self.mambalayer_p4 = MambaVisionLayer(dim = 512, depth = depth4, num_heads = 16, window_size = 7, mlp_ratio = 4, qkv_bias = True,
                                     qk_scale = None, conv = False, drop = 0, attn_drop = 0., drop_path=dpr4, downsample= False,
                                     layer_scale= None, layer_scale_conv= None, transformer_blocks=list(range(depth4//2+1, depth4)) if depth4%2!=0 else list(range(depth4//2, depth4)),)

        self.ca_i = ChannelAttention(512)
        self.sa_i = SpatialAttention()
        self.ca_p = ChannelAttention(512)
        self.sa_p = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.convip = nn.Sequential(nn.Conv2d(1024, 768, (1, 1)),
                    nn.Conv2d(768, 512, (1, 1)))

        
        self.attention = nn.Sequential(
            nn.Linear(512, 256 ),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x, xp):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.maxpool(x)

        x = self.mambalayer_i1(x)
        x = self.mambalayer_i2(x)
        x = self.mambalayer_i3(x)
        x = self.mambalayer_i4(x)  
        x = self.ca_i(x) * x
        x = self.sa_i(x) * x

        xp = self.conv1p(xp)
        xp = self.cap(xp) * xp
        xp = self.sap(xp) * xp
        xp = self.maxpoolp(xp)
        xp = self.mambalayer_p1(xp)
        xp = self.mambalayer_p2(xp)
        xp = self.mambalayer_p3(xp)
        xp = self.mambalayer_p4(xp)  
        xp = self.ca_p(x) * xp
        xp = self.sa_p(x) * xp

        x = self.convip(torch.cat([x, xp], 1))
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        H=x
        A = self.attention(H) 
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)  
        M = torch.mm(A, H)  
        feature = M               
        output = self.classifier(M)
        return output, feature

