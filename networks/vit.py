import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from .network_utils import (
    Classifier, 
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6,
    convert_to_rpm_matrix_mnr
)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    # [B, 1, 80, 80] [B, 25, 256]
    def __init__(self, img_size=80, patch_size=8, in_chans=1, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"Image size ({img_size}) must be divisible by patch size ({patch_size})."
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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

# 1. 把这个函数加到你的文件最上面
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# 2. 修改 Block 类，加入 drop_path
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.): # <--- 新增参数
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # 注意这里：如果 drop_path > 0 就初始化 DropPath，否则是 Identity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # 关键修改：把 drop_path 包在残差连接外面
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=80, patch_size=16, in_chans=1, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        init.trunc_normal_(self.pos_embed, std=.02)
        init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) 

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0] # Only return CLS token

def ConvNormAct(inplanes, ouplanes, kernel_size=3, padding=0, stride=1, activate=True):
    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm2d(ouplanes)]
    if activate:
        block += [nn.ReLU(inplace=True)]
    return nn.Sequential(*block)

class ResBlock(nn.Module):
    def __init__(self, inplanes, ouplanes, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvNormAct(inplanes, ouplanes, 3, 1, stride=stride)
        self.conv2 = ConvNormAct(ouplanes, ouplanes, 3, 1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()
        
        self.downsample = nn.Identity()
        if stride != 1 or inplanes != ouplanes:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride) if stride > 1 else nn.Identity(),
                ConvNormAct(inplanes, ouplanes, 1, 0, activate=False)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        return out

class PredRNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 num_classes=8,
                 num_contexts=8,
                 # ViT Params
                 img_size=80,
                 patch_size=8,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 # ResNet Params
                 num_filters=32, 
                 block_drop=0.1,
                 # Common
                 classifier_drop=0.1,
                 classifier_hidreduce=1.0,
                 **kwargs):
        super().__init__()

        self.num_contexts = num_contexts
        self.ou_channels = num_classes
        self.featr_dims = 1024

        self.backbone_type = 'vit'
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=block_drop
        )
        self.embed_dim = embed_dim
        
        # self.vit_img_size = 224           # ViT 预训练分辨率
        # # 1. 创建预训练的 ViT，去掉自己的分类头，把它当特征提取
        # self.vit = timm.create_model("vit_base_patch16_224", 
        #                              pretrained=True, in_chans=1, img_size=80)
        # self.vit.reset_classifier(0)      # 输出 [B, embed_dim]
        # vit_dim = self.vit.num_features   # 一般 768
        # self.featr_dims = vit_dim * 2

        # self.backbone_type = 'resnet'
        # self.in_planes = in_channels
        # channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        # strides = [2, 2, 2, 2]
        # for l in range(len(strides)):
        #     setattr(
        #         self, "res"+str(l), 
        #         self._make_resnet_layer(
        #             channels[l], stride=strides[l], dropout=block_drop
        #         )
        #     )
        # self.embed_dim = channels[-1]

        
        # ---------------------------------------------------------
        
        self.classifier = Classifier(
            self.featr_dims, 1,  # Output 1 score per candidate
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )
        
        # Initialize weights for ResNet parts if active
        if self.backbone_type == 'resnet':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_resnet_layer(self, planes, stride, dropout):
        block = ResBlock
        stage = block(self.in_planes, planes, stride=stride, dropout=dropout)
        self.in_planes = planes
        return stage
    
    # def _prep_images(self, x):
        """
        输入 x:
          - 灰度: [B, 16, 1, H, W]
          - 彩色: [B, 16, 3, H, W]
        输出:
          - x_vit: [B*16, 3, 224, 224]
          - b: 原始 batch_size B
        """
        if x.dim() == 5:
            # [B, 16, C, H, W]
            b, n, c, h, w = x.size()
            assert n == 16, f"expect 16 panels, got {n}"
            assert c in (1, 3), f"expect 1 or 3 channels, got {c}"

            x = x.reshape(b * n, c, h, w)  # [B*16, C, H, W]

            if c == 1:
                # 灰度 → 复制成 3 通道
                x = x.repeat(1, 3, 1, 1)    # [B*16, 3, H, W]

        elif x.dim() == 4:
            # 兼容以前写法: [B, 16, H, W] 且是灰度
            b, n, h, w = x.size()
            assert n == 16
            x = x.reshape(b * n, 1, h, w)
            x = x.repeat(1, 3, 1, 1)        # [B*16, 3, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # resize 到 ViT 输入大小
        x = F.interpolate(
            x,
            size=(self.vit_img_size, self.vit_img_size),
            mode="bilinear",
            align_corners=False
        )
        return x, b
    
    def forward(self, x, train=False):
        # # x, b = self._prep_images(x)        # x: [B*16, 3, 224, 224]
        # # x: [B, 16, C, H, W] or [B, 16, H, W]
        # if x.dim() == 4:
        #     b, n, h, w = x.size()
        #     x = x.unsqueeze(2) # [B, 16, 1, H, W]
        # else:
        #     b, n, c, h, w = x.size()
        
        # x = x.view(b * n, -1, h, w)

        # feats = self.vit(x)                # [B*16, D]
        # feats = feats.reshape(b, 16, -1)   # [B, 16, D]

        # contexts = feats[:, :8, :]         # [B, 8, D]
        # choices  = feats[:, 8:, :]         # [B, 8, D]

        # context_feat = contexts.mean(dim=1, keepdim=True)      # [B, 1, D]
        # context_feat = context_feat.repeat(1, self.ou_channels, 1)  # [B, 8, D]

        # combined = torch.cat([context_feat, choices], dim=-1)  # [B, 8, 2D]
        # combined = combined.reshape(b * self.ou_channels, -1)  # [B*8, 2D]

        # out = self.classifier(combined)                        # [B*8, 1]
        # return out.view(b, self.ou_channels)                   # [B, 8]

        # x: [B, 16, C, H, W] or [B, 16, H, W]
        if x.dim() == 4:
            b, n, h, w = x.size()
            x = x.unsqueeze(2) # [B, 16, 1, H, W]
        else:
            b, n, c, h, w = x.size()
        
        x = x.view(b * n, -1, h, w)
        
        # vit
        features = self.encoder(x) # [B*16, embed_dim]
        features = features.view(b * n, -1, 1, 1) 
        
        # # resnet
        # for l in range(4):
        #     x = getattr(self, "res"+str(l))(x)
        # # Output: (B*16, 128, 5, 5) -> Global Pooling -> (B*16, 128, 1, 1)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # features = x # Keep 4D shape for compatibility with size() calls below
        
        
        # RPM Logic (Scoring Mode)
        _, c_feat, h_feat, w_feat = features.size()
        
        # 这里 h_feat, w_feat 对于 ViT 和 Pooled ResNet 都是 1
        # 将特征重组为 (Context + Candidate) 的组合
        if self.num_contexts == 8:
            x = convert_to_rpm_matrix_v9(features, b, h_feat, w_feat)
        elif self.num_contexts == 3:
            x = convert_to_rpm_matrix_mnr(features, b, h_feat, w_feat)
        else:
            x = convert_to_rpm_matrix_v6(features, b, h_feat, w_feat)

        # x shape: [B, 8, (Context+1), C, 1, 1]
        
        # Flatten and Score
        # [B, 8, (Context+1)*C]
        x = x.reshape(b, self.ou_channels, -1)
        
        # Dimension Reduction / Feature Fusion
        # [B, 8, featr_dims]
        x = F.adaptive_avg_pool1d(x, self.featr_dims) 
        
        # Flatten for classifier: [B*8, featr_dims]
        x = x.reshape(-1, self.featr_dims)
        
        # Calculate Score: [B*8, 1]
        out = self.classifier(x)
        
        # Reshape to [B, 8]
        return out.view(b, self.ou_channels)

def predrnet_raven_vit(**kwargs):
    return PredRNet(**kwargs, num_contexts=8)

def predrnet_mnr(**kwargs):
    return PredRNet(**kwargs, num_contexts=3)

def predrnet_analogy(**kwargs):
    return PredRNet(**kwargs, num_contexts=5, num_classes=4)

# Legacy aliases
def mrnet(**kwargs): return PredRNet(**kwargs)
def hcvarr(**kwargs): return PredRNet(**kwargs)
def scar(**kwargs): return PredRNet(**kwargs)