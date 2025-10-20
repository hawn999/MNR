import itertools
from functools import partial
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from .network_utils import (
    Classifier,
    ResBlock,
    ConvNormAct,
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6,
    convert_to_rpm_matrix_mnr,
    LinearNormAct,
    CausalReasoningModule,
    CausalInterventionEngine,
    CausalGraphDiscovery
)
from .position_embedding import PositionalEncoding, LearnedAdditivePositionalEmbed
from .HCVARR import HCVARR
from .SCAR import RelationNetworkSCAR
from .Pred import Pred
from .MM import MM
from .MRnet import MRNet
from torch.nn import init
## slot + casual


class SymbolicEncoder(nn.Module):
    """
    把 one-hot → index → Embedding，再 reshape
    """
    def __init__(self, h: int = 20, w: int = 20):
        super().__init__()
        self.h, self.w = h, w
        self.embed = nn.Embedding(8, h * w)  # 8 个离散值 → 9-D 向量

    def forward(self, x):                   # (B*16, 4, 8) one-hot
        # 先把 one-hot 转成整数标签
        idx = x.argmax(dim=-1)              # (B*16, 4)
        B, n, C = idx.shape
        x = self.embed(idx)                 # (B*16, 4, 9)
        x = x.view(B*n, C, self.h, self.w)  # (B*16, 4, 20, 20)
        return x

class SymbolEncoding(nn.Module):
    def __init__(self, num_contexts=4, d_model=32, f_len=24):
        super(SymbolEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, d_model, num_contexts, f_len))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self):
        return self.position_embeddings

class SymbolEncoding(nn.Module):
    def __init__(self, num_contexts=4, d_model=32, f_len=24):
        super(SymbolEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, d_model, num_contexts, f_len))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self):
        return self.position_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:, :x.size(1), :]


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim=96, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None, return_attn=False):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # B, K, N
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return (slots, attn) if return_attn else slots

class SlotRouterGroup(nn.Module):
    """
    把通道软分成 num_slots 个 slot → 再按组装进 num_giprbs 个分支；
    每个分支处理 n=num_slots//num_giprbs 个 slots。
    """
    def __init__(self, in_channels, num_slots, num_giprbs, slot_dim=32, iters=3):
        super().__init__()
        assert num_slots > 0 and num_giprbs > 0
        self.in_channels  = in_channels
        self.num_slots    = num_slots
        self.num_giprbs   = num_giprbs
        self.slot_dim     = slot_dim
        self.n_per_branch = (num_slots + num_giprbs - 1) // num_giprbs  # ceil 分桶，后续会pad

        # 1) SlotAttention（对每个通道的时空轨迹做聚合与竞争）
        self.slot_attn = SlotAttention(num_slots=num_slots, iters=iters)

        # 2) 每个 slot 自己的一套 C→slot_dim 的 1x1 卷积（参数共享 or 不共享二选一）
        #    共享更省参：一个共享投影 + 每slot一个标量门控；不共享更灵活：每slot一个Conv。
        self.shared_proj = nn.Conv2d(in_channels, slot_dim, kernel_size=1, bias=False)
        self.slot_gates  = nn.Parameter(torch.ones(num_slots))  # (K,) 每slot一个标量门

        # 3) 每个分支的聚合器：把 n*slot_dim → 32（或你想要的GIPRB输入维度）
        self.branch_mix = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_per_branch * slot_dim, slot_dim, kernel_size=1, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(slot_dim),
            ) for _ in range(num_giprbs)
        ])

    def forward(self, x):
        """
        x: (B, C, T, L)
        return: list of length num_giprbs, 每个张量形状 (B, slot_dim, T, L)
        """
        B, C, T, L = x.shape
        # === 把“每个通道的一条 T*L 轨迹”当作一个 token ===
        tokens = x.view(B, C, T*L)  # (B, N=C, D=T*L)

        # === SlotAttention 得到 K=num_slots 个 slots 及注意力指派 ===
        _, attn = self.slot_attn(tokens, return_attn=True)  # attn: (B, K, N=C)
        # 归一化防数值飘：对 N 做 L1 归一（可选）
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        # === 为每个 slot 生成一个门控后的特征图 (B, slot_dim, T, L) ===
        # 用注意力在通道维做“软选择” → 当作通道门控，然后共享1×1把 C→slot_dim
        slots_feats = []
        x_shared = self.shared_proj(x)  # (B, slot_dim, T, L) 先投影一份
        for k in range(self.num_slots):
            gate = attn[:, k, :].view(B, C, 1, 1)       # (B, C, 1, 1)
            xk   = self.shared_proj(x * gate)           # (B, slot_dim, T, L)
            alpha = torch.relu(self.slot_gates[k])
            slots_feats.append(alpha * xk)

        # === 对齐分桶：按顺序把 K 个 slot 分到 num_giprbs 组，每组 n_per_branch 个 ===
        # 如果 K 不是整除，会在最后一组右侧 pad 0
        needed = self.n_per_branch * self.num_giprbs
        if needed > self.num_slots:
            pad_k = needed - self.num_slots
            slots_feats += [torch.zeros_like(slots_feats[0]) for _ in range(pad_k)]

        # 组织为 [B, G, n, slot_dim, T, L]
        feats = torch.stack(slots_feats, dim=1)  # (B, K_pad, slot_dim, T, L)
        feats = feats.view(B, self.num_giprbs, self.n_per_branch, self.slot_dim, T, L)

        # === 每个分支把 n 个slot 在通道维 concat 再用1×1聚合成 slot_dim ===
        outs = []
        for g in range(self.num_giprbs):
            fg = feats[:, g]                                   # (B, n, slot_dim, T, L)
            fg = fg.reshape(B, self.n_per_branch*self.slot_dim, T, L)
            fg = self.branch_mix[g](fg)                        # (B, slot_dim, T, L)
            outs.append(fg)
        return torch.cat(outs, dim=1)  # list of tensors, each (B, slot_dim*32, T, L)


class FusionAttention(nn.Module):
    def __init__(
            self,
            in_planes,
            dropout=0.1,
            num_heads=8
    ):
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.k = nn.Linear(in_planes, in_planes)
        self.v = nn.Linear(in_planes, in_planes)
        self.num_heads = num_heads
        self.head_dim = in_planes // num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        shortcut = q
        b, t, l, c = q.shape
        b_, t_, l_, c_ = b, t, l, c

        q = self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q = F.normalize(q, dim=-1)

        b, t, l, c = k.shape
        k = self.k(k).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = F.normalize(k, dim=-1)

        v = self.v(v).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = F.normalize(v, dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)

        x = self.m(x.permute(0, 1, 3, 2, 4).reshape(b_, t_, l_, c_)) + shortcut
        return x


class PredictionIntraAttention(nn.Module):
    def __init__(self, d_model, token_len, nhead=8, dropout=0.1, num_contexts=9):
        super(PredictionIntraAttention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.m = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm((32, num_contexts, token_len))
        self.norm2 = nn.LayerNorm((32, num_contexts, token_len))

        self.pre_prompt = SymbolEncoding(num_contexts, d_model, token_len)
        self.p = nn.Sequential(ConvNormAct(32, 32, 3, 1), nn.Linear(token_len, 6))

        self.token_len = token_len

    def forward(self, x, atten_flag):
        b, c, t, l = x.shape
        x = self.norm1(x)
        pre_prompt = self.pre_prompt().expand(b, -1, -1, -1)
        q, k, v = x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)

        q = F.normalize(self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k = F.normalize(self.k(k).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        v = F.normalize(self.v(v).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = self.drop(atten @ v)

        x = self.m(x.permute(0, 1, 3, 2, 4).reshape(b, t, l, c)).permute(0, 3, 1, 2).contiguous()
        x = self.norm2(x)

        if atten_flag == 1:
            con = torch.cat([x[:, :, :, :18], pre_prompt[:, :, :, 18:]], dim=3)
            tar = x[:, :, :, 18:]
        elif atten_flag == 2:
            con = torch.cat([x[:, :, :, :12], pre_prompt[:, :, :, 12:18], x[:, :, :, 18:]], dim=3)
            tar = x[:, :, :, 12:18]
        elif atten_flag == 3:
            con = torch.cat([x[:, :, :, :6], pre_prompt[:, :, :, 6:12], x[:, :, :, 12:]], dim=3)
            tar = x[:, :, :, 6:12]
        else:
            con = torch.cat([pre_prompt[:, :, :, :6], x[:, :, :, 6:]], dim=3)
            tar = x[:, :, :, :6]

        p = self.p(con).contiguous()
        p = F.relu(tar) - p

        if atten_flag == 1:
            x = torch.cat([x[:, :, :, :18], p], dim=-1)
        elif atten_flag == 2:
            x = torch.cat([x[:, :, :, :12], p, x[:, :, :, 18:]], dim=-1)
        elif atten_flag == 3:
            x = torch.cat([x[:, :, :, :6], p, x[:, :, :, 12:]], dim=-1)
        else:
            x = torch.cat([p, x[:, :, :, 6:]], dim=-1)
        # print(x.shape)
        # return x
        return self.drop2(x)

class GIPRB(nn.Module):

    def __init__(
            self,
            in_planes,
            token_len,
            dropout=0.0,
            num_heads=8,
            num_contexts=3
    ):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes * 2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        # self.norm = nn.LayerNorm((4, 24, 32))
        # self.norm1 = nn.LayerNorm((4, 24, 32))
        # self.norm2 = nn.LayerNorm((4, 24, 32))
        # self.norm3 = nn.LayerNorm((4, 24, 32))
        # self.norm4 = nn.LayerNorm((4, 24, 32))
        # ablation
        self.norm = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm1 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm2 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm3 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm4 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)

        self.pre_att = PredictionIntraAttention(in_planes, nhead=num_heads, token_len=token_len,
                                                num_contexts=num_contexts + 1)  # ablation
        self.pre_att2 = PredictionInterAttention(in_planes, nhead=num_heads, token_len=token_len,
                                                 num_contexts=num_contexts + 1)  # ablation
        self.conv1 = nn.Sequential(ConvNormAct((num_contexts + 1), (num_contexts + 1) * 4, 3, 1, activate=True),
                                   ConvNormAct((num_contexts + 1) * 4, (num_contexts + 1), 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes * 4, 3, 1, activate=True),
                                   ConvNormAct(in_planes * 4, in_planes, 3, 1, activate=True))
        self.conv3 = nn.Sequential(ConvNormAct(in_planes, in_planes * 4, 3, 1, activate=True),
                                   ConvNormAct(in_planes * 4, in_planes, 3, 1, activate=True))
        self.fusion = FusionAttention(in_planes)

    def forward(self, x, atten_flag):
        # b, c, t, l = x.shape
        # x = x.permute(0,2,3,1).reshape(b,t*l,c)
        # sltx = self.sltA(x)
        # x = self.cro(x, sltx).reshape(b,t,l,c).permute(0,3,1,2).contiguous()
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0, 2, 3, 1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        x = x.permute(0, 3, 1, 2)

        p1 = self.pre_att(x, atten_flag)
        p2 = self.pre_att2(x, atten_flag)

        p = self.conv3(p1+p2).permute(0,2,3,1)
        x = self.norm(self.lp2(F.gelu(g) * p)).permute(0, 3, 1, 2).contiguous()
        # p = (p1 + p2).permute(0, 2, 3, 1)
        # x = self.lp2(F.gelu(g) * p).permute(0, 3, 1, 2).contiguous()

        x = self.drop(self.conv2(x)) + shortcut

        x = self.fusion(self.norm1(x.permute(0,2,3,1).contiguous()), self.norm2(p1.permute(0,2,3,1)), self.norm3(p2.permute(0,2,3,1)))
        x = self.norm4(x).permute(0,3,1,2).contiguous()
        # x = self.fusion(x.permute(0, 2, 3, 1).contiguous(), p1.permute(0, 2, 3, 1),
        #                 p2.permute(0, 2, 3, 1))
        # x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PredictionInterAttention(nn.Module):
    def __init__(self, d_model, token_len, nhead=8, dropout=0.1, num_contexts=9):
        super(PredictionInterAttention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.m = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.num_heads = nhead
        self.head_dim = d_model // nhead

        self.pre_prompt = SymbolEncoding(1, d_model, token_len)
        self.p = nn.Sequential(ConvNormAct(32, 32, (num_contexts, 1)), nn.Linear(token_len, token_len))
        self.norm1 = nn.LayerNorm((32, num_contexts, token_len))
        self.norm2 = nn.LayerNorm((32, num_contexts, token_len))

        self.token_len = token_len
        self.num_contexts = num_contexts

    def forward(self, x, atten_flag):
        x = self.norm1(x)
        b, c, t, l = x.shape
        pre_prompt = self.pre_prompt().expand(b, -1, -1, -1)
        q, k, v = x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)

        q = F.normalize(self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k = F.normalize(self.k(k).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        v = F.normalize(self.v(v).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = self.drop(atten @ v)

        x = self.m(x.permute(0, 1, 3, 2, 4).reshape(b, t, l, c)).permute(0, 3, 1, 2).contiguous()
        x = self.norm2(x)

        # num_contexts = 4 or 9
        con = torch.cat([x[:, :, :self.num_contexts - 1], pre_prompt], dim=2)
        p = self.p(con).contiguous()
        p = F.relu(x[:,:,self.num_contexts-1:]) - p
        # p = x[:, :, self.num_contexts - 1:] - p

        x = torch.cat([x[:, :, :self.num_contexts - 1], p], dim=2)
        # return x
        return self.drop2(x)

# ====================== PERIC-OC: Object-Centric front-end ======================

class ObjectCentricEncoder(nn.Module):
    """
    CNN backbone 特征 -> panel 内 patch token -> SlotAttention(24个槽)
    输出形状: [B*ou, C_slot, T, S]  其中 S=24 (token_len), C_slot=slot_dim(缺省=128)
    """
    def __init__(self, in_channels=128, slot_dim=128, num_slots=24, sa_iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # 把 CNN 的通道对齐到 slot_dim（等价于 1x1 conv）
        self.token_proj = nn.Linear(in_channels, slot_dim)

        # 直接复用本文件已有的 SlotAttention 实现
        self.slot_attn = SlotAttention(num_slots=num_slots, dim=slot_dim, iters=sa_iters, hidden_dim=slot_dim)

    @torch.no_grad()
    def _to_tokens(self, x4):
        """
        x4: [B*ou*T, C, H, W] 或者 [B*ou, C, T, H*W]
        这里我们接在 PredRNet 的 rpm-matrix 之后，输入形状是 [B*ou, C, T, L]
        """
        assert x4.dim() == 4
        return x4  # 由 PredRNet 负责把 C 对齐成我们需要的 in_channels

    def forward(self, x):  # x: [B*ou, C, T, L=25]  ->  [B*ou, slot_dim, T, S=24]
        BOU, C, T, L = x.shape
        # 交换成 [BOU*T, L, C] 做每个面板各自的 SlotAttention
        x_perm = x.permute(0, 2, 3, 1).contiguous().view(BOU*T, L, C)
        x_tok  = self.token_proj(x_perm)                                  # [BOU*T, L, slot_dim]
        slots  = self.slot_attn(x_tok)                                    # [BOU*T, S, slot_dim]
        slots  = slots.view(BOU, T, self.num_slots, self.slot_dim)        # [BOU, T, S, slot_dim]
        # 回到 [BOU, slot_dim, T, S]
        slots  = slots.permute(0, 3, 1, 2).contiguous()
        return slots
# ====================== Query-Anchored Slots Encoder (stable OC front-end) ======================
import torch
import torch.nn as nn

class QueryAnchoredSlots(nn.Module):
    """
    25个patch特征  --cross-attn-->  固定顺序的24槽
    输出: [BOU, slot_dim, T, 24]
    """
    def __init__(self, in_channels=128, slot_dim=128, num_slots=24, num_heads=8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.token_proj = nn.Linear(in_channels, slot_dim, bias=False)
        self.key_pe    = nn.Parameter(torch.zeros(1, 25, slot_dim))    # 25 patch 的可学位置编码
        self.queries   = nn.Parameter(torch.randn(1, num_slots, slot_dim))  # 固定24槽查询
        self.cross_attn = nn.MultiheadAttention(slot_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim*2),
            nn.GELU(),
            nn.Linear(slot_dim*2, slot_dim),
        )

    def forward(self, x):  # x: [BOU, C, T, L=25]
        BOU, C, T, L = x.shape
        assert L == 25, f"expect 25 patches, got {L}"

        xt = x.permute(0,2,3,1).contiguous().view(BOU*T, L, C)         # [BOU*T, 25, C]
        k  = self.token_proj(xt) + self.key_pe[:, :L, :]               # [BOU*T, 25, D]
        q  = self.queries.expand(BOU*T, -1, -1)                        # [BOU*T, 24, D]

        # Cross-attn: queries attend to all 25 tokens
        out, _ = self.cross_attn(q, k, k)                              # [BOU*T, 24, D]
        out = self.ffn(out)                                            # [BOU*T, 24, D]

        out = out.view(BOU, T, self.num_slots, self.slot_dim)          # [BOU, T, 24, D]
        out = out.permute(0, 3, 1, 2).contiguous()                     # [BOU, D, T, 24]
        return out


# =============== MPER on object slots: keep your GIPRB idea ===============

class MPER_OC(nn.Module):
    """
    以对象槽为输入的 MPER：
    - 仍然把通道 reduce_planes 按 32 一组切块，逐组送入你原来的 GIPRB（保留实现思路）
    - CCR/MCR 分别对应 inter/intra，沿用你的 PredictionInter/PredictionIntraAttention
    - 不再使用 HYLA/DRACO，这里只产出 elementary_rule_embeddings 供因果模块使用
    """
    def __init__(self, in_planes_per_group=32, reduce_planes=128, num_contexts=8, token_len=24):
        super().__init__()
        self.num_contexts = num_contexts
        self.token_len = token_len
        self.reduce_planes = reduce_planes

        # 把 slot_dim 对齐为 reduce_planes（等价 channel_reducer）
        self.channel_reducer = ConvNormAct(128, reduce_planes, 1, 0, activate=False)

        # 分组（每组 32 通道） -> 走你现有的 GIPRB
        assert reduce_planes % in_planes_per_group == 0
        self.num_groups = reduce_planes // in_planes_per_group

        # 25->24 的线性映射在对象槽里不需要了，槽数就是 24
        self.to_24 = nn.Identity()

        # group-wise map + GIPRB（复用你已有实现）
        for g in range(self.num_groups):
            setattr(self, f"map{g}", nn.Linear(token_len, token_len))
            setattr(self, f"GIPRB{g}", GIPRB(in_planes=in_planes_per_group, token_len=token_len, num_contexts=num_contexts))

        # 把拼接后的 group 输出再做一次 conv 融合，得到 elementary rule embeddings
        self.post_conv = ConvNormAct(in_planes_per_group*self.num_groups, reduce_planes, 3, 1)

    def forward(self, x_slots):  # [BOU, slot_dim(=128), T, S=24]
        x = self.channel_reducer(x_slots)                                  # [BOU, 128->reduce_planes, T, 24]
        x = self.to_24(x)                                                  # identity, 保持 24 槽

        # 按组切分通道，逐组过 GIPRB
        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)             # N组, 每组32通道
        outs = []
        for g, xi in enumerate(chunks):
            xi = getattr(self, f"map{g}")(xi)                               # 仍然沿用你在 PRB 里的 token 线性
            xi = getattr(self, f"GIPRB{g}")(xi, g + 1)                      # 保留你的 GIPRB 实现/思路
            outs.append(xi)
        x = torch.cat(outs, dim=1)                                          # [BOU, reduce_planes, T, 24]
        elem_rule_embeds = self.post_conv(x)                                # elementary rule embeddings
        return elem_rule_embeds

# ====================== Causal-PERIC: Causal Reasoning backend ======================

def acyclicity_loss(A: torch.Tensor) -> torch.Tensor:
    """
    NOTEARS 风格的无环约束: h(A) = trace(exp(A∘A)) - d
    """
    d = A.size(0)
    expm = torch.matrix_exp(A * A)
    return torch.trace(expm) - d

class CausalReasoningModule(nn.Module):
    """
    输入/输出形状与 MPER 的 elementary rule embeddings 对齐：
      x: [B_like, C_embed, T, L] -> out: 同形状
    关键修复点：
      1) 编码后得到 z_attr: [B, D_attr, TL]，必须先对 TL 做平均 -> z_mean: [B, D_attr]
      2) 因果传播在 z_mean 空间进行，得到 y_attr: [B, D_attr]
      3) 解码用 Conv1d 接收 [B, D_attr, 1]（3D），再广播回 [B, C_embed, T, L]
    """
    def __init__(self, embed_dim=128, token_len=24, num_contexts=8, attr_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_len = token_len
        self.num_contexts = num_contexts
        self.attr_dim = attr_dim

        # 编码到“属性空间”（对通道做 1x1 conv），输入视作 [B, C_embed, TL]
        self.enc = nn.Sequential(
            nn.Conv1d(embed_dim, attr_dim, kernel_size=1, bias=False),
            nn.GELU()
        )
        # 从属性空间解码回通道（[B, D_attr, 1] -> [B, C_embed, 1]）
        self.dec = nn.Sequential(
            nn.Conv1d(attr_dim, embed_dim, kernel_size=1, bias=False),
        )

        # 可学习的有向图（无自环），大小 D_attr×D_attr
        self.A = nn.Parameter(torch.zeros(attr_dim, attr_dim))
        self.register_buffer("I", torch.eye(attr_dim))

        # 从全局表示预测“被干预属性”的掩码 m \in [0,1]^{D_attr}
        self.rule_head = nn.Sequential(
            nn.Linear(embed_dim, attr_dim),
            nn.GELU(),
            nn.Linear(attr_dim, attr_dim)
        )

        # 残差融合系数
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):   # x: [B, C_embed, T, L]
        assert x.dim() == 4, f"CausalReasoningModule expects 4D input, got {x.shape}"
        B, C, T, L = x.shape
        TL = T * L

        # 1) 规则掩码 m（来自全局 pooled 表示）
        pooled = x.mean(dim=(2, 3))                 # [B, C_embed]
        inter_mask_logits = self.rule_head(pooled)  # [B, D_attr]
        m = torch.sigmoid(inter_mask_logits)        # [B, D_attr]

        # 2) 编码到属性空间（注意要 3D 给 Conv1d）
        z = x.contiguous().view(B, C, TL)           # [B, C_embed, TL]
        z_attr = self.enc(z)                        # [B, D_attr, TL]

        # ★ 关键修复：对 TL 做平均，得到每个样本的属性向量 [B, D_attr]
        z_mean = z_attr.mean(dim=-1)                # [B, D_attr]

        # 3) 因果图无环约束 + 干预：屏蔽进入被干预节点的边
        A = self.A - torch.diag(torch.diag(self.A)) # 去自环
        A_eff = A * (1.0 - m.unsqueeze(1))          # [D_attr, D_attr] * [B,1,D] 广播 -> [B, D_attr, D_attr]
        A_eff = A_eff if A_eff.dim() == 3 else A_eff.unsqueeze(0).expand(B, -1, -1)

        # y_attr = z_mean @ (I + A_eff)^T   -> [B, D_attr]
        IA_T = (self.I + A_eff).transpose(1, 2)     # [B, D_attr, D_attr]
        y_attr = torch.bmm(z_mean.unsqueeze(1), IA_T).squeeze(1)  # [B, D_attr]

        # 4) 解码回通道维，并广播回 [B, C_embed, T, L]
        y3 = y_attr.unsqueeze(-1)                   # [B, D_attr, 1]  —— 这是 Conv1d 期望的 3D
        y_dec = self.dec(y3)                        # [B, C_embed, 1]
        y_dec = y_dec.expand(-1, -1, TL).view(B, C, T, L)  # [B, C_embed, T, L]

        # 5) 残差融合
        out = x + self.gamma * y_dec

        # 6) 无环约束损失（对 batch 共享 A，取单标量）
        dag_loss = acyclicity_loss(self.A)

        return out, dag_loss

class PredictiveReasoningBlock(nn.Module):

    def __init__(
            self,
            in_planes,
            ou_planes,
            steps=4,
            token_len=24,
            dropout=0.1,
            num_heads=8,
            num_contexts=8,
            # num_slots=8,
            # ablation
            num_giprbs=4, num_hylas=3, reduce_planes=128
    ):

        super().__init__()
        self.m = nn.Linear(25, token_len)
        self.steps = steps
        # ablation
        # self.num_slots = num_slots
        self.num_giprbs = num_giprbs
        self.num_hylas = num_hylas
        self.reduce_planes = reduce_planes


        for l in range(self.num_giprbs):
            setattr(
                self, "map" + str(l),
                nn.Linear(token_len, token_len)
            )

        for l in range(self.num_giprbs):
            setattr(
                self, "GIPRB" + str(l),
                GIPRB(in_planes, num_contexts=num_contexts, token_len=token_len)
            )

        # self.sltA = SlotAttention(4, in_planes)
        self.conv = ConvNormAct(reduce_planes, reduce_planes, 3, 1)
        # for l in range(self.num_hylas):
        #     setattr(
        #         self, "norm" + str(l),
        #         nn.LayerNorm((num_contexts + 1, 24, reduce_planes))
        #     )
        #
        # for l in range(self.num_hylas):
        #     setattr(
        #         self, "hyla" + str(l),
        #         HYLA(reduce_planes)
        #     )
        # -------------------------------------------------------------------


        # self.slot_router_group = SlotRouterGroup(
        #     in_channels=self.reduce_planes,
        #     num_slots=4,
        #     num_giprbs=self.num_giprbs,
        #     slot_dim=32,
        #     iters=3
        # )
        #
        #
        # self.causal_reasoner = CausalReasoningModule(
        #     feature_dim=reduce_planes * token_len,
        #     rule_dim=reduce_planes * token_len,
        #     graph_dim=64
        # )
        # self.conv = ConvNormAct(128, 128, 3, 1)
        # self.norm2 = nn.LayerNorm((4, 24, 128))
        # self.norm3 = nn.LayerNorm((4, 24, 128))
        # self.cro1 = CroAttention(128)

    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.m(x)
        # slot
        # x = self.slot_router_group(x)

        # =============for ablation=========================
        chunks = torch.chunk(x, chunks=self.num_giprbs, dim=1)

        outs = []
        for g, xi in enumerate(chunks):
            xi = getattr(self, f"map{g}")(xi)
            xi = getattr(self, f"GIPRB{g}")(xi, g + 1)
            outs.append(xi)
        x = torch.cat(outs, dim=1)

        x = x.permute(0, 2, 3, 1)



        for l in range(self.num_hylas):
            x = getattr(self, f"norm{l}")(x)
            x = getattr(self, f"hyla{l}")(x, x)

        x = x.permute(0, 3, 1, 2).contiguous()

        # ================================================
        return x



class SelfAttention(nn.Module):
    def __init__(
            self,
            in_planes,
            dropout=0.1,
            num_heads=8
    ):
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.kv = nn.Linear(in_planes, in_planes * 2)
        self.num_heads = num_heads
        self.head_dim = in_planes // num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        b, t, l, c = x.shape
        shortcut = x
        q = F.normalize(self.q(x).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads * 2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = self.drop1(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)

        x = self.drop2(self.m(x.permute(0, 1, 3, 2, 4).reshape(b, t, l, c)))
        return x


class HYLA(nn.Module):
    def __init__(
            self,
            in_planes,
            dropout=0.1,
            num_heads=8
    ):
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.kv = nn.Linear(in_planes, in_planes * 2)
        self.gate = nn.Sequential(nn.Linear(in_planes, in_planes), nn.Sigmoid())
        self.num_heads = num_heads
        self.head_dim = in_planes // num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        v_init = torch.empty(1, 1, 24, in_planes)
        nn.init.xavier_normal_(v_init)
        v_init = F.normalize(v_init, dim=-1)
        self.v_init = nn.Parameter(v_init)
        self.ln = nn.LayerNorm(in_planes)

    def forward(self, q, kv):
        shortcut = q
        b, t, l, c = q.shape
        vp = self.v_init.expand(b, t, l, c)
        b_, t_, l_, c_ = b, t, l, c

        q = self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q = F.normalize(q, dim=-1)

        b, t, l, c = kv.shape
        k, v = self.kv(kv).chunk(2, dim=-1)
        k = k.reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = F.relu(self.gate(v) * torch.tanh(vp)).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)

        x = self.m(x.permute(0, 1, 3, 2, 4).reshape(b_, t_, l_, c_)) + shortcut
        return x




class Alignment(nn.Module):

    def __init__(
            self,
            in_planes,
            ou_planes,
            dropout=0.1,
            num_heads=8,
            ffn=True
    ):
        super().__init__()
        self.selfatten = SelfAttention(in_planes)
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.m = nn.Sequential(nn.Linear(in_planes, ou_planes), nn.LayerNorm(ou_planes), nn.GELU())
        self.position1 = PositionalEncoding(in_planes)
        self.position2 = PositionalEncoding(in_planes)
        self.position3 = PositionalEncoding(in_planes)
        self.position4 = PositionalEncoding(in_planes)
        self.position5 = PositionalEncoding(in_planes)
        self.position6 = PositionalEncoding(in_planes)
        self.position7 = PositionalEncoding(in_planes)
        self.position8 = PositionalEncoding(in_planes)
        self.position9 = PositionalEncoding(in_planes)
        self.ffn = ffn
        self.drop = nn.Dropout(dropout)

    def forward(self, x, num_contexts):
        b, c, t, l = x.shape
        shortcut = self.downsample(x)
        x = x.permute(0, 2, 3, 1)

        x = self.selfatten(x).permute(0, 3, 1, 2)
        out = self.drop(x) + shortcut
        if self.ffn:
            out = self.m(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class PredRNet(nn.Module):

    def __init__(self, num_filters=48, block_drop=0.0, classifier_drop=0.0,
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8,
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=8,
                 reduce_planes=128, num_hylas=3):

        super().__init__()

        channels = [num_filters, num_filters * 2, num_filters * 3, num_filters * 4]
        strides = [2, 2, 2, 2]

        # -------------------------------------------------------------------
        # frame encoder

        #sraven
        # self.symbolic_Encoder = SymbolicEncoder(5, 5)

        self.in_planes = in_channels

        for l in range(len(strides)):
            setattr(
                self, "res" + str(l),
                self._make_layer(
                    channels[l], stride=strides[l],
                    block=ResBlock, dropout=block_drop,
                )
            )
        # -------------------------------------------------------------------
        # reduced_channel = 32
        # self.position1 = LearnedAdditivePositionalEmbed(128)
        # self.position2 = LearnedAdditivePositionalEmbed(128)
        # self.position3 = LearnedAdditivePositionalEmbed(128)
        # self.position4 = LearnedAdditivePositionalEmbed(128)
        # self.position5 = LearnedAdditivePositionalEmbed(128)
        # self.position6 = LearnedAdditivePositionalEmbed(128)
        # self.position7 = LearnedAdditivePositionalEmbed(128)
        # self.position8 = LearnedAdditivePositionalEmbed(128)
        # self.position9 = LearnedAdditivePositionalEmbed(128)

        # -------------------------------------------------------------------
        # predictive coding
        self.num_contexts = num_contexts
        self.atten = Alignment(128, 512)
        self.think_branches = 1
        self.reduce_planes = reduce_planes
        # sraven
        # self.channel_reducer = ConvNormAct(4, reduce_planes, 1, 0, activate=False)
        self.channel_reducer = ConvNormAct(128, reduce_planes, 1, 0, activate=False)
        self.num_giprbs = reduce_planes // 32
        self.num_hylas = num_hylas

        for l in range(self.think_branches):
            setattr(
                self, "MAutoRR" + str(l),
                PredictiveReasoningBlock(32, 32, num_heads=8, num_contexts=self.num_contexts,
                                         num_giprbs=self.num_giprbs, num_hylas=self.num_hylas,
                                         reduce_planes=self.reduce_planes)
            )
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1,
            norm_layer=nn.BatchNorm1d,
            dropout=classifier_drop,
            hidreduce=classifier_hidreduce
        )

        self.in_channels = in_channels
        self.ou_channels = num_classes

        # predictive coding (replaced by: ObjectCentricEncoder -> MPER_OC -> CausalReasoningModule)
        self.num_contexts = num_contexts

        # 先把 ResNet 后的 128 通道矩阵改为对象槽（S=24）
        self.oc_encoder = QueryAnchoredSlots(in_channels=128, slot_dim=128, num_slots=24,num_heads=8)#ObjectCentricEncoder(in_channels=128, slot_dim=128, num_slots=24, sa_iters=3)

        # 在对象槽上做 MPER（保留你的 GIPRB 思路）
        self.mper = MPER_OC(in_planes_per_group=32, reduce_planes=reduce_planes, num_contexts=self.num_contexts, token_len=24)

        # 因果推理(替代 HYLA/DRACO)
        self.causal = CausalReasoningModule(embed_dim=reduce_planes, token_len=24, num_contexts=self.num_contexts, attr_dim=64)


    def _make_layer(self, planes, stride, dropout, block, downsample=True):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride) if stride != 1 else nn.Identity(),
                ConvNormAct(self.in_planes, planes, 1, 0, activate=False, stride=1),
            )
        else:
            downsample = nn.Identity()

        if block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride=stride, dropout=dropout)

        self.in_planes = planes

        return stage

    def forward(self, x, train=False):
        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b * n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b * n, 3, h, w)

        for l in range(4):
            x = getattr(self, "res" + str(l))(x)
        # sraven
        # x = self.symbolic_Encoder(x)

        if self.num_contexts == 8:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        elif self.num_contexts == 3:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_mnr(x, b, h, w)
        else:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v6(x, b, h, w)

        # new
        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        x = x.permute(0, 2, 1, 3)  # [BOU, C=128, T, L=25]

        # 1) 对象中心编码：25 patch -> 24 slots
        x = self.oc_encoder(x)  # [BOU, 128, T, 24]

        # 2) MPER on slots（保留你 GIPRB 思路得到 elementary rule embeddings）
        x = self.mper(x)  # [BOU, reduce_planes, T, 24]

        self._last_mper_slots = x.detach()

        # 3) 因果推理（DAG + do(·)）
        x, dag_loss = self.causal(x)  # [BOU, reduce_planes, T, 24], scalar

        # 后续分类头保持一致
        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, self.featr_dims)
        x = x.reshape(b * self.ou_channels, self.featr_dims)
        out = self.classifier(x)

        # 把 dag_loss 暴露出去，主训练循环里加权到总损失
        if self.training:
            return out.view(b, self.ou_channels), {"dag_loss": dag_loss}
        else:
            return out.view(b, self.ou_channels)

        # x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w) # B*8,9,128,25
        #
        # x = x.permute(0, 2, 1, 3) # B*8,128,9,25
        #
        # x = self.channel_reducer(x)
        #
        #
        # # # sim
        # # x, xis = getattr(self, "MAutoRR"+str(0))(x)
        # x = getattr(self, "MAutoRR" + str(0))(x)
        # # x = self.atten(x, self.num_contexts)
        #
        # x = x.reshape(b, self.ou_channels, -1)
        # x = F.adaptive_avg_pool1d(x, self.featr_dims)
        #
        # x = x.reshape(b * self.ou_channels, self.featr_dims)
        #
        # out = self.classifier(x)
        #
        # return out.view(b, self.ou_channels)  # , xis


def predrnet_raven(**kwargs):
    return PredRNet(**kwargs, num_contexts=8)


def predrnet_analogy(**kwargs):
    return PredRNet(**kwargs, num_contexts=5, num_classes=4)


def predrnet_mnr(**kwargs):
    return PredRNet(**kwargs, num_contexts=3)


def hcvarr(**kwargs):
    return HCVARR(**kwargs, num_contexts=5, num_classes=4)


def scar(**kwargs):
    return RelationNetworkSCAR(**kwargs, num_contexts=5, num_classes=4)


def pred(**kwargs):
    return Pred(**kwargs, num_contexts=5, num_classes=4)


def mm(**kwargs):
    return MM(**kwargs, num_contexts=5, num_classes=4)


def mrnet(**kwargs):
    return MRNet(**kwargs, num_contexts=5, num_classes=4)


def mrnet_price_analogy(**kwargs):
    return MRNet_PRIC(**kwargs, num_contexts=5, num_classes=4)


def mrnet_pric_raven(**kwargs):
    return MRNet_PRIC(**kwargs, num_contexts=8)


def hcv_pric_analogy(**kwargs):
    return HCV_PRIC(**kwargs, num_contexts=5, num_classes=4)


def hcv_pric_raven(**kwargs):
    return HCV_PRIC(**kwargs, num_contexts=8)