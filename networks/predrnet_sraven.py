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
    LinearNormAct
)
from .position_embedding import PositionalEncoding, LearnedAdditivePositionalEmbed
from .HCVARR import HCVARR
from .SCAR import RelationNetworkSCAR
from .Pred import Pred
from .MM import MM
from .MRnet import MRNet
from torch.nn import init

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
            num_slots=8,
            # ablation
            num_giprbs=4, num_hylas=3, reduce_planes=128
    ):

        super().__init__()
        self.m = nn.Linear(25, token_len)
        self.steps = steps
        # ablation
        self.num_slots = num_slots
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
        for l in range(self.num_hylas):
            setattr(
                self, "norm" + str(l),
                nn.LayerNorm((num_contexts + 1, 24, reduce_planes))
            )

        for l in range(self.num_hylas):
            setattr(
                self, "hyla" + str(l),
                HYLA(reduce_planes)
            )

        self.slot_router_group = SlotRouterGroup(
            in_channels=self.reduce_planes,
            num_slots=32,
            num_giprbs=self.num_giprbs,  # 例如 4
            slot_dim=32,  # 和你每路的通道对齐
            iters=3
        )

        # self.conv = ConvNormAct(128, 128, 3, 1)
        # self.norm2 = nn.LayerNorm((4, 24, 128))
        # self.norm3 = nn.LayerNorm((4, 24, 128))
        # self.cro1 = CroAttention(128)

    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.m(x)
        # slot
        x = self.slot_router_group(x)

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


class CroAttention(nn.Module):
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

    def forward(self, q, kv):
        shortcut = q
        b, t, l, c = q.shape
        b_, t_, l_, c_ = b, t, l, c

        q = self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q = F.normalize(q, dim=-1)

        b, t, l, c = kv.shape
        k, v = self.kv(kv).reshape(b, t, l, self.num_heads * 2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)

        x = self.drop2(self.m(x.permute(0, 1, 3, 2, 4).reshape(b_, t_, l_, c_))) + shortcut
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

        # if num_contexts == 3:
        #     c1, c2, c3, c4 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3])
        #     x = torch.stack([c1, c2, c3, c4], dim=1)
        # elif num_contexts == 5:
        #     c1, c2, c3, c4, c5, c6 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3]), self.position1(x[:,4]), self.position2(x[:,5])
        #     x = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
        # elif num_contexts == 8:
        #     c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3]), self.position1(x[:,4]), self.position2(x[:,5]), self.position3(x[:,6]), self.position4(x[:,7]), self.position4(x[:,8])
        #     x = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
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

        # sraven
        self.symbolic_Encoder = SymbolicEncoder(5, 5)

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
        reduced_channel = 32
        self.position1 = LearnedAdditivePositionalEmbed(128)
        self.position2 = LearnedAdditivePositionalEmbed(128)
        self.position3 = LearnedAdditivePositionalEmbed(128)
        self.position4 = LearnedAdditivePositionalEmbed(128)
        self.position5 = LearnedAdditivePositionalEmbed(128)
        self.position6 = LearnedAdditivePositionalEmbed(128)
        self.position7 = LearnedAdditivePositionalEmbed(128)
        self.position8 = LearnedAdditivePositionalEmbed(128)
        self.position9 = LearnedAdditivePositionalEmbed(128)

        # -------------------------------------------------------------------
        # predictive coding
        self.num_contexts = num_contexts
        self.atten = Alignment(128, 512)
        self.think_branches = 1
        self.reduce_planes = reduce_planes
        # sraven
        self.channel_reducer = ConvNormAct(4, reduce_planes, 1, 0, activate=False)
        # self.channel_reducer = ConvNormAct(128, reduce_planes, 1, 0, activate=False)
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
        x = self.symbolic_Encoder(x)

        if self.num_contexts == 8:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        elif self.num_contexts == 3:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_mnr(x, b, h, w)
        else:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v6(x, b, h, w)

        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        x = x.permute(0, 2, 1, 3)

        x = self.channel_reducer(x)
        # x = x.permute(0,2,3,1)
        # if self.num_contexts == 3:
        #     c1, c2, c3, c4 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3])
        #     x = torch.stack([c1, c2, c3, c4], dim=1)
        # elif self.num_contexts == 8:
        #     c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3]), self.position1(x[:,4]), self.position2(x[:,5]), self.position3(x[:,6]), self.position4(x[:,7]), self.position4(x[:,8])
        #     x = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
        # x = x.permute(0, 3, 1, 2)

        # # sim
        # x, xis = getattr(self, "MAutoRR"+str(0))(x)
        x = getattr(self, "MAutoRR" + str(0))(x)
        # x = self.atten(x, self.num_contexts)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, self.featr_dims)

        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        return out.view(b, self.ou_channels)  # , xis


def predrnet_sraven(**kwargs):
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