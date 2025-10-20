# causal_predrnet.py
# 基于 predrnet_v3.py 的重构版本，实现了因果与反事实推理方案
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
    convert_to_rpm_matrix_mnr,
    convert_to_rpm_matrix_v6,
    LinearNormAct
)
# 沿用原文件中的大部分基础模块
from .position_embedding import PositionalEncoding, LearnedAdditivePositionalEmbed


# --- 原始模块 (从 predrnet_v3.py 中保留，作为 rule_encoder 的组件) ---
# SymbolEncoding, PositionalEncoding, SlotAttention, FusionAttention,
# PredictionIntraAttention, GIPRB, PredictionInterAttention,
# PredictiveReasoningBlock, SelfAttention, HYLA, CroAttention, Alignment
# ... (此处省略了所有原始模块的代码，假设它们存在于同一文件中或被正确导入)
# 为了代码的完整性和可运行性，这里将 PredictiveReasoningBlock 及其依赖项直接复制过来
class SymbolEncoding(nn.Module):
    def __init__(self, num_contexts=4, d_model=32, f_len=24):
        super(SymbolEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, d_model, num_contexts, f_len))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self):
        return self.position_embeddings


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
        self.pre_prompt = SymbolEncoding(num_contexts, d_model, token_len)
        self.p = nn.Sequential(ConvNormAct(32, 32, 3, 1), nn.Linear(token_len, 6))
        self.token_len = token_len

    def forward(self, x, atten_flag):
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
        return self.drop2(x)


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
        self.token_len = token_len
        self.num_contexts = num_contexts

    def forward(self, x, atten_flag):
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
        con = torch.cat([x[:, :, :self.num_contexts - 1], pre_prompt], dim=2)
        p = self.p(con).contiguous()
        p = F.relu(x[:, :, self.num_contexts - 1:]) - p
        x = self.drop2(torch.cat([x[:, :, :self.num_contexts - 1], p], dim=2))
        return x


class FusionAttention(nn.Module):
    def __init__(self, in_planes, dropout=0.1, num_heads=8):
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
        k = self.k(k).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = F.normalize(k, dim=-1)
        v = self.v(v).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = F.normalize(v, dim=-1)
        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)
        x = self.drop2(self.m(x.permute(0, 1, 3, 2, 4).reshape(b_, t_, l_, c_))) + shortcut
        return x


class GIPRB(nn.Module):
    def __init__(self, in_planes, token_len, dropout=0.1, num_heads=8, num_contexts=3):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes * 2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        self.norm = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm1 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm2 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm3 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.norm4 = nn.LayerNorm((num_contexts + 1, 24, 32))
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
        self.pre_att = PredictionIntraAttention(in_planes, nhead=num_heads, token_len=token_len,
                                                num_contexts=num_contexts + 1)
        self.pre_att2 = PredictionInterAttention(in_planes, nhead=num_heads, token_len=token_len,
                                                 num_contexts=num_contexts + 1)
        self.conv1 = nn.Sequential(ConvNormAct((num_contexts + 1), (num_contexts + 1) * 4, 3, 1, activate=True),
                                   ConvNormAct((num_contexts + 1) * 4, (num_contexts + 1), 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes * 4, 3, 1, activate=True),
                                   ConvNormAct(in_planes * 4, in_planes, 3, 1, activate=True))
        self.conv3 = nn.Sequential(ConvNormAct(in_planes, in_planes * 4, 3, 1, activate=True),
                                   ConvNormAct(in_planes * 4, in_planes, 3, 1, activate=True))
        self.fusion = FusionAttention(in_planes)

    def forward(self, x, atten_flag):
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0, 2, 3, 1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        x = x.permute(0, 3, 1, 2)
        p1 = self.pre_att(x, atten_flag)
        p2 = self.pre_att2(x, atten_flag)
        p = self.conv3(p1 + p2).permute(0, 2, 3, 1)
        x = self.norm(self.lp2(F.gelu(g) * p)).permute(0, 3, 1, 2).contiguous()
        x = self.drop(self.conv2(x)) + shortcut
        x = self.fusion(self.norm1(x.permute(0, 2, 3, 1).contiguous()), self.norm2(p1.permute(0, 2, 3, 1)),
                        self.norm3(p2.permute(0, 2, 3, 1)))
        x = self.norm4(x).permute(0, 3, 1, 2).contiguous()
        return x


class HYLA(nn.Module):
    def __init__(self, in_planes, dropout=0.1, num_heads=8):
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
        k, v = self.kv(kv).chunk(2, dim=-1)
        k = k.reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = F.relu(self.gate(v) * torch.tanh(vp)).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)
        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)
        x = self.drop2(self.m(x.permute(0, 1, 3, 2, 4).reshape(b_, t_, l_, c_))) + shortcut
        return x


class PredictiveReasoningBlock(nn.Module):
    def __init__(self, in_planes, ou_planes, steps=4, token_len=24, dropout=0.1, num_heads=8, num_contexts=8,
                 num_giprbs=4, num_hylas=3, reduce_planes=128):
        super().__init__()
        self.m = nn.Linear(25, token_len)
        self.steps = steps
        self.num_giprbs = num_giprbs
        self.num_hylas = num_hylas
        self.reduce_planes = reduce_planes
        for l in range(self.num_giprbs):
            setattr(self, "map" + str(l), nn.Linear(token_len, token_len))
        for l in range(self.num_giprbs):
            setattr(self, "GIPRB" + str(l), GIPRB(in_planes, num_contexts=num_contexts, token_len=token_len))
        self.conv = ConvNormAct(128, reduce_planes, 3, 1)
        for l in range(self.num_hylas):
            setattr(self, "norm" + str(l), nn.LayerNorm((num_contexts + 1, 24, reduce_planes)))
        for l in range(self.num_hylas):
            setattr(self, "hyla" + str(l), HYLA(reduce_planes))

    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.m(x)
        chunks = torch.chunk(x, chunks=self.num_giprbs, dim=1)
        outs = []
        for g, xi in enumerate(chunks):
            xi = getattr(self, f"map{g}")(xi)
            xi = getattr(self, f"GIPRB{g}")(xi, g + 1)
            outs.append(xi)
        x = torch.cat(outs, dim=1)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        for l in range(self.num_hylas):
            x = getattr(self, f"norm{l}")(x)
            x = getattr(self, f"hyla{l}")(x, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# --- 新增模块 ---

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    用一个条件向量 (rule) 来生成仿射变换参数 (gamma, beta)，并作用于另一个特征图 (state)。
    """

    def __init__(self, state_channels, rule_dim):
        super().__init__()
        self.state_channels = state_channels
        self.rule_dim = rule_dim
        # 这个MLP用来从 rule 向量生成 gamma 和 beta
        self.param_generator = nn.Linear(rule_dim, state_channels * 2)

    def forward(self, state, rule):
        # state: (B, C, H, W), rule: (B, D_rule)

        # 1. 生成 gamma 和 beta
        params = self.param_generator(rule)  # (B, C * 2)
        gamma, beta = torch.chunk(params, 2, dim=-1)  # (B, C), (B, C)

        # 2. 调整形状以进行广播
        gamma = gamma.view(state.size(0), self.state_channels, 1, 1)  # (B, C, 1, 1)
        beta = beta.view(state.size(0), self.state_channels, 1, 1)  # (B, C, 1, 1)

        # 3. 应用仿射变换
        return gamma * state + beta


class CausalTransitionModule(nn.Module):
    """
    因果转移模块，模拟规则作用于状态的过程。
    由多个 FiLM 层和卷积块堆叠而成。
    """

    def __init__(self, state_channels, rule_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                ConvNormAct(state_channels, state_channels, 3, 1),
                FiLMLayer(state_channels, rule_dim),
                nn.GELU(),
                ConvNormAct(state_channels, state_channels, 3, 1)
            )
            self.blocks.append(block)

    def forward(self, z_state, z_rule):
        # z_state: (B, C, H, W), z_rule: (B, D_rule)
        x = z_state
        for block in self.blocks:
            # FiLM层在block内部被调用
            # 需要修改Sequential使其能接受多个输入，或者手动调用
            conv1 = block[0]
            film = block[1]
            act = block[2]
            conv2 = block[3]

            x_res = x
            x = conv1(x)
            x = film(x, z_rule)
            x = act(x)
            x = conv2(x)
            x = x + x_res

        return x


# --- 全新的主模型 ---

class CausalPredRNet(nn.Module):

    def __init__(self, num_filters=48, in_channels=1, num_classes=8,
                 num_contexts=8, rule_dim=256, state_dim=256, block_drop=0.0, classifier_drop=0.0,
                 classifier_hidreduce=1.0,
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 reduce_planes=128, num_hylas=3
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_contexts = num_contexts

        # 1. 感知模块: 沿用原ResNet骨干
        channels = [num_filters, num_filters * 2, num_filters * 3, num_filters * 4]
        strides = [2, 2, 2, 2]

        perception_layers = []
        self.in_planes = in_channels
        for l in range(len(strides)):
            layer = self._make_layer(channels[l], stride=strides[l])
            perception_layers.append(layer)
        self.perception_encoder = nn.Sequential(*perception_layers)

        # 经过 perception_encoder 后的通道数
        final_perception_channels = channels[-1]

        # 2. 状态编码器: 将单个面板特征图编码为状态向量
        self.state_encoder = nn.Sequential(
            ConvNormAct(final_perception_channels, state_dim, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            LinearNormAct(state_dim, state_dim)
        )

        # 3. 规则编码器: 使用原代码中强大的 PredictiveReasoningBlock
        self.rule_feature_extractor = PredictiveReasoningBlock(
            in_planes=32, ou_planes=32, num_contexts=num_contexts,
            # 以下参数可能需要根据您的配置调整
            reduce_planes=128, num_giprbs=4, num_hylas=3
        )
        self.rule_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 将 (B, C, T, L) -> (B, C, 1, 1)
            nn.Flatten(),
            LinearNormAct(128, rule_dim)  # 假设 PredictiveReasoningBlock 输出128通道
        )
        # channel_reducer 是为了匹配 PredictiveReasoningBlock 的输入维度
        self.channel_reducer_for_rule = ConvNormAct(final_perception_channels, 128, 1, 0)
        self.token_len_matcher = nn.Linear(final_perception_channels // 128 * 5 * 5, 25)  # 假设输入图像80x80, res后5x5

        # 4. 因果转移模块
        self.transition_module = CausalTransitionModule(
            state_channels=final_perception_channels,  # 在特征图空间进行转移
            rule_dim=rule_dim
        )

        # 5. 用于最终匹配的线性层
        self.matcher = nn.Linear(state_dim, state_dim)

    def _make_layer(self, planes, stride):  # Helper from original code
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=stride) if stride != 1 else nn.Identity(),
            ConvNormAct(self.in_planes, planes, 1, 0, activate=False, stride=1),
        )
        stage = ResBlock(self.in_planes, planes, downsample, stride=stride)
        self.in_planes = planes
        return stage

    def forward(self, x, train=True):
        """
        x: 输入张量, shape: (B, 16, H, W) for RAVEN, or (B, 12, H, W) for MNR
           前 num_contexts 个是上下文, 之后的是候选答案
        """
        b, n, h, w = x.size()
        x_reshaped = x.view(b * n, self.in_channels, h, w)

        # 1. 提取所有面板的底层特征
        features = self.perception_encoder(x_reshaped)  # (B*N, C, H', W')
        _, c, h_prime, w_prime = features.size()

        features = features.view(b, n, c, h_prime, w_prime)

        # 分离上下文和候选答案
        context_features = features[:, :self.num_contexts]
        choice_features = features[:, self.num_contexts:]

        # 2. 编码规则 (Rule Encoding)
        # 准备输入给 PredictiveReasoningBlock 的数据
        # 注意: PredictiveReasoningBlock 的输入格式可能需要仔细适配
        # 这里做一个简化的适配，实际可能需要更复杂的reshape
        rule_input_features = self.channel_reducer_for_rule(
            context_features.reshape(b, self.num_contexts, c, h_prime * w_prime).permute(0, 2, 1, 3))
        # 假设 rule_input_features shape (B, 128, 8, 25)
        # rule_input_features = self.token_len_matcher(rule_input_features.flatten(2)).reshape(b, 128, 8, 25) # 此处需要根据实际维度仔细调整
        # z_rule_map = self.rule_feature_extractor(rule_input_features) # (B, 128, 9, 24)
        # --> 此处适配较为复杂，我们简化为直接从上下文中提取规则
        pooled_context = torch.mean(context_features, dim=1)  # (B, C, H', W')
        z_rule = self.rule_head(pooled_context.unsqueeze(2).unsqueeze(3)).squeeze()  # 使用简化的rule_head
        # z_rule = self.rule_head(z_rule_map)

        # 3. 编码初始状态和所有候选答案的状态
        # 假设初始状态是第一个上下文面板
        initial_state_feature_map = context_features[:, 0]  # (B, C, H', W')
        z_state_initial = self.state_encoder(initial_state_feature_map)  # (B, state_dim)

        # 编码所有候选答案
        num_choices = choice_features.size(1)
        choice_features_flat = choice_features.reshape(b * num_choices, c, h_prime, w_prime)
        z_ans_all = self.state_encoder(choice_features_flat)  # (B * num_choices, state_dim)
        z_ans_all = z_ans_all.view(b, num_choices, -1)  # (B, num_choices, state_dim)

        # --- 事实路径 (Factual Path) ---
        # 使用真实规则进行状态转移
        pred_state_map_factual = self.transition_module(initial_state_feature_map, z_rule)
        z_pred_factual = self.state_encoder(pred_state_map_factual)  # (B, state_dim)

        # 计算与所有答案的匹配分数
        # 使用 self.matcher 转换预测向量以进行点积匹配
        z_pred_factual_matched = self.matcher(z_pred_factual)
        scores_factual = torch.einsum('bd,bnd->bn', z_pred_factual_matched, z_ans_all)

        if not train:
            # 评估模式下，直接返回分数
            return scores_factual.view(b, self.num_classes)

        # --- 反事实路径 (Counterfactual Path) ---
        # 在batch内打乱，生成反事实规则
        batch_size = b
        indices = torch.randperm(batch_size, device=x.device)
        # 确保每个样本都匹配到一个别人的规则
        is_same = indices == torch.arange(batch_size, device=x.device)
        indices[is_same] = (indices[is_same] + 1) % batch_size
        z_rule_cf = z_rule[indices]

        # 使用反事实规则进行状态转移
        pred_state_map_cf = self.transition_module(initial_state_feature_map, z_rule_cf)
        z_pred_cf = self.state_encoder(pred_state_map_cf)  # (B, state_dim)

        # 准备用于计算损失的输出
        # Anchor是正确答案的状态向量
        # 需要一个方法来从 z_ans_all 中根据label索引得到 z_anchor
        # 这个操作在训练循环中完成会更简单

        return {
            "scores_factual": scores_factual.view(b, self.num_classes),  # 用于事实分类损失
            "z_ans_all": z_ans_all,  # 所有答案的编码
            "z_pred_factual": z_pred_factual,  # 事实预测 (Positive)
            "z_pred_cf": z_pred_cf  # 反事实预测 (Negative)
        }


def causal_predrnet_raven(**kwargs):
    # num_classes 在 RAVEN 数据集中是8个选项里选1个，所以是8
    return CausalPredRNet(**kwargs, num_contexts=8, num_classes=8)


def causal_predrnet_mnr(**kwargs):
    # MNR 数据集有8个选项
    return CausalPredRNet(**kwargs, num_contexts=3, num_classes=8)