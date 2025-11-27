import torch
import torch.nn as nn


class CausalGraphDiscovery(nn.Module):
    """
    因果图发现模块

    任务：从MPER提供的基本规则嵌入中，学习一个潜在的、可微的因果图。
    这个图用一个邻接矩阵表示，描述了不同特征维度（可视为抽象属性）之间的因果关系。
    """

    def __init__(self, rule_dim, graph_dim):
        super().__init__()
        self.rule_dim = rule_dim
        self.graph_dim = graph_dim

        # 一个简单的MLP，将平均后的规则嵌入映射到邻接矩阵的 logits
        self.mlp = nn.Sequential(
            nn.Linear(rule_dim, rule_dim * 2),
            nn.ReLU(),
            nn.Linear(rule_dim * 2, graph_dim * graph_dim)
        )

    def forward(self, context_rule_embeddings):
        """
        Args:
            context_rule_embeddings (Tensor): 来自MPER的、关于上下文面板的规则嵌入。
                                             Shape: (B, D_rule) or (B, N_rules, D_rule)

        Returns:
            torch.Tensor: 可微的邻接矩阵。Shape: (B, G_dim, G_dim)
        """
        # 如果有多个规则，先取平均来获得一个全局的规则表示
        if context_rule_embeddings.dim() > 2:
            avg_rule_embedding = context_rule_embeddings.mean(dim=1)
        else:
            avg_rule_embedding = context_rule_embeddings

        # 1. 生成邻接矩阵的 logits
        adj_matrix_logits = self.mlp(avg_rule_embedding).view(-1, self.graph_dim, self.graph_dim)

        # 2. 使用 Gumbel-Softmax 生成可微的、类似one-hot的邻接矩阵
        #    - tau 控制平滑度，越小越接近离散
        #    - hard=True 让前向传播是离散的(0/1)，但反向传播是可微的，这对于学习图结构至关重要
        adj_matrix = F.gumbel_softmax(adj_matrix_logits, tau=0.8, hard=True, dim=-1)

        # 3. 强制无自环 (causal graphs are DAGs, no self-loops is a necessary condition)
        #    创建一个对角线为0，其余为1的掩码
        mask = 1.0 - torch.eye(self.graph_dim, device=adj_matrix.device).unsqueeze(0)
        adj_matrix = adj_matrix * mask

        return adj_matrix


class CausalInterventionEngine(nn.Module):
    """
    因果干预引擎

    任务：接收学习到的因果图、干预前的状态（最后一个context panel）以及干预规则，
          通过模拟干预来预测干预后的状态（answer panel）。
    """

    def __init__(self, feature_dim, rule_dim, graph_dim):
        super().__init__()

        # 线性层，用于将不同维度的输入映射到统一的因果空间 (graph_dim)
        self.feature_to_node = nn.Linear(feature_dim, graph_dim)
        self.rule_to_intervention = nn.Linear(rule_dim, graph_dim)

        # 一个简单的图神经网络(GNN)层，用于在图上传播干预效果
        # 这里用一个线性层来模拟消息传递函数
        self.gnn_layer = nn.Linear(graph_dim, graph_dim)

        # 线性层，用于将最终的因果状态映射回特征空间，以便与候选答案进行比较
        self.node_to_feature = nn.Linear(graph_dim, feature_dim)

        self.norm1 = nn.LayerNorm(graph_dim)
        self.norm2 = nn.LayerNorm(graph_dim)

    def forward(self, last_context_features, intervention_rule, causal_graph):
        """
        Args:
            last_context_features (Tensor): 干预前的状态 (最后一个context panel的特征)。Shape: (B, D_feat)
            intervention_rule (Tensor):     要施加的干预 (从context到target的规则)。Shape: (B, D_rule)
            causal_graph (Tensor):          学习到的因果图邻接矩阵。Shape: (B, G_dim, G_dim)

        Returns:
            torch.Tensor: 预测出的答案面板特征。Shape: (B, D_feat)
        """

        # 1. 将输入映射到因果空间的节点表示
        #    这是我们对世界状态的初始认知
        initial_state = self.feature_to_node(last_context_features)

        # 2. 将规则映射为干预向量
        #    这代表了我们要对世界施加的改变
        intervention_vector = self.rule_to_intervention(intervention_rule)

        # 3. 应用干预 (模拟 do-operation)
        #    最简单的模拟方式是将干预向量加到初始状态上
        intervened_state = self.norm1(initial_state + intervention_vector)

        # 4. 在因果图上传播干预效果
        #    - 首先，通过GNN层处理每个节点自身的信息 (应用变换)
        messages = F.relu(self.gnn_layer(intervened_state))
        #    - 然后，根据因果图的连接关系（邻接矩阵）聚合信息
        #    - torch.bmm 实现了批量矩阵乘法，正是我们需要的信息传播方式
        propagated_state = torch.bmm(causal_graph, messages.unsqueeze(2)).squeeze(2)

        # 5. 残差连接，并再次归一化，得到最终的因果状态
        final_state = self.norm2(intervened_state + propagated_state)

        # 6. 将最终状态映射回特征空间，得到预测结果
        predicted_features = self.node_to_feature(final_state)

        return predicted_features


class CausalReasoningModule(nn.Module):
    """
    完整的因果推理模块，封装了图发现和干预引擎。
    这个模块将替换掉你代码中的DRACO (即HYLA层)。
    """

    def __init__(self, feature_dim, rule_dim, graph_dim=64):
        super().__init__()
        self.graph_discovery = CausalGraphDiscovery(rule_dim=rule_dim, graph_dim=graph_dim)
        self.intervention_engine = CausalInterventionEngine(feature_dim=feature_dim, rule_dim=rule_dim,
                                                            graph_dim=graph_dim)

    def forward(self, last_context_features, mper_outputs):
        """
        Args:
            last_context_features (Tensor): 最后一个上下文面板的特征。Shape: (B, D_feat)
            mper_outputs (Tensor):          MPER模块的完整输出。Shape: (B, num_contexts+1, D_rule)
                                            其中num_contexts是上下文面板数量 (e.g., 8 for RAVEN, 3 for MNR)

        Returns:
            torch.Tensor: 预测的答案特征。Shape: (B, D_feat)
        """
        # MPER的输出包含了对所有上下文面板和问题面板的分析
        # 我们假设前 num_contexts 个嵌入是关于上下文的，最后一个是关于问题->答案的转换规则
        context_rule_embeddings = mper_outputs[:, :-1, :]
        intervention_rule = mper_outputs[:, -1, :]

        # 1. 从上下文规则中发现因果图
        causal_graph = self.graph_discovery(context_rule_embeddings)

        # 2. 执行因果干预来预测结果
        predicted_features = self.intervention_engine(last_context_features, intervention_rule, causal_graph)

        return predicted_features

def convert_to_rpm_matrix_v9(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 16, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:8], output[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], 
        dim=1
    )

    return output
def convert_to_rpm_matrix_v9_mask12(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 16, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:6], output[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], 
        dim=1
    )

    return output
def convert_to_rpm_matrix_v15(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    # 14 + 8 = 22
    output = input.reshape(b, 22, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:14], output[:,i].unsqueeze(1)), dim=1) for i in range(14, 22)],
        dim=1
    )

    return output

def convert_to_rpm_matrix_v15_mask34(input, b, h, w):
    output = input.reshape(b, 22, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:12], output[:,i].unsqueeze(1)), dim=1) for i in range(14, 22)],
        dim=1
    )

    return output

def convert_to_rpm_matrix_v15_mask4(input, b, h, w):
    output = input.reshape(b, 22, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:13], output[:,i].unsqueeze(1)), dim=1) for i in range(14, 22)],
        dim=1
    )

    return output

def convert_to_rpm_matrix_v15_mask234(input, b, h, w):
    output = input.reshape(b, 22, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:11], output[:,i].unsqueeze(1)), dim=1) for i in range(14, 22)],
        dim=1
    )

    return output
def convert_to_rpm_matrix_v15_mask3(input, b, h, w):
    output = input.reshape(b, 22, -1, h, w)
    contexts = torch.cat((output[:,:12], output[:,13:14]), dim=1)
    output = torch.stack(
        [torch.cat((contexts, output[:,i].unsqueeze(1)), dim=1) for i in range(14, 22)],
        dim=1
    )

    return output
def convert_to_rpm_matrix_v15_mask1234(input, b, h, w):
    output = input.reshape(b, 22, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:10], output[:,i].unsqueeze(1)), dim=1) for i in range(14, 22)],
        dim=1
    )

    return output
def convert_to_rpm_matrix_v6(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 9, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:5], output[:,i].unsqueeze(1)), dim=1) for i in range(5, 9)], 
        dim=1
    )

    return output


def convert_to_rpm_matrix_mnr(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 11, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:3], output[:,i].unsqueeze(1)), dim=1) for i in range(3, 11)], 
        dim=1
    )

    return output



def ConvAct(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    if activate:
        block += [nn.GELU()]
    
    return nn.Sequential(*block)


def LinearNormAct(
        inplanes, ouplanes, activate=True
    ):

    block = [nn.Linear(inplanes, ouplanes)]
    block += [nn.BatchNorm3d(ouplanes)]
    if activate:
        block += [nn.GELU()]
    
    return nn.Sequential(*block)


def ConvNormAct(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm2d(ouplanes)]
    if activate:
        block += [nn.GELU()]
    
    return nn.Sequential(*block)


def ConvNormAct1D(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv1d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm1d(ouplanes)]
    if activate:
        block += [nn.GELU()]
    
    return nn.Sequential(*block)


class ResBlock(nn.Module):

    def __init__(self, inplanes, ouplanes, downsample, stride=1, dropout=0.0):
        super().__init__()

        mdplanes = ouplanes

        self.conv1 = ConvNormAct(inplanes, mdplanes, 3, 1, stride=stride)
        self.conv2 = ConvNormAct(mdplanes, mdplanes, 3, 1)
        self.conv3 = ConvNormAct(mdplanes, ouplanes, 3, 1)

        self.downsample = downsample
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        return out

class Classifier(nn.Module):

    def __init__(self, inplanes, ouplanes, norm_layer=nn.BatchNorm2d, dropout=0.0, hidreduce=1.0):
        super().__init__()

        midplanes = inplanes // hidreduce

        self.mlp = nn.Sequential(
            nn.Linear(inplanes, midplanes, bias=False),
            norm_layer(midplanes),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(midplanes, ouplanes)
        )

    def forward(self, x):
        return self.mlp(x)

class SubspaceProjector(nn.Module):
    """
    可学习的语义子空间映射：
    - 使用 1x1 Conv 将 reduce_planes -> K * group_planes
    - 用一个 router(1x1 Conv) 产生 K 路 gate（softmax/sigmoid），对每个子空间做软分配
    约定输入/输出形状均为 (B, C, T, L)，其中 T = num_contexts+1, L = token_len
    """
    def __init__(self, in_planes: int, group_planes: int, K: int,
                 router: str = "softmax", temperature: float = 1.0):
        super().__init__()
        assert K >= 1 and in_planes >= group_planes
        self.in_planes = in_planes
        self.group_planes = group_planes
        self.K = K
        self.temperature = temperature
        self.router_type = router

        # 投影到 K 个子空间（每个子空间 group_planes 通道）
        self.proj = nn.Conv2d(in_planes, K * group_planes, kernel_size=1, bias=False)
        # 软路由门控（每个位置对 K 个子空间的分配）
        self.router = nn.Conv2d(in_planes, K, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (B, C=in_planes, T, L)
        B, C, T, L = x.shape
        z = self.proj(x)                       # (B, K*group_planes, T, L)
        z = z.view(B, self.K, self.group_planes, T, L)  # (B, K, G, T, L)

        gates = self.router(x)                 # (B, K, T, L)
        if self.router_type == "softmax":
            pi = torch.softmax(gates / self.temperature, dim=1)  # K 维归一化
        else:
            pi = torch.sigmoid(gates / self.temperature)         # 独立门控

        z = z * pi.unsqueeze(2)                # (B, K, G, T, L) * (B, K, 1, T, L)
        # 返回 K 个 (B, G, T, L)
        return [z[:, k, ...] for k in range(self.K)]

    def orthogonal_loss(self) -> torch.Tensor:
        """
        可选：对子空间投影权重施加（组间）正交/去相关正则。
        使用时手动把该 loss 加到总损失里即可；默认不影响前向/训练。
        """
        W = self.proj.weight.view(self.K, self.group_planes, self.in_planes)  # (K, G, C)
        Wf = F.normalize(W.reshape(self.K, -1), dim=1)                        # (K, G*C)
        gram = Wf @ Wf.t()                                                    # (K, K)
        off_diag = gram - torch.eye(self.K, device=W.device, dtype=W.dtype)
        return (off_diag ** 2).sum()
