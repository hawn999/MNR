import itertools
import torch.nn.functional as F
import torch
import torch.distributed as dist
from enum import Enum

from torch import nn


def total_correlation_penalty(slots: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    slots: [BOU, C, T, S] 或 [B, S, D]
    我们统一把槽维度展平成 [B_like, F]，对特征做零均值、单位方差，再惩罚相关矩阵的 off-diagonal。
    """
    if slots.dim() == 4:
        B, C, T, S = slots.shape
        Z = slots.permute(0, 2, 3, 1).contiguous().view(B*T, S*C)  # [B*T, S*C]
    elif slots.dim() == 3:
        Z = slots.view(slots.size(0), -1)
    else:
        raise ValueError("Unexpected slots shape")

    Z = Z - Z.mean(dim=0, keepdim=True)
    Z = Z / (Z.std(dim=0, keepdim=True) + eps)
    COV = (Z.T @ Z) / (Z.size(0) - 1)           # 相关矩阵
    off_diag = COV - torch.diag(torch.diag(COV))
    tc = (off_diag ** 2).mean()
    return tc
def evaluate_cosine_similarity(data_loader, model, args):
    """
    在整个数据集上计算、聚合并报告中间特征的余弦相似度。
    """
    model.eval()  # 切换到评估模式

    # 初始化一个字典来存储每一对特征的相似度结果列表
    # { 'sim_0_1': [], 'sim_0_2': [], ... }
    similarity_storage = {}

    print("Starting cosine similarity analysis on the test set...")
    with torch.no_grad():  # 关闭梯度计算
        for i, (images, target, _, _, _) in enumerate(data_loader):
            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)

            images = normalize_image(images)

            # 调用修改后的模型，获取中间特征
            # _, intermediate_outs = model(images)
            model_to_run = model.module if isinstance(model, torch.nn.DataParallel) else model
            _, intermediate_outs = model_to_run(images)

            # `intermediate_outs` 是一个包含4个张量的列表, shape: (b, c, t, l)
            # 1. 展平张量
            flattened_outs = [t.flatten(start_dim=1) for t in intermediate_outs]

            # 2. 计算所有唯一组合的余弦相似度
            num_chunks = len(flattened_outs)
            for i_chunk, j_chunk in itertools.combinations(range(num_chunks), 2):
                pair_key = f'sim({i_chunk},{j_chunk})'
                if pair_key not in similarity_storage:
                    similarity_storage[pair_key] = []

                # 计算当前批次的相似度
                sim_batch = F.cosine_similarity(flattened_outs[i_chunk], flattened_outs[j_chunk], dim=1)

                # 存储当前批次的结果
                similarity_storage[pair_key].append(sim_batch.cpu())

    print("\nAnalysis finished. Aggregating results...")
    print("=" * 50)

    # 聚合所有批次的结果并打印
    for pair_key, sim_list in similarity_storage.items():
        # 将列表中的所有张量拼接成一个大张量
        all_similarities = torch.cat(sim_list, dim=0)

        # 确保所有样本都被计算
        # assert len(all_similarities) == len(data_loader.dataset)

        # 计算平均值和标准差
        mean_sim = all_similarities.mean().item()
        std_sim = all_similarities.std().item()

        # 打印最终结果
        print(f"Pair {pair_key}:")
        print(f"  - Mean Similarity: {mean_sim:.4f}")
        print(f"  - Std. Deviation:  {std_sim:.4f}")
        print(f"  - Report as:       {mean_sim:.4f} ± {std_sim:.4f}\n")
    print("=" * 50)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def get_message(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ('\t').join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)


        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def parse_gpus(gpu_ids):
    if gpu_ids == None:
        return None
    gpus = gpu_ids.split(',')
    gpu_ids = []
    for g in gpus:
        g_int = int(g)
        if g_int >= 0:
            gpu_ids.append(g_int)
    if not gpu_ids:
        return None
    return gpu_ids


class ToTensor(object):
    def __call__(self, inputs):
        if type(inputs) == torch.Tensor:
            return inputs
        else:
            return torch.from_numpy(inputs).type(torch.float32)

def normalize_image(images):
    return (images / 255.0 - 0.5) * 2

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
