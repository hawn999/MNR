"""Implementation of different positional embeddings."""
import torch
from torch import nn
import math

class SoftPositionEmbed(nn.Module):
    """Embeding of positions using convex combination of learnable tensors.

    This assumes that the input positions are between 0 and 1.
    """

    def __init__(
        self, n_spatial_dims: int, feature_dim: int, cnn_channel_order=False, savi_style=False
    ):
        """__init__.

        Args:
            n_spatial_dims (int): Number of spatial dimensions.
            feature_dim (int): Dimensionality of the input features.
            cnn_channel_order (bool): Assume features are in CNN channel order (i.e. C x H x W).
            savi_style (bool): Use savi style positional encoding, where positions are normalized
                between -1 and 1 and a single dense layer is used for embedding.
        """
        super().__init__()
        self.savi_style = savi_style
        n_features = n_spatial_dims if savi_style else 2 * n_spatial_dims
        self.dense = nn.Linear(in_features=n_features, out_features=feature_dim)
        self.cnn_channel_order = cnn_channel_order

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        if self.savi_style:
            # Rescale positional encoding to -1 to 1
            positions = (positions - 0.5) * 2
        else:
            positions = torch.cat([positions, 1 - positions], axis=-1)
        emb_proj = self.dense(positions)
        if self.cnn_channel_order:
            emb_proj = emb_proj.permute(*range(inputs.ndim - 3), -1, -3, -2)
        return inputs + emb_proj


class LearnedAdditivePositionalEmbed(nn.Module):
    """Add positional encoding as in SLATE."""

    def __init__(self, d_model, max_len=25, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])

class LearnedAdditivePositionalEmbed_slot(nn.Module):
    """
    (修正后的实现)
    一个标准的可学习的二维位置编码模块。
    它创建一个与输入特征图空间维度相同的、可学习的参数`pe`，并将其直接加到输入上。
    """

    def __init__(self, dim, spatial_dims):
        super().__init__()
        # spatial_dims 是一个元组，例如 (height, width)
        if isinstance(spatial_dims, int):
            spatial_dims = (spatial_dims, spatial_dims)

        # 创建一个形状为 (1, channel_dim, height, width) 的可学习参数
        self.pe = nn.Parameter(torch.zeros(1, dim, *spatial_dims))
        # 使用标准正态分布进行初始化
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        # x 的形状是 (batch, channel_dim, height, width)
        # self.pe 会通过广播机制自动扩展到 batch 维度，然后与 x 相加
        if x.shape[2:] != self.pe.shape[2:]:
            raise ValueError(
                f"Input spatial dimensions {x.shape[2:]} do not match "
                f"positional embedding dimensions {self.pe.shape[2:]}"
            )
        return x + self.pe

class DummyPositionEmbed(nn.Module):
    """Embedding that just passes through inputs without adding any positional embeddings."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        return inputs

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
