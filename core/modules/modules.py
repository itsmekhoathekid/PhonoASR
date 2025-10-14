import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
import torch.nn.init as init

class ScaledDotProductAttention(nn.Module):
    ''' 
        Scaled Dot-Product Attention 
    '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T]
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
    def get_pe(self, seq_len: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, self.d_model)
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.get_pe(seq_len).to(x.device)
        pe = pe.expand_as(x)

        x = x + pe

        return x
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnectionBase(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return self.norm(x + self.dropout(sublayer(x)))
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch ,seqlen, d_model -> batch, seqlen, vocab_size
        return torch.log_softmax(self.proj(x), dim = -1)
    
def add_nan_hook(model):
    """
    Thêm hook để kiểm tra NaN sau mỗi layer forward.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.MultiheadAttention, nn.Embedding)):
            module.register_forward_hook(make_hook(name))


def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"❌ NaN xuất hiện trong layer: {name}")
                print(f"→ Layer: {module}")
                print(f"→ Input NaN: {torch.isnan(input[0]).any().item()}")
                print(f"→ Output shape: {output.shape}")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    print(f"❌ NaN trong output[{i}] của layer {name}")
    return hook


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, input_lengths: Tensor):
        x = x.unsqueeze(1)  # (batch, 1, time, dim)
        B, C, T, F = x.shape
        for layer in self.sequential:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                d = layer.dilation[0]
                p = layer.padding[0]
                out_T = (T + 2 * p - d * (k - 1) - 1) // s + 1
                pad_len = T - input_lengths
                data_len = input_lengths
                new_len = calc_data_len(
                    result_len=out_T,
                    pad_len=pad_len,
                    data_len=data_len,
                    kernel_size=k,
                    stride=s,
                )
                T = out_T
        B, C, T, F = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, C * F)
        return x, new_len

class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnectionCM(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer, mutiplier):
        return x + mutiplier * self.norm(x + self.dropout(sublayer(x)))
    
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super(FeedForwardModule, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = Swish()
        else:
            raise ValueError("Only relu and swish are supported.")

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ConvolutionalModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super(ConvolutionalModule, self).__init__()
        self.layer_norm = LayerNormalization(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, dim)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, dim, time)
        x = self.pointwise_conv1(x)  # (batch, 2*dim, time)
        x = self.glu(x)  # (batch, dim, time)
        x = self.depthwise_conv(x)  # (batch, dim, time)
        x = self.batch_norm(x)  # (batch, dim, time)
        x = self.swish(x)  # (batch, dim, time)
        x = self.pointwise_conv2(x)  # (batch, dim, time)
        x = self.dropout(x)  # (batch, dim, time)
        return x.transpose(1, 2)  # (batch, time, dim)

def get_activation(act):


    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()

class MultiConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Multi Convolutional Spatial Gating Unit (M-CSGU)."""

    def __init__(
        self,
        size: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNormalization(n_channels)

        kernel_sizes = list(map(int, kernel_sizes.split(",")))
        no_kernels = len(kernel_sizes)

        assert (
            n_channels % no_kernels == 0
        ), f"{n_channels} input channels cannot be divided between {no_kernels} kernels"

        self.arch_type = arch_type
        if arch_type in ["sum", "weighted_sum"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        elif arch_type in ["concat", "concat_fusion"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels // no_kernels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels // no_kernels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        else:
            raise NotImplementedError(
                f"Unknown architecture type for MultiConvCGMLP: {arch_type}"
            )
        self.use_non_linear = use_non_linear
        if arch_type == "weighted_sum":
            self.kernel_prob_gen = torch.nn.Sequential(
                torch.nn.Linear(n_channels * no_kernels, no_kernels),
                torch.nn.Softmax(dim=-1),
            )
            self.depthwise_conv_fusion = None
        elif arch_type == "concat_fusion":
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = torch.nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=merge_conv_kernel,
                stride=1,
                padding=(merge_conv_kernel - 1) // 2,
                groups=n_channels,
                bias=True,
            )
        else:
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = None

        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.model_act = activation
        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=1e-6)
            torch.nn.init.ones_(conv.bias)
        if self.depthwise_conv_fusion is not None:
            torch.nn.init.normal_(self.depthwise_conv_fusion.weight, std=1e-6)
            torch.nn.init.ones_(self.depthwise_conv_fusion.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """
        x_r, x_i = x.chunk(2, dim=-1)

        x_i = self.norm(x_i).transpose(1, 2)  # (N, D/2, T)

        # TODO(gituser): Parallelize this convolution computation
        xs = []
        for conv in self.convs:
            xi = conv(x_i).transpose(1, 2)  # (N, T, D/2)
            if self.arch_type == "sum" and self.use_non_linear:
                xi = self.model_act(xi)
            xs.append(xi)

        if self.arch_type in ["sum", "weighted_sum"]:
            x = torch.stack(xs, dim=-2)
            if self.arch_type == "weighted_sum":
                prob = self.kernel_prob_gen(torch.cat(xs, dim=-1))
                x = prob.unsqueeze(-1) * x

            x_g = x.sum(dim=-2)
        else:
            x_concat = torch.cat(xs, dim=-1)  # (N, T, D)

            if self.arch_type == "concat_fusion":
                x_tmp = x_concat.transpose(1, 2)
                x_tmp = self.depthwise_conv_fusion(x_tmp)
                x_concat = x_concat + x_tmp.transpose(1, 2)

            x_g = x_concat

        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class MultiConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        if arch_type not in ["sum", "weighted_sum", "concat", "concat_fusion"]:
            raise NotImplementedError(f"Unknown MultiConvCGMLP type: {type}")

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = MultiConvolutionalSpatialGatingUnit(
            size=linear_units,
            arch_type=arch_type,
            kernel_sizes=kernel_sizes,
            merge_conv_kernel=merge_conv_kernel,
            use_non_linear=use_non_linear,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            activation=activation,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out