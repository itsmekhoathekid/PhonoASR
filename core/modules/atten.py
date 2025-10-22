from .modules import ScaledDotProductAttention, FeedForwardBlock, ResidualConnection
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
from typing import Optional
import torch

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        # q = self.dropout(self.fc(q))
        # q += residual

        # q = self.layer_norm(q)

        return q, attn


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # print("Mask shape:", mask.shape)  # [B, T]
            # print("attention_scores shape:", attention_scores.shape)  # [B, h, T    , T]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)




class PositionalEncoder(nn.Module):
    """
    Generate positional encodings used in the relative multi-head attention module.
    Same encodings as the original transformer model [Attention Is All You Need]:
    https://arxiv.org/abs/1706.03762

    Parameters:
      max_len (int): Maximum sequence length (time dimension)

    Inputs:
      len (int): Length of encodings to retrieve

    Outputs
      Tensor (len, d_model): Positional encodings
    """

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        encodings = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        self.register_buffer("encodings", encodings)

    def forward(self, len):
        return self.encodings[:len, :]

class RelativeMultiHeadAttention(nn.Module):
    """
    Relative Multi-Head Self-Attention Module.
    Method proposed in Transformer-XL paper: https://arxiv.org/abs/1901.02860

    Parameters:
      d_model (int): Dimension of the model
      num_heads (int): Number of heads to split inputs into
      dropout (float): Dropout probability
      positional_encoder (nn.Module): PositionalEncoder module

    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices

    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the attention module.

    """

    def __init__(
        self,
        d_model=144,
        num_heads=4,
        dropout=0.1,
        positional_encoder=PositionalEncoder(144),
    ):
        super(RelativeMultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_pos = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)

        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
        self.positional_encoder = positional_encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        x = self.layer_norm(x)
        pos_emb = self.positional_encoder(seq_length)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
        k = (
            self.W_k(x)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.W_v(x)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .permute(0, 2, 3, 1)
        )
        pos_emb = (
            self.W_pos(pos_emb)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 3, 1)
        )

        AC = torch.matmul((q + self.u).transpose(1, 2), k)
        BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        BD = self.rel_shift(BD)
        attn = (AC + BD) / math.sqrt(self.d_model)

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask_value = -1e30 if attn.dtype == torch.float32 else -1e4
            attn.masked_fill_(mask, mask_value)

        attn = F.softmax(attn, -1)

        output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2)
        output = output.contiguous().view(batch_size, -1, self.d_model)
        output = self.W_out(output)
        return self.dropout(output)

    def rel_shift(self, emb):
        """
        Pad and shift form relative positional encodings.
        Taken from Transformer-XL implementation: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
        """
        batch_size, num_heads, seq_length1, seq_length2 = emb.size()
        zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_emb = torch.cat([zeros, emb], dim=-1)
        padded_emb = padded_emb.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        shifted_emb = padded_emb[:, :, 1:].view_as(emb)
        return shifted_emb

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        # self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttentionBlock(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, x , mask: Optional[Tensor] = None):


        x = self.layer_norm(x)
        outputs = self.attention(x, x, x, mask = mask)

        return self.dropout(outputs)


class TASA_attention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h  
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.transmit_module = nn.Conv2d(
            in_channels=h, 
            out_channels=h,
            kernel_size=(3, 3),
            padding=1
        )

        self.aggregate_module = nn.Conv2d(
            in_channels=h * 2,
            out_channels=h,
            kernel_size=(3, 3),
            padding=1
        )
        self._init_conv_weights()

    def _init_conv_weights(self):
        # Use small initial weights for stability
        nn.init.normal_(self.transmit_module.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.aggregate_module.weight, mean=0.0, std=0.01)

    def attention(self, query, key, value, mask, dropout, previous_attention_scores):
        M = torch.matmul(query, key.transpose(-2, -1))  # [B, H, T , d] @ [B, H, d, T] --> [B, H, T, T]

        if previous_attention_scores is not None:
            Mt = self.transmit_module(previous_attention_scores)  # CNNᵗ
            Ma_input = torch.cat((M, Mt), dim=1)  # [B, 2H, T, T]
            Ma = self.aggregate_module(Ma_input)  # CNNᵃ
        else:
            Ma = M  # No aggregation in the first layer

        # Normalize then apply mask
        A = Ma / math.sqrt(self.d_k)  
        

        # print("Attention shape:", A.shape)  # [B, H, T, T]

        if mask is not None:

            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            A = A.masked_fill(mask == 0, -1e9)

        A = A.softmax(dim=-1)  # [B, H, T, T]
        if dropout is not None:
            A = dropout(A)
        return A

    def forward(self, q, k, v, mask=None, previous_attention_scores=None):
        B, T, _ = q.size()

        query = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        A = self.attention(query, key, value, mask, self.dropout, previous_attention_scores)

        out = (A @ value).transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.w_o(out), A


class Self_Attention_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttentionBlock(d_model, h, p_dropout)
        self.feed_forward = FeedForwardBlock(d_model, ff_size,  p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.residual_connections = nn.ModuleList(
            ResidualConnection(d_model, p_dropout) for _ in range(2)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.residual_connections[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.residual_connections[1](x, lambda x : self.feed_forward(x))
        self.dropout(x)
        return x