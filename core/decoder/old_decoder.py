from core.modules import (
    FeedForwardBlock,
    ResidualConnection,
    ProjectionLayer,
    PositionalEncoding,
    FeedForwardModule, 
    ConvolutionalModule, 
    ResidualConnectionCM, 
    MultiConvolutionalGatingMLP,
    LayerNormalization, 
    MultiHeadedSelfAttentionModule,
    ScaledDotProductAttention
)
from core.modules import (
    MultiHeadAttentionBlock,
    ConvDec
)
import torch
from torch import nn
import random

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        self.ffn = FeedForwardBlock(d_model=d_model, d_ff=ff_size, dropout=dropout)
        self.self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.residual_connections =  nn.ModuleList([
            ResidualConnection(features=d_model, dropout=dropout),
            ResidualConnection(features=d_model, dropout=dropout),
            ResidualConnection(features=d_model, dropout=dropout)
        ])

    def forward(self, x, encoder_out, enc_mask, dec_mask):
        
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, dec_mask))

        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, enc_mask))
        
        x = self.residual_connections[2](x, lambda x: self.ffn(x))
        
        return x

class ConformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ff_size: int, dropout: float, kernel_size : int,  conv_config : None, conv_type : str = "default",) -> None:
        super().__init__()
        self.ffm1 = FeedForwardModule(d_model, ff_size, dropout, activation="gelu")
        self.attention = MultiHeadedSelfAttentionModule(d_model, h, dropout)
        
        if conv_type != "default":
            self.conv_module = MultiConvolutionalGatingMLP(
                conv_config["size"],
                conv_config["linear_units"],
                conv_config["fuse_type"],
                conv_config["kernel_sizes"],
                conv_config["merge_conv_kernel_size"],
                conv_config["use_non_linear"],
                dropout, 
                conv_config["use_linear_after_conv"],
                conv_config["activation"],
                conv_config["gate_activation"]
            )
        else:
            self.conv_module = ConvolutionalModule(d_model, kernel_size, dropout, ver = 'old', causal = True)
        self.ffm2 = FeedForwardModule(d_model, ff_size, dropout, activation="gelu")

        self.residual_connections = nn.ModuleList([
            ResidualConnectionCM(d_model, dropout) for _ in range(4)
        ])

        self.layer_norm = LayerNormalization(d_model)
    
    def forward(self, x, encoder_out, enc_mask, dec_mask):
        x = self.residual_connections[0](x, self.ffm1, 0.5)
        x = self.residual_connections[1](x, lambda x: self.attention(x,encoder_out,encoder_out, enc_mask), 1.0)
        x = self.residual_connections[2](x, lambda x : self.conv_module(x, dec_mask), 1.0)
        x = self.residual_connections[3](x, self.ffm2, 0.5)
        x = self.layer_norm(x)
        return x

class ConformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, ff_size: int, h: int, p_dropout: float, kernel_size : int, conv_config : None, conv_type : str = "default", k : int = 1) -> None:
        super().__init__()
        self.emb = EmbeddingModule(vocab_size=vocab_size, d_model=d_model, dropout=p_dropout, k = k, pos_enc=False)
        self.layers = nn.ModuleList(
            [ConformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout, kernel_size=kernel_size, conv_config=conv_config, conv_type=conv_type) for _ in range(n_layers)]
        )
        self.enc_linears = nn.ModuleList(
            [nn.Linear(in_features=d_model, out_features=d_model) for _ in range(k)]
        )
        self.heads = nn.ModuleList(
            [ConformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout, kernel_size=kernel_size, conv_config=conv_config, conv_type=conv_type) for _ in range(k)]
        )
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
        self.k = k
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input `x` through the decoder layers.

        Args:
            x (Tensor): The input tensor of shape [B, M]
            encoder_out (Tensor): The output from the encoder of shape [B, T, d_model]
            enc_mask (Tensor): The mask for the encoder output of shape [B, T]
            dec_mask (Tensor): The mask for the decoder input of shape [B, M]

        Returns:
            Tensor: The decoded output of shape [B, M, d_model].
        """
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, encoder_out, enc_mask, dec_mask)
        if self.k != 1:
            enc_outs = [linear(encoder_out) for linear in self.enc_linears]
            latent = [head(out, enc_out, enc_mask, dec_mask) for head, enc_out in zip(self.heads, enc_outs)]
        else:
            latent = [out]
        out = [self.projection(l) for l in latent]  
        return out

class LatentHead(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, dropout : float) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, latent_dim)
        self.residual = ResidualConnection(features=d_model, dropout = dropout)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x, lambda x : self.tanh(self.linear(x)))

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, k, conv_dec : bool = False, pos_enc : bool = True  ):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.role_emb = nn.Embedding(k, d_model)

        if k != 1:
            self.projection = nn.Linear(
                d_model * k,
                d_model
            )
        if pos_enc:
            self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.k = k
        if conv_dec:
            self.enc = ConvDec(
                num_blocks = 4, 
                in_channels= d_model ,  
                out_channels=[d_model, d_model, d_model, d_model],
                kernel_sizes=[3,3,3,3],
                dropout=dropout,
            )
    def forward(self, x):
        x = self.embedding(x) # (B, M, 3) -> (B, M, 3, d_model)
        if self.k != 1:
            # role_ids = torch.arange(self.k, device=x.device)  # [k]
            # role = self.role_emb(role_ids).view(1, 1, self.k, -1)  # [1,1,k,D]
            # x = x + role
            x = self.projection(x.view(x.size(0), x.size(1), -1))
        if hasattr(self, 'enc'):
            x = self.enc(x)
        if hasattr(self, 'pos_enc'):
            x = self.pos_enc(x)
        return self.dropout(x)
    
class Head(nn.Module):
    def __init__(self, d_model: int, ff_size: int, dropout: float, h : int) -> None:
        super().__init__()


        self.ffn = FeedForwardBlock(d_model=d_model, d_ff=ff_size, dropout=dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, dec_out) -> torch.Tensor:
        dec_out = dec_out + self.ffn(self.norm(dec_out))      
        return dec_out

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        self.ffn = FeedForwardBlock(d_model=d_model, d_ff=ff_size, dropout=dropout)
        # self.self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.residual_connections =  nn.ModuleList([
            ResidualConnection(features=d_model, dropout=dropout),
            ResidualConnection(features=d_model, dropout=dropout)
        ])

    def forward(self, x, encoder_out, enc_mask, dec_mask):
        
        
        x = self.residual_connections[0](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, enc_mask))
        
        x = self.residual_connections[1](x, lambda x: self.ffn(x))
        
        return x


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
        x = self.dropout(x)
        return x


import torch
import torch.nn as nn
import math
from typing import Optional

# -------------------------
# Small utilities
# -------------------------

class IntraSlotSelfAttention(nn.Module):
    """
    Self-attention on the slot dimension S (e.g., S=3), per time-step M and batch B.
    Input:  x  [B, M, S, d_model]
    Output: y  [B, M, S, d_model]
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, S, D = x.shape
        qkv = self.qkv(x)  # [B,M,S,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,M,S,H,dh] -> [B,M,H,S,dh]
        q = q.view(B, M, S, self.n_heads, self.d_head).transpose(2, 3)
        k = k.view(B, M, S, self.n_heads, self.d_head).transpose(2, 3)
        v = v.view(B, M, S, self.n_heads, self.d_head).transpose(2, 3)

        # attention logits: [B,M,H,S,S]
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if self.causal:
            # lower-triangular mask over slot dimension
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.drop(att)

        y = torch.matmul(att, v)  # [B,M,H,S,dh]
        y = y.transpose(2, 3).contiguous().view(B, M, S, D)  # [B,M,S,D]
        y = self.out(y)
        return y


class GatedFusion(nn.Module):
    """
    Gated fusion over slots.
    Input:  x [B, M, S, d_model]
    Output: z [B, M, d_model]
    """
    def __init__(self, d_model: int, n_slots: int, dropout: float = 0.1):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        # compute per-slot gate from concatenated slot features
        self.gate = nn.Linear(d_model * n_slots, n_slots * d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, S, D = x.shape
        assert S == self.n_slots

        flat = x.reshape(B, M, S * D)                 # [B,M,S*D]
        g = self.gate(flat).reshape(B, M, S, D)       # [B,M,S,D]
        g = torch.sigmoid(g)
        g = self.drop(g)
        z = torch.sum(g * x, dim=2)                   # [B,M,D]
        return z


# -------------------------
# Embedding variants
# -------------------------

class PhonemeTripleEmbedding_V1_Concat(nn.Module):
    """
    Variant A:
    token embedding + role(slot) embedding -> concat 3 slots -> linear projection -> (optional conv_dec) -> (optional pos_enc)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float,
        k: int = 3,
        pos_enc: bool = True,
        conv_dec: bool = False,
        PositionalEncoding=None,   # pass your class
        ConvDec=None               # pass your class
    ):
        super().__init__()
        assert k >= 1
        self.k = k
        self.embedding = nn.Embedding(vocab_size, d_model)

        # role/slot embedding: indices 0..k-1
        self.role_emb = nn.Embedding(k, d_model)

        # concat -> project
        self.projection = nn.Linear(d_model * k, d_model) if k != 1 else nn.Identity()

        self.dropout = nn.Dropout(dropout)

        if conv_dec:
            assert ConvDec is not None, "ConvDec class must be provided when conv_dec=True"
            self.enc = ConvDec(
                num_blocks=4,
                in_channels=d_model,
                out_channels=[d_model, d_model, d_model, d_model],
                kernel_sizes=[3, 3, 3, 3],
                dropout=dropout,
            )

        if pos_enc:
            assert PositionalEncoding is not None, "PositionalEncoding class must be provided when pos_enc=True"
            self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, M] if k==1, else [B, M, k]
        return: [B, M, d_model]
        """
        if self.k == 1:
            # standard embedding
            y = self.embedding(x)  # [B,M,D]
        else:
            # [B,M,k] -> [B,M,k,D]
            y = self.embedding(x)

            # add role embeddings
            role_ids = torch.arange(self.k, device=x.device)  # [k]
            role = self.role_emb(role_ids).view(1, 1, self.k, -1)  # [1,1,k,D]
            y = y + role

            # concat over slot dimension and project
            y = y.reshape(y.size(0), y.size(1), -1)  # [B,M,kD]
            y = self.projection(y)                   # [B,M,D]

        if hasattr(self, "enc"):
            y = self.enc(y)
        if hasattr(self, "pos_enc"):
            y = self.pos_enc(y)

        return self.dropout(y)


class PhonemeTripleEmbedding_V2_IntraAttn(nn.Module):
    """
    Variant B:
    token embedding + role embedding -> intra-slot self-attention (optional causal) -> merge -> (optional conv_dec) -> (optional pos_enc)

    merge can be:
      - "concat": concat slots then linear (like V1 but after attention)
      - "mean": average over slots
      - "first": take slot0 (not recommended unless slot0 is special)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float,
        k: int = 3,
        n_heads: int = 1,
        causal_slots: bool = False,
        merge: str = "concat",
        pos_enc: bool = True,
        conv_dec: bool = False,
        PositionalEncoding=None,
        ConvDec=None
    ):
        super().__init__()
        assert k >= 1
        assert merge in ("concat", "mean", "first")
        self.k = k
        self.merge = merge

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.role_emb = nn.Embedding(k, d_model)

        if k != 1:
            self.intra_attn = IntraSlotSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, causal=causal_slots)
        if merge == "concat" and k != 1:
            self.projection = nn.Linear(d_model * k, d_model)
        else:
            self.projection = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        if conv_dec:
            assert ConvDec is not None
            self.enc = ConvDec(
                num_blocks=4,
                in_channels=d_model,
                out_channels=[d_model, d_model, d_model, d_model],
                kernel_sizes=[3, 3, 3, 3],
                dropout=dropout,
            )

        if pos_enc:
            assert PositionalEncoding is not None
            self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k == 1:
            y = self.embedding(x)  # [B,M,D]
        else:
            y = self.embedding(x)  # [B,M,k,D]
            role_ids = torch.arange(self.k, device=x.device)
            role = self.role_emb(role_ids).view(1, 1, self.k, -1)
            y = y + role

            # intra-slot self-attention
            y = self.intra_attn(y)  # [B,M,k,D]

            # merge
            if self.merge == "concat":
                y = self.projection(y.reshape(y.size(0), y.size(1), -1))  # [B,M,D]
            elif self.merge == "mean":
                y = y.mean(dim=2)  # [B,M,D]
            else:  # "first"
                y = y[:, :, 0, :]  # [B,M,D]

        if hasattr(self, "enc"):
            y = self.enc(y)
        if hasattr(self, "pos_enc"):
            y = self.pos_enc(y)

        return self.dropout(y)


class PhonemeTripleEmbedding_V3_GatedFusion(nn.Module):
    """
    Variant C (requested):
    token embedding + role embedding -> gated fusion over slots -> (optional conv_dec) -> (optional pos_enc)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float,
        k: int = 3,
        pos_enc: bool = True,
        conv_dec: bool = False,
        PositionalEncoding=None,
        ConvDec=None
    ):
        super().__init__()
        assert k >= 1
        self.k = k

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.role_emb = nn.Embedding(k, d_model)

        if k != 1:
            self.fuser = GatedFusion(d_model=d_model, n_slots=k, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        if conv_dec:
            assert ConvDec is not None
            self.enc = ConvDec(
                num_blocks=4,
                in_channels=d_model,
                out_channels=[d_model, d_model, d_model, d_model],
                kernel_sizes=[3, 3, 3, 3],
                dropout=dropout,
            )

        if pos_enc:
            assert PositionalEncoding is not None
            self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k == 1:
            y = self.embedding(x)  # [B,M,D]
        else:
            y = self.embedding(x)  # [B,M,k,D]
            role_ids = torch.arange(self.k, device=x.device)
            role = self.role_emb(role_ids).view(1, 1, self.k, -1)
            y = y + role

            # gated fusion -> [B,M,D]
            y = self.fuser(y)

        if hasattr(self, "enc"):
            y = self.enc(y)
        if hasattr(self, "pos_enc"):
            y = self.pos_enc(y)

        return self.dropout(y)

class ConditionalProjectionAdapter(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.shared_out = nn.Linear(d_model, vocab_size)  # shared W,b

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(k)
        ])

    def forward(self, h_list):
        # h_list: list length k, each [B,M,D]
        logits = []
        for i, h in enumerate(h_list):
            h_i = self.adapters[i](h)          # [B,M,D]
            logits.append(self.shared_out(h_i)) # [B,M,V]
        return logits
    
class HeadOldVer(nn.Module):
    def __init__(self, d_model: int, ff_size: int, dropout: float, h : int) -> None:
        super().__init__()
        # self.self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.cross_atten_block = TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=dropout) 
        self.linear = nn.Linear(d_model, d_model)
        # self.residual = ResidualConnection(features=d_model, dropout=dropout)
        # self.n_layer = n_layer
        # self.linear = Self_Attention_Block(d_model=d_model, ff_size=ff_size, h=h, p_dropout=dropout)
    def forward(self, out, enc_out, enc_mask, dec_mask) -> torch.Tensor:
        enc_out = self.linear(enc_out)

        out =  self.cross_atten_block(out, enc_out, enc_mask, dec_mask)
        return out


    
    
class TransformerDecoderOlderVer(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, ff_size: int, h: int, p_dropout: float, k : int) -> None:
        super().__init__()
        # self.emb = PhonemeTripleEmbedding_V2_IntraAttn(
        #     vocab_size=vocab_size,
        #     d_model=d_model,
        #     dropout=p_dropout,
        #     k=k,
        #     n_heads=1,          # vì slot=3, head=1-3 đều OK
        #     causal_slots=False,  # ép slot0->slot1->slot2
        #     merge="concat",     # hoặc "mean"
        #     pos_enc=True,
        #     conv_dec=False,
        #     PositionalEncoding=PositionalEncoding,
        #     ConvDec=ConvDec,
        # )
        self.emb = EmbeddingModule(vocab_size=vocab_size, d_model=d_model, dropout=p_dropout, k = k, conv_dec=False)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout) for _ in range(n_layers)]
        )

        self.heads = nn.ModuleList(
            [HeadOldVer(d_model=d_model, ff_size=ff_size, dropout=p_dropout, h= h) for _ in range(k)] # ver 2
        )
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
        self.k = k
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input `x` through the decoder layers.

        Args:
            x (Tensor): The input tensor of shape [B, M]
            encoder_out (Tensor): The output from the encoder of shape [B, T, d_model]
            enc_mask (Tensor): The mask for the encoder output of shape [B, T]
            dec_mask (Tensor): The mask for the decoder input of shape [B, M]

        Returns:
            Tensor: The decoded output of shape [B, M, d_model].
        """
        out= self.emb(x)
        for layer in self.layers:
            out = layer(out, encoder_out, enc_mask, dec_mask)
        if self.k != 1:
            latent = [head(out) for head in self.heads]

        out = [self.projection(l) for l in latent]
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, ff_size: int, h: int, p_dropout: float, k : int) -> None:
        super().__init__()
        self.emb = EmbeddingModule(vocab_size=vocab_size, d_model=d_model, dropout=p_dropout, k = k, conv_dec=False)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout) for _ in range(n_layers)]
        )
        self.heads = nn.ModuleList(
            [Head(d_model=d_model, ff_size=ff_size, dropout=p_dropout, h= h) for _ in range(k)]
        )
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
        self.k = k
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input `x` through the decoder layers.

        Args:
            x (Tensor): The input tensor of shape [B, M]
            encoder_out (Tensor): The output from the encoder of shape [B, T, d_model]
            enc_mask (Tensor): The mask for the encoder output of shape [B, T]
            dec_mask (Tensor): The mask for the decoder input of shape [B, M]

        Returns:
            Tensor: The decoded output of shape [B, M, d_model].
        """
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, encoder_out, enc_mask, dec_mask)
        if self.k != 1:
            latent = [head(out) for head in self.heads]
        else:
            latent = [out]
        out = [self.projection(l) for l in latent]  
        return out

class BaseDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs)
        
        batch_size = inputs.size(0)
        max_len = inputs.size(1)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            
            
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, torch.tensor(sorted_seq_lengths, dtype=torch.int64).cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)    
        
        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        padded_output = torch.zeros(batch_size, max_len, outputs.size(2))
                
        if inputs.is_cuda: padded_output = padded_output.cuda()

        max_output_size = outputs.size(1)
        padded_output[:, :max_output_size, :] = outputs 
        
        #outputs = self.output_proj(outputs)
        outputs = self.output_proj(padded_output)
        
        return outputs, hidden

class SaaDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, embed_dropout=0.1, var_dropout=0.2):
        super(SaaDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.ModuleList([
            nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
            if i == 0 else 
            nn.LSTMCell(hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.attention = ScaledDotProductAttention(temperature=hidden_size**0.5)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.var_dropout = nn.Dropout(var_dropout)

    def forward(self, decoder_input, encoder_outputs, encoder_mask=None, decoder_mask=None, tfr=0.0):
        """
        Args:
            decoder_input: [batch, max_len] (bắt đầu bằng SOS)
            encoder_outputs: [batch, time, hidden]
            encoder_mask: [batch, time] (mask cho encoder)
            tfr: float (0.0 = no teacher forcing, 1.0 = full teacher forcing)
        """
        max_len = decoder_input.size(1)
        batch_size = decoder_input.size(0)

        h = [torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device) 
             for _ in range(self.num_layers)]
        context = torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device)

        outputs = []
        
        # Khởi tạo với SOS token
        current_input = decoder_input[:, 0]  # [batch] - SOS tokens
        
        for t in range(max_len):
            # Embed current input
            embedded = self.embedding(current_input)  # [batch, embed_dim]
            embedded = self.embed_dropout(embedded)
            
            # RNN forward
            rnn_input = torch.cat([embedded, context], dim=1)
            h[0], c[0] = self.rnn[0](rnn_input, (h[0], c[0]))
            h[0] = self.var_dropout(h[0]) 
            
            for i in range(1, self.num_layers):
                new_h, new_c = self.rnn[i](h[i-1], (h[i], c[i]))
                h[i] = self.var_dropout(new_h)
                c[i] = new_c
                
            # Attention
            query = h[-1].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, hidden]
            key = value = encoder_outputs.unsqueeze(1)  # [batch, 1, time, hidden]
            if encoder_mask is not None:
                attn_mask = encoder_mask.unsqueeze(1)  # [B, 1, time]
            else:
                attn_mask = None
                
            context, attn = self.attention(query, key, value, mask=attn_mask)
            context = context.squeeze(1).squeeze(1)  # [B, hidden]
            
            # Output projection
            char_input = torch.cat([h[-1], context], dim=1)
            output = self.mlp(char_input)
            outputs.append(output)
            
            # Decide next input token (except for last timestep)
            if t < max_len - 1:
                if random.random() < tfr:
                    # Teacher forcing: use ground truth
                    current_input = decoder_input[:, t + 1]
                else:
                    # No teacher forcing: use predicted token
                    predicted_id = output.argmax(dim=-1)
                    current_input = predicted_id

        logits = torch.stack(outputs, dim=1)  # [batch, max_len, vocab_size]
        logits = [logits]
        return logits  # [B, max_len, vocab]

class VGGTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, ff_size: int, h: int, p_dropout: float) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.enc = ConvDec(
            num_blocks = 4, 
            in_channels= d_model ,
            out_channels=[d_model, d_model, d_model, d_model],
            kernel_sizes=[3,3,3,3],
            dropout=p_dropout,
        )
        self.pe = PositionalEncoding(d_model=d_model) 
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout) for _ in range(n_layers)]
        )
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input `x` through the decoder layers.

        Args:
            x (Tensor): The input tensor of shape [B, M]
            encoder_out (Tensor): The output from the encoder of shape [B, T, d_model]
            enc_mask (Tensor): The mask for the encoder output of shape [B, T]
            dec_mask (Tensor): The mask for the decoder input of shape [B, M]

        Returns:
            Tensor: The decoded output of shape [B, M, d_model].
        """
        out = self.emb(x)
        # # out = x.unsqueeze(1)
        # # out = out.transpose(1,2)
        out = self.enc(out)
        out = self.pe(out)
        for layer in self.layers:
            out = layer(out, encoder_out, enc_mask, dec_mask)
        out = self.projection(out)
        out = [out]
        return out