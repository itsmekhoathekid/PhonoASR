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
    MultiHeadedSelfAttentionModule
)
from core.modules import (
    MultiHeadAttentionBlock,
    ConvDec
)
import torch
from torch import nn

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

class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, k, conv_dec : bool = False, pos_enc : bool = True  ):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        if k != 1:
            self.projection = nn.Linear(d_model * k, d_model)
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
            x = self.projection(x.view(x.size(0), x.size(1), -1))
        if hasattr(self, 'enc'):
            x = self.enc(x)
        if hasattr(self, 'pos_enc'):
            x = self.pos_enc(x)
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, ff_size: int, h: int, p_dropout: float, k : int) -> None:
        super().__init__()
        self.emb = EmbeddingModule(vocab_size=vocab_size, d_model=d_model, dropout=p_dropout, k = k)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout) for _ in range(n_layers)]
        )
        self.enc_linears = nn.ModuleList(
            [nn.Linear(in_features=d_model, out_features=d_model) for _ in range(k)]
        )
        self.heads = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout) for _ in range(k)]
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
