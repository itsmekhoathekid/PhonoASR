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