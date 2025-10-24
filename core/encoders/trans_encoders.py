import torch
from torch import nn
from core.modules import *
from core.modules import TASA_attention, MultiHeadAttentionBlock

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

def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)

class TASA_layers(nn.Module):
    def __init__(self, d_model, d_ff, h, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.p_dropout = p_dropout

        self.attention = TASA_attention(
            d_model=d_model,
            h=h,
            dropout=p_dropout
        )

        self.ffn = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=p_dropout
        )

        self.residual_attn = ResidualForTASA(features=d_model, dropout=p_dropout)
        self.residual_ffn = ResidualConnection(features=d_model, dropout=p_dropout)
    def forward(self, x, mask=None, previous_attention_scores=None):
        """
        x: [batch, time, features]
        mask: [batch, time]
        previous_attention_scores: [batch, h, time, time]
        """

        residual, atten_score = self.attention(x, x, x, mask, previous_attention_scores)
        x = self.residual_attn(x, residual)  # Residual connection cho attention
        
        # Residual connection cho FFN
        x = self.residual_ffn(x, self.ffn)
        
        return x, atten_score  # Giữ nguyên



class TASA_encoder(nn.Module):
    def __init__(self, enc_config):
        super().__init__()

        self.num_blocks = 2
        self.num_layers_per_block = 1

        self.frontend = ConvolutionFrontEnd(
            in_channels=1,
            num_blocks=enc_config['subsampling']['num_blocks'],
            num_layers_per_block=enc_config['subsampling']['num_layers_per_block'],
            out_channels=enc_config['subsampling']['out_channels'],
            kernel_sizes=enc_config['subsampling']['kernel_sizes'],
            strides=enc_config['subsampling']['strides'],
            residuals=enc_config['subsampling']['residuals'],
            activation=nn.ReLU,        
            norm=nn.BatchNorm2d,            
            dropout=enc_config['subsampling']['dropout'],
        )

        self.layers = nn.ModuleList(
            [
                TASA_layers(
                    d_model=enc_config['d_model'],
                    d_ff=enc_config['ff_size'],
                    h=  enc_config['h'],
                    p_dropout=enc_config['p_dropout'],
                ) for _ in range(enc_config['n_layers'])
            ]
        )

        self.projection = nn.Linear(enc_config['in_features'], enc_config['d_model'])
        self.pe = PositionalEncoding(enc_config['d_model'])

    
    def forward(self, x, mask=None, previous_attention_scores=None):
        x = x.unsqueeze(1)  # [batch, channels, time, features
        x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]
        
        
        
        x = self.projection(x)  # [batch, time, d_model]
        x = self.pe(x)  # [batch, time, d_model]

        for layer in self.layers:
            
            x, previous_attention_scores = layer(x, mask, previous_attention_scores)

        input_lens = torch.sum(mask, dim=1) # [B]
        return x , mask, input_lens

class TransformerEncoderLayer(nn.Module):
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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_config):
        super().__init__()
        self.linear = nn.Linear(in_features=enc_config['in_features'], out_features=enc_config['d_model'])
        self.pe = PositionalEncoding(enc_config['d_model'])
        # self.pe = PositionalEmbedding(d_model)
        

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=enc_config['d_model'],
                    ff_size=enc_config['ff_size'],
                    h=enc_config['h'],
                    p_dropout=enc_config['p_dropout'],
                )
                for _ in range( enc_config['n_layers'] )
            ]
        )

        self.frontend = ConvolutionFrontEnd(
            in_channels=1,
            num_blocks=enc_config['subsampling']['num_blocks'],
            num_layers_per_block=enc_config['subsampling']['num_layers_per_block'],
            out_channels=enc_config['subsampling']['out_channels'],
            kernel_sizes=enc_config['subsampling']['kernel_sizes'],
            strides=enc_config['subsampling']['strides'],
            residuals=enc_config['subsampling']['residuals'],
            activation=nn.ReLU,        
            norm=nn.BatchNorm2d,            
            dropout=enc_config['subsampling']['dropout'],
        )


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]


        out = self.linear(x)
        out = self.pe(out)
        mask_atten = mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            out = layer(out, mask_atten)
        
        enc_input_lengths = torch.sum(mask, dim=1) # [B]
        return out, mask, enc_input_lengths


class VGGTransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_config
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=enc_config['in_features'], out_features=enc_config['d_model'])
        self.pe = PositionalEncoding(enc_config['d_model'])
        

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=enc_config['d_model'],
                    ff_size=enc_config['ff_size'],
                    h=enc_config['h'],
                    p_dropout=enc_config['p_dropout']
                )
                for _ in range(enc_config['n_layers'])
            ]
        )

        self.frontend = VGGFrontEnd(
            num_blocks = enc_config['subsampling']['num_blocks'],
            in_channel=1,
            out_channels=enc_config['subsampling']['out_channels'],
            conv_kernel_sizes=enc_config['subsampling']['kernel_sizes'],
            pooling_kernel_sizes=enc_config['subsampling']['pooling_kernel_sizes'],
            num_conv_layers=enc_config['subsampling']['num_conv_layers'],
            layer_norms=enc_config['subsampling']['layer_norms'],
            input_dim= enc_config['input_dim']
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        # print("x shape before frontend:", x.shape)  # [batch, 1, time, features]
        x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        # print("x shape after frontend:", x.shape)
        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]
        lengths = torch.sum(mask, dim=1)  # [B]

        out = self.linear(x)
        out = self.pe(out)
        mask_atten = mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            out = layer(out, mask_atten)
        
        return out, mask, lengths

