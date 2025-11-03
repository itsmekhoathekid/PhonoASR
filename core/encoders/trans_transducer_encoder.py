from core.modules import (
    Self_Attention_Block, calc_data_len, 
    get_mask_from_lens, PositionalEncoding, 
    ConvolutionFrontEnd
)
import torch
import torch.nn as nn
import math

class TransformerTransducerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_features = config['in_features']
        d_model = config['d_model']
        ff_size = config['ff_size']
        h = config['h']
        p_dropout = config.get('dropout', 0.1)
        n_layers = config['n_layer']

        self.linear = nn.Linear(in_features=in_features, out_features=d_model)
        self.pe = PositionalEncoding(d_model)
        

        self.layers = nn.ModuleList(
            [
                Self_Attention_Block(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    p_dropout=p_dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        lengths = torch.sum(mask, dim=1)
        out = self.linear(x)  # (batch, seq_len, d_model)
        out = self.pe(out)
        mask = mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            out = layer(out, mask)
        
        
        return out, mask, lengths