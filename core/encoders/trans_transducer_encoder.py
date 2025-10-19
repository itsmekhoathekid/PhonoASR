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
        self.in_features = config['enc']['in_features']
        self.d_model = config['enc']['d_model']
        self.ff_size = config['enc']['ff_size']
        self.h = config['enc']['h']
        self.p_dropout = config['enc'].get('dropout', 0.1)
        self.n_layers = config['enc']['n_layer']

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