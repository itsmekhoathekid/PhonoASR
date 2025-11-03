from core.modules import (
    ResidualForBase, ResidualConnection
)
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCNNEncoder(nn.Module):
    def __init__(self, kernel_size=5, stride=1, feature_dim= 80, dim_out= 160):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 100, kernel_size= kernel_size, stride= stride, padding=0)  # causal pad time
        self.conv2 = nn.Conv2d(100, 100, kernel_size= kernel_size, stride= stride, padding=0)
        self.conv3 = nn.Conv2d(100, 64, kernel_size= kernel_size, stride= stride, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size= kernel_size, stride= stride, padding=0)
        
        self.fc = nn.Linear(64 * feature_dim, dim_out)

    def forward(self, x):  # x: [B, 1, T, F]
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))  # (left, right, top, bottom)
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv2(x))
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv3(x))
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv4(x))
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # [B, T, 64*F]
        
        x = self.fc(x)  # [B, T, dim_out]
        return x  # [B, T, dim_out]

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.se(x)  
        return x * se  



class GlobalCNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size_pw, kernel_size_dw, dilation, n_dropout=0.0):
        super(GlobalCNNBlock, self).__init__()
        # Point-wise CNN 1
        self.pw_cnn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size_pw)

        # Dilated Depth-wise CNN
        self.dw_cnn = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size_dw,
            dilation=dilation,
            groups= hidden_dim,
            padding= dilation * (kernel_size_dw - 1) // 2  # this auto keeps T
        )

        # Point-wise CNN 2
        self.pw_cnn2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=kernel_size_pw)

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(input_dim, 8)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(input_dim)

        # ReLU
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(n_dropout)

        self.kernel_size_dw = kernel_size_dw
        self.dilation = dilation

        # self.residual_base = ResidualForBase(input_dim, n_dropout)

    def forward(self, x):
        # x: [B, T, D] -> Chuyển thành [B, D, T] cho Conv1d
        residual = x 
        x = x.transpose(1, 2)  # [B, D, T]
        

        # print(f"x.shape: {x.shape}")
        # Point-wise CNN 1
        x = self.pw_cnn1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # print(f"x.shape: {x.shape}")
        # Dilated Depth-wise CNN (padding thủ công để giữ nguyên chiều dài)
        # pad = (self.kernel_size_dw - 1) * self.dilation  # Padding bên trái để đảm bảo nhân quả
        # x = F.pad(x, (pad, 0))  # Chỉ pad bên trái
        x = self.dw_cnn(x)
        x = self.relu(x)
        x = self.bn2(x)
        # print(f"x.shape: {x.shape}")

        # Point-wise CNN 2
        x = self.pw_cnn2(x)
        x = self.bn3(x)

        # print(f"x.shape: {x.shape}")
        # Squeeze and Excitation
        x = self.se(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)
        x += residual

        return x
    
class GlobalCNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size_pw, kernel_size_dw, n_layers=6, n_dropout=0.0):
        super(GlobalCNNEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GlobalCNNBlock(input_dim, hidden_dim, kernel_size_pw, kernel_size_dw, dilation= 2**i, n_dropout= n_dropout) 
            for i in range(0, n_layers)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        self.local_cnn = LocalCNNEncoder(kernel_size=config["local_cnn_encoder"]["kernel_size"], 
                                         stride=config["local_cnn_encoder"]["stride"], 
                                         feature_dim= config["local_cnn_encoder"]["feature_dim"], 
                                         dim_out= config["local_cnn_encoder"]["dim_out"]
                                         )
        self.global_cnn = GlobalCNNEncoder(
            input_dim= config["global_cnn_encoder"]["input_dim"], 
            hidden_dim= config["global_cnn_encoder"]["hidden_dim"], 
            kernel_size_pw= config["global_cnn_encoder"]["kernel_size_pw"], 
            kernel_size_dw= config["global_cnn_encoder"]["kernel_size_dw"], 
            n_layers= config["global_cnn_encoder"]["n_layers"], 
            n_dropout= config["global_cnn_encoder"]["n_dropout"]
        )
        self.projected = nn.Linear(config["global_cnn_encoder"]["input_dim"] * 2, 
                                   config['cnn_enc_out'])  

    def forward(self, x):
        x = x.unsqueeze(1)
        local_out = self.local_cnn(x) 

        global_out = self.global_cnn(local_out)  
        global_out +=  local_out

        concat = torch.cat([local_out, global_out], dim=2) 
        
        output = self.projected(concat) 
        return output

class BaseLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=False):
        super(BaseLSTMLayer, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        self.output_proj = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size,
            output_size, bias=True
        )

        self.swish = Swish()  # Swish activation function

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3  # [B, T, F]

        # print("inputs", inputs.shape)
        B, T, F = inputs.shape
        # inputs = self.input_bn(inputs.view(-1, F)).view(B, T, F)

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs_sorted = inputs[indices]


            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                inputs_sorted, sorted_seq_lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        else:
            packed_inputs = inputs

        self.lstm.flatten_parameters()

        # print("packed_inputs", packed_inputs.data.shape)

        outputs, hidden = self.lstm(packed_inputs)
        

        if input_lengths is not None:
            unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            

            _, desorted_indices = torch.sort(indices)
            outputs = unpacked_outputs[desorted_indices]

        else:
            outputs = outputs


        logits = self.output_proj(outputs)
        logits = self.swish(logits)  # Apply Swish activation

        return logits, hidden

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ProjectedLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = input_size if i == 0 else output_size
            layer = BaseLSTMLayer(
                input_size=input_dim,
                hidden_size=hidden_size,
                output_size=output_size,
                n_layers=1,  # Each layer is a single LSTM layer
                dropout=dropout,
                bidirectional=bidirectional
            )
            self.layers.append(layer)

        

    def forward(self, x, lengths=None):
        for i in range(self.n_layers):
            x, hidden = self.layers[i](x, lengths)
        return x, hidden


class ConvRNNTEncoder(nn.Module):
    def __init__(self, config):
        super(ConvRNNTEncoder, self).__init__()

        self.cnn_encoder = CNNEncoder(config)
            
        self.lstm = nn.LSTM(
            input_size=config['cnn_enc_out'],
            hidden_size=config['hidden_size'],
            num_layers=config['n_layers'],
            batch_first=True,
            dropout= config['dropout'] if config['n_layers'] > 1 else 0.0,
            bidirectional=config['bidirectional']
        )
        
        self.output_proj = nn.Linear(
            2 * config['hidden_size'] if config['bidirectional'] else config['hidden_size'],
            config['output_size'], bias=True
        )

        self.input_bn = nn.BatchNorm1d(config['cnn_enc_out'])

    def forward(self, inputs, mask):
        assert inputs.dim() == 3  # [B, T, F]
        input_lengths = mask.sum(-1) # [B]
        # Sử dụng CNN encoder trước
        inputs = self.cnn_encoder(inputs)

        B, T, F = inputs.shape
        inputs = self.input_bn(inputs.view(-1, F)).view(B, T, F)

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs_sorted = inputs[indices]

            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                inputs_sorted, sorted_seq_lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        else:
            packed_inputs = inputs

        self.lstm.flatten_parameters()
        
        outputs, _ = self.lstm(packed_inputs)

        if input_lengths is not None:
            unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            _, desorted_indices = torch.sort(indices)
            outputs = unpacked_outputs[desorted_indices]

        else:
            outputs = outputs

        logits = self.output_proj(outputs)

        return logits, mask, input_lengths