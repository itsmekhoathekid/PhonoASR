import torch
import torch.nn as nn


import torch
import torch.nn as nn


class RNNTEncoder(nn.Module):
    def __init__(self, config):
        super(RNNTEncoder, self).__init__()
        input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=False
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.n_layers = config["n_layers"]
        self.dropout = config.get("dropout", 0.2)
        self.bidirectional = config.get("bidirectional", False)

        self.lstm = nn.LSTM(
            input_size= self.input_size,
            hidden_size= self.hidden_size,
            num_layers= self.n_layers,
            batch_first= True,
            dropout= self.dropout if self.n_layers > 1 else 0.0,
            bidirectional= self.bidirectional
        )
        
        self.output_proj = nn.Linear(
            2 * self.hidden_size if bidirectional else self.hidden_size,
            self.output_size, bias=True
        )

        self.input_bn = nn.BatchNorm1d(self.input_size)

    def forward(self, inputs, mask):
        assert inputs.dim() == 3  # [B, T, F]
        input_lengths = mask.sum(-1)
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
        

        outputs, hidden = self.lstm(packed_inputs)


        if input_lengths is not None:
            unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)


            _, desorted_indices = torch.sort(indices)
            outputs = unpacked_outputs[desorted_indices]

        else:
            outputs = outputs


        logits = self.output_proj(outputs)


        return logits, mask
