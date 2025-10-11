from core.encoders import build_encoder
from core.decoder import build_decoder
import torch
import torch.nn as nn

class AcousticModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(AcousticModel, self).__init__()
        self.encoder = build_encoder(config, vocab_size)
        self.decoder = build_decoder(config, vocab_size)

        self.sos_id = config['sos_id']
        self.eos_id = config['eos_id']
        self.blank_id = config['blank_id']

    def forward(self, inputs, decoder_input, encoder_mask=None, tfr=1.0):
        encoder_outputs = self.encoder(inputs, encoder_mask)
        decoder_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask, tfr)  

        return encoder_outputs, decoder_outputs
    
    def recognize(self, enc_inputs, enc_mask=None):
        """
        Greedy decode for inference
        Args:
            enc_inputs: [1, time, feature] - batch_size = 1
            speech_length: [1] - lengths of input sequences
            target_length: int - max target length
            enc_mask: [1, time] - mask for encoder inputs
        Returns:
            list of lists: token IDs for each batch item
        """
        encoder_outputs, _ = self.encoder(enc_inputs, enc_mask)
        device = enc_inputs.device
        
        # Khởi tạo decoder input với SOS token
        decoder_input = torch.tensor([[self.sos_id]], device=device)  # [1, 1]
        token_list = []
        
        for step in range(500):
            # Gọi decoder với tfr=0.0 (no teacher forcing)
            with torch.no_grad():
                logits = self.decoder(decoder_input, encoder_outputs, enc_mask, tfr=0.0)
            
            predicted_token = logits[:, -1, :].argmax(dim=-1).item()  
            token_list.append(predicted_token)

            if predicted_token == self.eos_id:
                break
            
            new_token = torch.tensor([[predicted_token]], device=device)  # [1, 1]
            decoder_input = torch.cat([decoder_input, new_token], dim=1)  # [1, step+2]
        
        return [token_list]