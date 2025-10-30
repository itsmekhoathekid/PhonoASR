from core.encoders import build_encoder
from core.decoder import build_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class AcousticModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(AcousticModel, self).__init__()
        self.encoder = build_encoder(config['model']['enc'], vocab_size)
        self.decoder = build_decoder(config['model'], vocab_size)
        self.config = config
        if self.config['training']['type_training'] == 'ctc-kldiv':
            self.ctc_lin = nn.Linear(config['model']['enc']['d_model'], vocab_size)
        
    def forward(self, inputs, decoder_input, encoder_mask=None, decoder_mask=None):
        encoder_outputs, encoder_mask, encoder_lengths = self.encoder(inputs, encoder_mask)
        decoder_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask, decoder_mask)
        if self.config['training']['type_training'] == 'ctc-kldiv':
            encoder_outputs = self.ctc_lin(encoder_outputs)  # [B, T, vocab_size]
            encoder_outputs = encoder_outputs.log_softmax(dim=-1)
        return encoder_outputs, decoder_outputs, encoder_lengths
    
    def encode(self, src, src_mask=None):
        """
        Encode input sequences
        Args:
            src: [B, time, feature]
            src_mask: [B, time]
        Returns:
            enc_outputs: [B, time, feature]
        """
        enc_outputs = self.encoder(src, src_mask)
        return enc_outputs
    
    def decode(self, decoder_input, encoder_outputs, encoder_mask=None, decoder_mask=None):
        """
        Decode sequences
        Args:
            decoder_input: [B, M]
            encoder_outputs: [B, T, feature]
            encoder_mask: [B, T]
            decoder_mask: [B, M]
        Returns:
            dec_outputs: [B, M, vocab_size]
        """
        dec_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask, decoder_mask)

        return dec_outputs

    def verify(self, tgt, enc_out, src_mask, tgt_mask):
        """
        Verify the target sequence.
        Args:
            tgt (Tensor): Target sequence tensor of shape (B, U).
            enc_out (Tensor): Encoded output from the encoder of shape (B, T, d_model).
            src_mask (Tensor): Mask for the input sequence of shape (B, T).
            tgt_mask (Tensor): Mask for the target sequence of shape (B, U).
        Returns:
            Tensor: Verified output of shape (B, U, vocab_size).
        """
        dec_out = self.decoder.verify(tgt, enc_out, src_mask, tgt_mask)
        return dec_out

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()
        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs

class TransducerAcousticModle(nn.Module):
    def __init__(self, config, vocab_size):
        super(TransducerAcousticModle, self).__init__()
        self.encoder = build_encoder(config['model']['enc'], vocab_size)
        self.lin_enc = nn.Linear(config['model']['enc']['d_model'], config['model']['enc']['output_dim'])
        self.decoder = build_decoder(config['model'], vocab_size)

        self.joint = JointNet(
            input_size=config['model']['joint']['input_size'],
            inner_dim=config['model']['joint']['inner_size'],
            vocab_size=vocab_size,
        )
        self.pad_id = config['training'].get('pad_id', 0)

    def forward(self, inputs, targets, inputs_length,  decoder_mask):
        enc_state, _, fbank_len = self.encoder(inputs, inputs_length)
        enc_state = self.lin_enc(enc_state)

        true_lengths = (targets != self.pad_id).sum(dim=1).cpu()  # shape [B]

        dec_state, _ = self.decoder(targets, true_lengths)
        joint_outputs = self.joint(enc_state, dec_state)
        return joint_outputs, dec_state, fbank_len
    
    def recognize(self, inputs, inputs_length):
        batch_size = inputs.size(0)

        enc_states, inputs_length = self.encoder(inputs, inputs_length)
        zero_token = torch.LongTensor([[self.sos]]) 
        
        if inputs.is_cuda:
            zero_token = zero_token.cuda()
        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)
            for t in range(lengths):
                enc_step = enc_states[:, t, :]
                dec_proj = dec_state[:, -1, :]
                logits = self.joint(enc_step, dec_proj)
                logits = F.softmax(logits.squeeze(1).squeeze(1), dim=-1) 
                pred = torch.argmax(logits, dim=-1).item()

                if pred == self.eos: # eos
                    break

                if pred not in [self.eos, self.blank, self.sos]:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = [decode(enc_states[i], inputs_length[i]) for i in range(batch_size)]
        return results