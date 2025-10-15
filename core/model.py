from core.encoders import build_encoder
from core.decoder import build_decoder
import torch
import torch.nn as nn

class AcousticModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(AcousticModel, self).__init__()
        self.encoder = build_encoder(config, vocab_size)
        self.decoder = build_decoder(config, vocab_size)


    def forward(self, inputs, decoder_input, encoder_mask=None, decoder_mask=None):
        encoder_outputs, encoder_mask = self.encoder(inputs, encoder_mask)

        decoder_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask, decoder_mask)

        return encoder_outputs, decoder_outputs
    
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