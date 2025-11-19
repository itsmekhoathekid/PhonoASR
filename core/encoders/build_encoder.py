from . import * 

def build_encoder(config, vocab_size):
    try:
        enc_type = config["type"]
        if enc_type == 'SAA':
            return InterleaveHybridAcousticEncoder(config, vocab_size)
        elif enc_type == 'Conformer' or enc_type == 'ConvConformer':
            return ConformerEncoder(config)
        elif enc_type == 'SpeechTransformer':
            return TransformerEncoder(config)
        elif enc_type == 'TASA':
            return TASA_encoder(config)
        elif enc_type == 'VGGTransformer':
            return VGGTransformerEncoder(config)
        elif enc_type == 'ConvRNNT':
            return ConvRNNTEncoder(config)
        elif enc_type == 'RNNT':
            return RNNTEncoder(config)
        elif enc_type == 'TransformerTransducer':
            return TransformerTransducerEncoder(config)
        elif enc_type == 'SBConformer':
            return ModifiedSBConformerEncoder(
                num_layers=config["num_encoder_layers"],
                d_model=config["encoder_dim"],
                nhead=config["num_attention_heads"],
                d_ffn = config["encoder_dim"] * config["feed_forward_expansion_factor"],
                dropout = config["dropout_rate"],
                kernel_size=config["conv_kernel_size"]
            )
        elif enc_type == 'Zipformer':
            return ZipformerEncoder(config)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")