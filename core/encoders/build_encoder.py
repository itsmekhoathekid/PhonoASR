from . import * 

def build_encoder(config, vocab_size):
    try:
        if config["name"] == 'SAA':
            return InterleaveHybridAcousticEncoder(config, vocab_size)
        elif config["name"] == 'Conformer' or config["name"] == 'ConvConformer':
            return ConformerEncoder(config)
        elif config['name'] == 'SpeechTransformer':
            return TransformerEncoder(config)
        elif config['name'] == 'TASA':
            return TASA_encoder(config)
        elif config['name'] == 'VGGTransformer':
            return VGGTransformerEncoder(config)
        elif config['name'] == 'ConvRNNT':
            return ConvRNNTEncoder(config)
        elif config['name'] == 'RNNT':
            return RNNTEncoder(config)
        elif config['name'] == 'TransformerTransducer':
            return TransformerTransducerEncoder(config)
        elif config['name'] == 'SBConformer':
            return ModifiedSBConformerEncoder(
                num_layers=config["num_encoder_layers"],
                d_model=config["encoder_dim"],
                nhead=config["num_attention_heads"],
                d_ffn = config["encoder_dim"] * config["feed_forward_expansion_factor"],
                dropout = config["dropout_rate"],
                kernel_size=config["conv_kernel_size"]
            )
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")