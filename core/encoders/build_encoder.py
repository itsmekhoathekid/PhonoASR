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
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")