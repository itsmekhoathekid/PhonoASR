from . import *

def build_encoder(config, vocab_size):
    try:
        if config["name"] == 'SAA':
            n_head = config['enc']['n_head']
            d_model = config['enc']['d_model']
            d_hidden = config['enc']['d_hidden']
            dropout = config['enc']['dropout']
            n_layer = config['enc']['n_layer']

            return InterleaveHybridAcousticEncoder(n_head, d_model, d_hidden, vocab_size, dropout, n_layer)
        elif config["name"] == 'Conformer' or config["name"] == 'ConvConformer':
            return ConformerEncoder(config['enc'])
        elif config['name'] == 'SpeechTransformer':
            return TransformerEncoder(config['enc'])
        elif config['name'] == 'TASA':
            return TASA_encoder(config['enc'])
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")