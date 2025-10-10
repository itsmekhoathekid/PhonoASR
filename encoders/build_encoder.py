from . import InterleaveHybridAcousticEncoder

def build_encoder(config, vocab_size):
    try: 
        n_head = config['enc']['n_head']
        d_model = config['enc']['d_model']
        d_hidden = config['enc']['d_hidden']
        dropout = config['enc']['dropout']
        n_layer = config['enc']['n_layer']

        return InterleaveHybridAcousticEncoder(n_head, d_model, d_hidden, vocab_size, dropout, n_layer)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")