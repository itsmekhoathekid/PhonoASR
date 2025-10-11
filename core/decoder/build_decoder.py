from . import TransformerDecoder

def build_decoder(config, vocab_size):
    try:
        vocab_size = vocab_size
        n_layer = config['dec']['n_layer']
        d_model = config['dec']['d_model']
        d_hidden = config['dec']['ff_size']
        n_head = config['dec']['n_head']
        dropout = config['dec']['dropout']
        k = config['k']

        return TransformerDecoder(vocab_size, n_layer, d_model, d_hidden, n_head, dropout, k)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")