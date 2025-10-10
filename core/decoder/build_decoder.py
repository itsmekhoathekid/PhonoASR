from . import TransformerDecoder

def build_decoder(config, vocab_size):
    try:
        n_head = config['dec']['n_head']
        d_model = config['dec']['d_model']
        d_hidden = config['dec']['d_hidden']
        dropout = config['dec']['dropout']
        n_layer = config['dec']['n_layer']
        max_len = config['dec']['max_len']
        k = config['dec']['k']
        
        return TransformerDecoder(vocab_size, n_layer, d_model, d_hidden, n_head, dropout, k)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")