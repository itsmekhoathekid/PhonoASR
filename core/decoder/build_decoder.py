from . import TransformerDecoder, ConformerDecoder

def build_decoder(config, vocab_size):
    try:
        if config['enc']['name'] == 'conformer':
            
            return ConformerDecoder(
                vocab_size=vocab_size,
                n_layers=config['dec']['n_layer'],
                d_model=config['dec']['d_model'],
                ff_size=config['dec']['ff_size'],
                h=config['dec']['n_head'],
                p_dropout=config['dec']['dropout'],
                kernel_size=config['dec']['conv_kernel_size'],
                conv_config=config['dec'].get('conv_config', None),
                conv_type=config['dec'].get('type', 'default'),
                k = config['dec']['k']
            )
        else:
            vocab_size = vocab_size
            n_layer = config['dec']['n_layer']
            d_model = config['dec']['d_model']
            d_hidden = config['dec']['ff_size']
            n_head = config['dec']['n_head']
            dropout = config['dec']['dropout']
            k = config['dec']['k']

            return TransformerDecoder(vocab_size, n_layer, d_model, d_hidden, n_head, dropout, k)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")