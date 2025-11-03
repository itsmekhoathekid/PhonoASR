from . import TransformerDecoder, ConformerDecoder, BaseDecoder

def build_decoder(config, vocab_size):
    try:
        decoder_type = config['dec'].get('type', 'base')
        if decoder_type == 'base':
            vocab_size = vocab_size
            n_layer = config['dec']['n_layer']
            d_model = config['dec']['d_model']
            d_hidden = config['dec']['ff_size']
            n_head = config['dec']['n_head']
            dropout = config['dec']['dropout']
            k = config['dec']['k']

            return TransformerDecoder(vocab_size, n_layer, d_model, d_hidden, n_head, dropout, k)
        else:
            return BaseDecoder(
                embedding_size=config["dec"]["embedding_size"],
                hidden_size=config["dec"]["hidden_size"],
                vocab_size=vocab_size,
                output_size=config["dec"]["output_size"],
                n_layers=config["dec"]["n_layers"],
            )
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")