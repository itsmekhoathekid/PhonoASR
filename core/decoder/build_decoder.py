from . import TransformerDecoder, ConformerDecoder, BaseDecoder, SaaDecoder, VGGTransformerDecoder

def build_decoder(config, vocab_size):
    try:
        decoder_type = config['dec'].get('type', 'base')
        valid_types = ['base', 'saa_dec', 'vgg_dec', 'transducer']
        
        if decoder_type == 'base':
            vocab_size = vocab_size
            n_layer = config['dec']['n_layer']
            d_model = config['dec']['d_model']
            d_hidden = config['dec']['ff_size']
            n_head = config['dec']['n_head']
            dropout = config['dec']['dropout']
            k = config['dec']['k']

            return TransformerDecoder(vocab_size, n_layer, d_model, d_hidden, n_head, dropout, k)
        elif decoder_type == 'saa_dec':
            return SaaDecoder(
                vocab_size=vocab_size,
                embedding_dim=config["dec"]["embed_dim"],
                hidden_size=config["dec"]["d_hidden"],
                num_layers=config["dec"]["num_layers"],
                embed_dropout=config["dec"].get("embed_dropout", 0.1),
                var_dropout=config["dec"].get("var_dropout", 0.2),
            )
        elif decoder_type == 'vgg_dec':
            return VGGTransformerDecoder(
                vocab_size=vocab_size,
                n_layers=config["dec"]["n_layers"],
                d_model=config["dec"]["d_model"],
                ff_size=config["dec"]["ff_size"],
                h=config["dec"]["n_head"],
                p_dropout=config["dec"]["dropout"],
            )
        elif decoder_type == 'transducer':
            return BaseDecoder(
                embedding_size=config["dec"]["embedding_size"],
                hidden_size=config["dec"]["hidden_size"],
                vocab_size=vocab_size,
                output_size=config["dec"]["output_size"],
                n_layers=config["dec"]["n_layers"],
                dropout=config["dec"]["dropout"]
            )
        else: 
            raise ValueError(f"Decoder type '{decoder_type}' is not supported. "
                     f"Supported types: {valid_types}")
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")