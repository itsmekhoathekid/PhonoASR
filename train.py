import torch
from dataset import Speech2Text, speech_collate_fn
from core.modules import (
    logg
)
from core import *
import argparse
import yaml



def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def main():
    from torch.optim import Adam
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']
    logg(training_cfg['logg'])

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
        type_training= config['training'].get('type_training', 'ctc-kldiv')
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )

    dev_dataset = Speech2Text(
        json_path=training_cfg['dev_path'],
        vocab_path=training_cfg['vocab_path'],
        type_training= config['training'].get('type_training', 'ctc-kldiv')
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )

    trainer = Engine(config, vocab = train_dataset.vocab)
    trainer.run_train_eval(train_loader, dev_loader)



if __name__ == "__main__":
    main()