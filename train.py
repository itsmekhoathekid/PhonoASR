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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']
    logg(training_cfg['logg'])

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        training_cfg,
        type='train',
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
        training_cfg,
        type='dev',
        type_training= config['training'].get('type_training', 'ctc-kldiv')
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )

    test_dataset = Speech2Text(
        training_config=config['training'],
        type='test',
        type_training= config['training'].get('type_training', 'ctc-kldiv')
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'] if args.type_decode == "mtp_stack" else 1,  
        shuffle=False,
        collate_fn=speech_collate_fn
    )

    trainer = Engine(config, vocab = train_dataset.vocab)
    trainer.run_train(train_loader, dev_loader)
    trainer.run_eval(test_loader)

if __name__ == "__main__":
    main()