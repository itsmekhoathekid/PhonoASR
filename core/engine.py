import torch 
from core.modules import *
from .model import *
from speechbrain.nnet.schedulers import NoamScheduler
import os 
import logging
from tqdm import tqdm

class Engine:
    def __init__(self, config, vocab):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = config['training']['save_path']
        self.model, self.optimizer, self.scheduler = self.inits(vocab_size=len(vocab))
        self.alpha_k = [1.0] if self.config['model']['dec']['k'] == 1 else [0.2 for _ in range(self.config['model']['dec']['k'])]
        self.type_training = self.config['training']['type_training']
        
        self.ctc_loss = CTCLoss(blank=vocab.get_blank_token(), reduction='batchmean').to(self.device)
        self.kldiv_loss = Kldiv_Loss(pad_idx=vocab.get_pad_token(), reduction='batchmean')
        self.ce_loss = CELoss(ignore_index=vocab.get_pad_token(), reduction='mean').to(self.device)
        self.transducer_loss = RNNTLoss(blank=vocab.get_blank_token(), reduction='mean').to(self.device)

    def inits(self, vocab_size):
        if self.config['training']['type_training'] == 'transducer':
            model = TransducerAcousticModle(
                config=self.config,
                vocab_size=vocab_size
            ).to(self.device)
        else:
            model = AcousticModel(
                config=self.config,
                vocab_size=vocab_size
            ).to(self.device)
        optimizer = Optimizer(model.parameters(), self.config['optim'])
        scheduler = NoamScheduler(
            n_warmup_steps=self.config['scheduler']['n_warmup_steps'],
            lr_initial=self.config['scheduler']['lr_initial']
        )

        return model, optimizer, scheduler
    
    def model_load(self):
        past_epoch = 0
        path_list = [path for path in os.listdir(self.checkpoint_path)]
        if len(path_list) > 0:
            for path in path_list:
                try:
                    past_epoch = max(int(path.split("_")[-1]), past_epoch)
                except:
                    continue
            
            load_path = os.path.join(self.checkpoint_path, f"{self.config['model']['model_name']}_epoch_{past_epoch}")
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load(self.config['training']['save_path'] + f'/scheduler_{self.config["model"]["enc"]["name"]}.ckpt')
            logging.info(f"Reloaded model from {load_path} at epoch {past_epoch}")
        else:
            logging.info("No checkpoint found. Starting from scratch.")
        
        return past_epoch + 1

    
    def get_loss(self, enc_out, dec_out, enc_lens, text_len, tokens_eos):
        
        if self.type_training == 'ctc-kldiv':
            ctc_weight = self.config['training']['ctc_weight']
            loss_ctc =  self.ctc_loss(enc_out, tokens_eos, enc_lens, text_len)
            loss_kldiv = self.kldiv_loss(dec_out[0], tokens_eos) # cus k = 1

            loss = loss_ctc * ctc_weight + loss_kldiv * (1 - ctc_weight)
            return loss 
        elif self.type_training == 'ce':
            B = tokens_eos.size(0)
            loss_ep = sum(self.alpha_k[i] * self.ce_loss(dec_out[i], tokens_eos[...,i].view(B,-1)) for i in range(len(dec_out)))
            return loss_ep
        elif self.type_training == 'transducer':
            loss = self.transducer_loss(enc_out, tokens_eos, enc_lens, text_len)
            return loss

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        device = self.device

        
        progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            speech = batch["fbank"].to(device)
            tokens_eos = batch["text"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            fbank_len = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)
            self.optimizer.zero_grad()

            enc_out, dec_out, enc_lens = self.model(
                speech, 
                decoder_input,
                speech_mask,
                text_mask
            )  # [B, T_text, vocab_size]

            loss = self.get_loss(enc_out, dec_out, enc_lens, text_len, tokens_eos)
            loss.backward()

            self.optimizer.step()

            curr_lr, _ = self.scheduler(self.optimizer.optimizer)

            total_loss += loss.item()

            # === In loss từng batch ===
            progress_bar.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Average training loss: {avg_loss:.4f}")
        return avg_loss, curr_lr

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)
        device = self.device
        with torch.no_grad():
            for batch in progress_bar:
                speech = batch["fbank"].to(device)
                tokens_eos = batch["text"].to(device)
                speech_mask = batch["fbank_mask"].to(device)
                text_mask = batch["text_mask"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                text_len = batch["text_len"].to(device)
                tokens = batch["tokens"].to(device)
                tokens_lens = batch["tokens_lens"].to(device)

                self.optimizer.zero_grad()

                enc_out, dec_out, enc_lens = self.model(
                    speech, 
                    decoder_input,
                    speech_mask,
                    text_mask
                )  # [B, T_text, vocab_size]

                loss = self.get_loss(enc_out, dec_out, enc_lens, text_len, tokens_eos)
                
                # loss = loss_ctc * ctc_weight + loss_ep * (1- ctc_weight)

                total_loss += loss.item()
                progress_bar.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Average validation loss: {avg_loss:.4f}")
        return avg_loss

    def run_train_eval(self, train_loader, valid_loader):
        start_epoch = 1 
        if self.config['training']['reload']:
            start_epoch = self.model_load()
        num_epochs = self.config['training']['epochs']

        for epoch in range(start_epoch, num_epochs + 1):
            logging.info(f"Epoch {epoch}/{num_epochs}")

            train_loss, curr_lr = self.train(train_loader)
            val_loss = self.evaluate(valid_loader)


            logging.info(f"End of epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {curr_lr:.6f}")
        
            if not os.path.exists(self.config['training']['save_path']):
                os.makedirs(self.config['training']['save_path'])
            model_filename = os.path.join(
                self.config['training']['save_path'],
                f"{self.config['model']['model_name']}_epoch_{epoch}"
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_filename)

            self.scheduler.save(self.config['training']['save_path'] + f'/scheduler_{self.config["model"]["enc"]["name"]}.ckpt')
    
    def make_block_targets(self, target, k, pad_id=-100, device='cpu'):
        """
        target: (B, T)
        k: block size
        return: (B, T, k) block targets
        """
        B, T = target.size()
        block_targets = torch.full((B, T, k), pad_id, dtype=target.dtype).to(device)

        for j in range(k):
            if j == 0:
                # p1 giữ nguyên target gốc
                block_targets[:,:,j] = target
            else:
                block_targets[:,:-j,j] = target[:,j:]
                block_targets[:,-j:,j] = pad_id
        return block_targets