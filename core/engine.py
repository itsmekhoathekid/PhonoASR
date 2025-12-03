import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from core.modules import *

from .model import *
# from speechbrain.nnet.schedulers import NoamScheduler
from core.inference import GreedyMPStackPredictor, GreedyMutiplePredictor, GreedyPredictor

import os 
import logging
from tqdm import tqdm
from jiwer import wer, cer
import json

class Engine:
    def __init__(self, config, vocab):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = config['training']['save_path']
        self.warmup = self.config["scheduler"]["n_warmup_steps"]
        self.d_model = self.config["model"].get("d_model", self.warmup)
        if self.d_model == self.warmup:
            self.normalize = self.warmup ** 0.5
        else:
            self.normalize = self.d_model ** -0.5

        self.model, self.optimizer = self.inits(vocab_size=len(vocab))
        self.scheduler = LambdaLR(self.optimizer, self.lambda_lr)
        self.alpha_k = [1.0] if self.config['model']['dec']['k'] == 1 else [1.0 for _ in range(self.config['model']['dec']['k'])]
        self.type_training = self.config['training']['type_training']
        
        self.ctc_loss = CTCLoss(blank=vocab.get_blank_token(), reduction='batchmean').to(self.device)
        self.kldiv_loss = Kldiv_Loss(pad_idx=vocab.get_pad_token(), reduction='batchmean')
        self.ce_loss = CELoss(ignore_index=vocab.get_pad_token(), reduction='mean').to(self.device)
        self.transducer_loss = RNNTLoss(blank=vocab.get_blank_token(), reduction='mean').to(self.device)
        self.vocab = vocab  
        self.predictor = self.get_predictor()
        self.no_improve_epochs = 0

    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return self.normalize * min(step ** -.5, step * warm_up ** -1.5)
        
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
        optimizer = Adam(model.parameters(), lr=self.config['optim']["lr"])
        
        # scheduler = NoamScheduler(
        #     n_warmup_steps=self.config['scheduler']['n_warmup_steps'],
        #     lr_initial=self.config['scheduler']['lr_initial']
        # )

        return model, optimizer

    def get_predictor(self):
        max_len = self.config['infer']['max_output_len']
        type_decode = self.config["infer"]['type_decode']
        if type_decode == "mtp_stack" and self.config['training']['type_training'] == 'ce' and self.config['model']['dec']['k'] == 3:
            predictor = GreedyMPStackPredictor(self.model, self.vocab, self.device, max_len = max_len)
        elif type_decode == "mtp_stack" and self.config['model']['dec']['k'] == 1:
            predictor = GreedyPredictor(self.model, self.vocab, self.device, max_len = max_len)
        else:
            predictor =  GreedyPredictor(self.model, self.vocab, self.device, max_len = max_len)
        
        return predictor

    def load_checkpoint(self):
        mode = self.config['training'].get('reload_mode', 'latest')
        model_name = f"{self.config['model']['model_name']}.ckpt" if mode == "latest" else f"best_{self.config['model']['model_name']}.ckpt"
        load_path = os.path.join(self.checkpoint_path, model_name)
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.no_improve_epochs = checkpoint.get("no_improve_epochs", 0)
        # self.scheduler.load(os.path.join(self.checkpoint_path, f"{self.config['model']['model_name']}_scheduler.ckpt"))
        # epoch = checkpoint["epoch"]
        # wer = checkpoint["wer"]

        return checkpoint
    
    def get_loss(self, enc_out, dec_out, enc_lens, text_len, tokens_eos):
        
        if self.type_training == 'ctc-kldiv':
            ctc_weight = self.config['training']['ctc_weight']
            loss_ctc =  self.ctc_loss(enc_out, tokens_eos, enc_lens, text_len)
            loss_kldiv = self.kldiv_loss(dec_out[0], tokens_eos) # cus k = 1

            loss = loss_ctc * ctc_weight + loss_kldiv * (1 - ctc_weight)
            return loss 
        elif self.type_training == 'ce':
            if self.config['model']['dec']['k'] == 1:
                loss_ep = self.ce_loss(dec_out[0], tokens_eos)
                return loss_ep
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

        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for _, batch in enumerate(progress_bar):
            speech = batch["fbank"].to(device)
            tokens_eos = batch["text"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            _ = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)
            self.optimizer.zero_grad()

            enc_out, dec_out, enc_lens = self.model(
                speech, 
                decoder_input,
                speech_mask,
                text_mask,
                0.8
            )  # [B, T_text, vocab_size]

            loss = self.get_loss(enc_out, dec_out, enc_lens, text_len, tokens_eos)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # === In loss từng batch ===
            progress_bar.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Average training loss: {avg_loss:.4f}, Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

    def inference(self, dataloader, save = False):
        self.model.eval()

        type_decode = self.config["infer"]['type_decode']
        type_training = self.config['training']['type_training']
        all_gold_texts = []
        all_predicted_texts = []
        results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                src = batch['fbank'].to(self.device)
                src_mask = batch['fbank_mask'].to(self.device)
                tokens = batch["tokens"].to(self.device)
                
                if type_training != "transducer":
                    predicted_tokens = self.predictor.greedy_decode(src, src_mask)
                else:
                    predicted_tokens = self.model.greedy_batch(src, src_mask, max_output_len=self.config['infer'].get('max_output_len', 150))

                batch_size = src.size(0)
                
                # Process each sample in the batch
                for batch_idx in range(batch_size):
                    if self.config['model']['dec']['k'] == 1:
                        sample_tokens = predicted_tokens[batch_idx]  # [seq_len]
                        
                        predicted_tokens_clean = [
                            token for token in sample_tokens
                            if token != self.predictor.sos and token != self.predictor.eos and token != self.predictor.blank
                        ]
                        predicted_text = [self.predictor.tokenizer[token] for token in predicted_tokens_clean]

                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        gold_text = [self.predictor.tokenizer[token] for token in sample_gold_tokens if token != self.predictor.blank and token != self.predictor.pad and token != self.predictor.eos]
                        gold_text_str = ' '.join(gold_text)
                        predicted_text_str = ' '.join([t for t in predicted_text if t != self.predictor.blank and t != self.predictor.eos and t != self.predictor.pad])
                    
                    else:
                        sample_tokens = predicted_tokens[batch_idx]
                        
                        predicted_tokens_flat = []
                        for triplet in sample_tokens:
                            predicted_tokens_flat.extend(triplet)
                            predicted_tokens_flat.append(self.predictor.space)
                        
                        predicted_tokens_clean = [
                            token for token in predicted_tokens_flat
                            if token != self.predictor.sos and token != self.predictor.eos and token != self.predictor.blank and token != self.predictor.pad
                        ]
                        predicted_text = [self.predictor.tokenizer[token] for token in predicted_tokens_clean]
        
                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        tokens_cpu_flat = []
                        for triplet in sample_gold_tokens:
                            tokens_cpu_flat.extend(triplet)
                            tokens_cpu_flat.append(self.predictor.space)
                        
                        gold_text = [self.predictor.tokenizer[token] for token in tokens_cpu_flat if token not in [self.predictor.blank, self.predictor.sos, self.predictor.eos, self.predictor.pad]]
                        gold_text_str = ''.join(gold_text)
                        gold_text_str = gold_text_str.replace(self.predictor.tokenizer[self.predictor.space], ' ')
                        predicted_text_str = ''.join([t for t in predicted_text if t not in [self.predictor.blank, self.predictor.sos, self.predictor.eos, self.predictor.pad]])
                        predicted_text_str = predicted_text_str.replace(self.predictor.tokenizer[self.predictor.space], ' ')

                    if self.config['training']['type'] == "phoneme":
                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        predicted_text_str = ''.join([t for t in predicted_text if t != self.predictor.blank and t != self.predictor.eos and t != self.predictor.pad])
                        space_token = self.vocab.get("<space>")
                        predicted_text_str = predicted_text_str.replace(self.predictor.tokenizer[space_token], ' ')

                        gold_text_str = ''.join([self.predictor.tokenizer[token] for token in sample_gold_tokens if token != self.predictor.blank and token != self.predictor.pad and token != self.predictor.eos])
                        gold_text_str = gold_text_str.replace(self.predictor.tokenizer[space_token], ' ')
                    elif self.config['training']['type'] == "char":
                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        predicted_text_str = ''.join([t for t in predicted_text if t != self.predictor.blank and t != self.predictor.eos and t != self.predictor.pad])
                        gold_text_str = ''.join([self.predictor.tokenizer[token] for token in sample_gold_tokens if token != self.predictor.blank and token != self.predictor.pad and token != self.predictor.eos])
                    
                    all_gold_texts.append(gold_text_str)
                    all_predicted_texts.append(predicted_text_str)

                    wer_score = wer(gold_text_str, predicted_text_str)
                    cer_score = cer(gold_text_str, predicted_text_str)

                    gold_text_str = gold_text_str.strip()
                    predicted_text_str = predicted_text_str.strip()

                    if save:
                        results.append({
                            "gold": gold_text_str,
                            "predicted": predicted_text_str,
                            "WER": wer_score,
                            "CER": cer_score,
                        })
                        print(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}")
            
            total_wer = wer(all_gold_texts, all_predicted_texts)
            total_cer = cer(all_gold_texts, all_predicted_texts)

            if save:
                results.append({
                    "total_WER": total_wer,
                    "total_CER": total_cer
                })

                result_path = self.config['training']['result']
                json.dump(results, open(result_path, "w+"), ensure_ascii=False, indent=4)

        return {
            "wer": total_wer,
            "cer": total_cer
        }

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                speech = batch["fbank"].to(self.device)
                tokens_eos = batch["text"].to(self.device)
                speech_mask = batch["fbank_mask"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                text_len = batch["text_len"].to(self.device)

                enc_out , dec_out, enc_lens   = self.model(
                    speech, 
                    decoder_input,
                    speech_mask,
                    text_mask
                )  # [B, T_text, vocab_size]
                
                loss = self.get_loss(enc_out, dec_out, enc_lens, text_len, tokens_eos)

                total_loss += loss.item()
                progress_bar.set_postfix(batch_loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Average validation loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch, mode, best_score):
        
        model_name = f"{self.config['model']['model_name']}.ckpt" if mode == "latest" else f"best_{self.config['model']['model_name']}.ckpt"
        # scheduler_name = f"{self.config['model']['model_name']}_scheduler.ckpt" if mode == "latest" else f"best_{self.config['model']['model_name']}_scheduler.ckpt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "no_improve_epochs" : self.no_improve_epochs, 
            "score": {
                "patience_objective": self.config['training'].get('patience_objective', 'WER'),
                "best_score": best_score
            } 
        }, os.path.join(
            self.config['training']['save_path'],
            model_name
        ))
        
        # self.scheduler.save(os.path.join(self.checkpoint_path, scheduler_name))

    def run_train(self, train_loader, valid_loader):
        if not os.path.exists(self.config['training']['save_path']):
            os.makedirs(self.config['training']['save_path'])
        
        if self.config['training']['reload']:
            checkpoint = self.load_checkpoint()
            epoch = checkpoint["epoch"] + 1
            scores = checkpoint["score"]
            best_score = scores['best_score']
            objective = scores['patience_objective']
            logging.info(f"Reloaded model from checkpoint at epoch {epoch-1} with best {objective}: {best_score:.4f}")
        else:
            epoch = 1
            best_score = 1000.0  # could be WER or CER or val_loss depending on patience_objective

        log_val_loss = self.config['training'].get('log_val_loss', False)
        num_epochs = self.config['training'].get('num_epochs', 0) # if 0 then train til early stop
        
        patience = self.config['training'].get('patience', 10)
        patience_objective = self.config['training'].get('patience_objective', 'WER')
        daily_save = self.config['training'].get('daily_save', True)

        while True:
            logging.info(f"Epoch {epoch}")

            self.train(train_loader)
            
            if log_val_loss:
                val_loss = self.evaluate(valid_loader)
                logging.info(f"Validation Loss: {val_loss:.4f}")
            
            if patience_objective in ['WER', 'CER']:
                results = self.inference(valid_loader)
                current_wer = results["wer"]
                current_cer = results["cer"]

                objective_metric = current_wer if patience_objective == 'WER' else current_cer 
            else:
                objective_metric = val_loss
                current_wer = 0
                current_cer = 0

            if objective_metric < best_score:
                best_score = objective_metric
                self.save_checkpoint(epoch,  mode="best", best_score= best_score)
                self.no_improve_epochs = 0
                
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= patience:
                    logging.info(f"No improvement for {patience} epochs. Stopping training.")
                    break
            if daily_save:
                self.save_checkpoint(epoch, mode="latest", best_score=objective_metric)
            epoch += 1
            logging.info(f"CER: {current_cer:.4f}, WER: {current_wer:.4f}, Best {patience_objective}: {best_score:.4f}")
            if num_epochs > 0 and epoch > num_epochs:
                logging.info("Reached maximum number of epochs. Stopping training.")
                break
            
    def run_eval(self, test_loader):
        load_path = os.path.join(self.checkpoint_path, f"best_{self.config['model']['model_name']}.ckpt")
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"]
            
            # self.scheduler.load(os.path.join(self.checkpoint_path, f"{self.config['model']['model_name']}_scheduler.ckpt"))
            logging.info(f"Reloaded model from {load_path} at epoch {epoch}")
            
            self.inference(test_loader, save=True)
        else:
            logging.info("No checkpoint found.")
    
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