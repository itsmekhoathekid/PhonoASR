import torch 
from core.modules import *

from .model import *
from speechbrain.nnet.schedulers import NoamScheduler
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
        self.model, self.optimizer, self.scheduler = self.inits(vocab_size=len(vocab))
        self.alpha_k = [1.0] if self.config['model']['dec']['k'] == 1 else [0.2 for _ in range(self.config['model']['dec']['k'])]
        self.type_training = self.config['training']['type_training']
        
        self.ctc_loss = CTCLoss(blank=vocab.get_blank_token(), reduction='batchmean').to(self.device)
        self.kldiv_loss = Kldiv_Loss(pad_idx=vocab.get_pad_token(), reduction='batchmean')
        self.ce_loss = CELoss(ignore_index=vocab.get_pad_token(), reduction='mean').to(self.device)
        self.transducer_loss = RNNTLoss(blank=vocab.get_blank_token(), reduction='mean').to(self.device)

        self.vocab = vocab

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
    
    def load_checkpoint(self):
        if os.path.isfile():
            load_path = os.path.join(self.checkpoint_path, f"{self.config['model']['model_name']}.ckpt")
            
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint["epoch"]
            scores = checkpoint["score"]
            
            self.scheduler.load(os.path.join(self.checkpoint_path, f"{self.config['model']['model_name']}_scheduler.ckpt"))
            logging.info(f"Reloaded model from {load_path} at epoch {epoch}")
        else:
            logging.info("No checkpoint found. Starting from scratch.")

        return {
            "epoch": epoch,
            **scores
        }
    
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

        type_decode = self.config["infer"]['type_decode']
        if type_decode == "mtp_stack":
            predictor = GreedyMPStackPredictor(self.model, self.vocab, self.device)
        elif type_decode == "mtp":
            predictor = GreedyMutiplePredictor(self.model, self.vocab, self.device, tau=self.config["infer"]["tau"])
        else:
            predictor = GreedyPredictor(self.model, self.vocab, self.device)

        all_gold_texts = []
        all_predicted_texts = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                src = batch['fbank'].to(self.device)
                src_mask = batch['fbank_mask'].to(self.device)
                tokens = batch["tokens"].to(self.device)
                predicted_tokens = predictor.greedy_decode(src, src_mask)

                batch_size = src.size(0)
                
                # Process each sample in the batch
                for batch_idx in range(batch_size):
                    if type_decode != "mtp_stack":
                        sample_tokens = predicted_tokens[batch_idx]  # [seq_len]
                        
                        predicted_tokens_clean = [
                            token for token in sample_tokens
                            if token != predictor.sos and token != predictor.eos and token != predictor.blank
                        ]
                        predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]

                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        gold_text = [predictor.tokenizer[token] for token in sample_gold_tokens if token != predictor.blank]
                        gold_text_str = ' '.join(gold_text)
                        predicted_text_str = ' '.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                    
                    else:
                        sample_tokens = predicted_tokens[batch_idx]
                        
                        predicted_tokens_flat = []
                        for triplet in sample_tokens:
                            predicted_tokens_flat.extend(triplet)
                            predicted_tokens_flat.append(predictor.space)
                        
                        predicted_tokens_clean = [
                            token for token in predicted_tokens_flat
                            if token != predictor.sos and token != predictor.eos and token != predictor.blank and token != predictor.pad
                        ]
                        predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]
        
                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        tokens_cpu_flat = []
                        for triplet in sample_gold_tokens:
                            tokens_cpu_flat.extend(triplet)
                            tokens_cpu_flat.append(predictor.space)
                        
                        gold_text = [predictor.tokenizer[token] for token in tokens_cpu_flat if token not in [predictor.blank, predictor.sos, predictor.eos, predictor.pad]]
                        gold_text_str = ''.join(gold_text)
                        gold_text_str = gold_text_str.replace(predictor.tokenizer[predictor.space], ' ')
                        predicted_text_str = ''.join([t for t in predicted_text if t not in [predictor.blank, predictor.sos, predictor.eos, predictor.pad]])
                        predicted_text_str = predicted_text_str.replace(predictor.tokenizer[predictor.space], ' ')

                    if self.config['training']['type'] == "phoneme":
                        predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                        space_token = self.vocab.get("<space>")
                        predicted_text_str = predicted_text_str.replace(predictor.tokenizer[space_token], ' ')

                        gold_text_str = ''.join([predictor.tokenizer[token] for token in sample_gold_tokens if token != predictor.blank])
                        gold_text_str = gold_text_str.replace(predictor.tokenizer[space_token], ' ')
                    elif self.config['training']['type'] == "char":
                        predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                        gold_text_str = ''.join([predictor.tokenizer[token] for token in sample_gold_tokens if token != predictor.blank])
                    
                    all_gold_texts.append(gold_text_str)
                    all_predicted_texts.append(predicted_text_str)
            
            total_wer = wer(all_gold_texts, all_predicted_texts)
            total_cer = cer(all_gold_texts, all_predicted_texts)

        return {
            "wer": total_wer,
            "cer": total_cer
        }

    def save_checkpoint(self, epoch, wer, cer, mode):
        
        model_name = f"{self.config['model']['model_name']}.ckpt" if mode == "latest" else f"best_{self.config['model']['model_name']}.ckpt"
        scheduler_name = f"{self.config['model']['model_name']}_scheduler.ckpt" if mode == "latest" else f"best_{self.config['model']['model_name']}_scheduler.ckpt"
        torch.save({
            'epoch': epoch,
            "wer": wer,
            "cer": cer,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(
            self.config['training']['save_path'],
            model_name
        ))
        
        self.scheduler.save(os.path.join(self.checkpoint_path, scheduler_name))

    def run_train(self, train_loader, valid_loader):
        if not os.path.exists(self.config['training']['save_path']):
            os.makedirs(self.config['training']['save_path'])
        
        if self.config['training']['reload']:
            checkpoint = self.load_checkpoint()
            epoch = checkpoint["epoch"]
            scores = checkpoint["scores"]
            best_wer = scores["wer"]
        else:
            epoch = 1
            best_wer = 1.

        num_epochs = self.config['training'].get('num_epochs', 0) # if 0 then train til early stop
        
        patience = self.config['training'].get('patience', 10)
        no_improve_epochs = 0
        daily_save = self.config['training'].get('daily_save', True)

        while True:
            logging.info(f"Epoch {epoch}")

            self.train(train_loader)
            results = self.evaluate(valid_loader)
            current_wer = results["wer"]
            current_cer = results["cer"]

            if current_wer < best_wer:
                best_wer = current_wer
                self.save_checkpoint(epoch, best_wer, current_cer,  mode="best")
                no_improve_epochs = 0
                
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logging.info(f"No improvement for {patience} epochs. Stopping training.")
                    break
            if daily_save:
                self.save_checkpoint(epoch, current_wer, current_cer, mode="latest")
            epoch += 1
            logging.info(f"CER: {current_cer:.4f}, WER: {current_wer:.4f}, Best WER: {best_wer:.4f}")
            if num_epochs > 0 and epoch > num_epochs:
                logging.info("Reached maximum number of epochs. Stopping training.")
                break
            
                

    def run_eval(self, test_loader):
        self.model.eval()

        type_decode = self.config["infer"].get('type_decode', 'mtp_stack')
        if type_decode == "mtp_stack":
            predictor = GreedyMPStackPredictor(self.model, self.vocab, self.device)
        elif type_decode == "mtp":
            predictor = GreedyMutiplePredictor(self.model, self.vocab, self.device, tau=self.config["infer"]["tau"])
        else:
            predictor = GreedyPredictor(self.model, self.vocab, self.device)

        all_gold_texts = []
        all_predicted_texts = []
        results = []
        result_path = self.config['training']['result']
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                src = batch['fbank'].to(self.device)
                src_mask = batch['fbank_mask'].to(self.device)
                tokens = batch["tokens"].to(self.device)
                predicted_tokens = predictor.greedy_decode(src, src_mask)

                batch_size = src.size(0)
                
                # Process each sample in the batch
                for batch_idx in range(batch_size):
                    if type_decode != "mtp_stack":
                        sample_tokens = predicted_tokens[batch_idx]  # [seq_len]
                        
                        predicted_tokens_clean = [
                            token for token in sample_tokens
                            if token != predictor.sos and token != predictor.eos and token != predictor.blank
                        ]
                        predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]

                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        gold_text = [predictor.tokenizer[token] for token in sample_gold_tokens if token != predictor.blank]
                        gold_text_str = ' '.join(gold_text)
                        predicted_text_str = ' '.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                    
                    else:
                        sample_tokens = predicted_tokens[batch_idx]
                        
                        predicted_tokens_flat = []
                        for triplet in sample_tokens:
                            predicted_tokens_flat.extend(triplet)
                            predicted_tokens_flat.append(predictor.space)
                        
                        predicted_tokens_clean = [
                            token for token in predicted_tokens_flat
                            if token != predictor.sos and token != predictor.eos and token != predictor.blank and token != predictor.pad
                        ]
                        predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]
        
                        sample_gold_tokens = tokens[batch_idx].cpu().tolist() 
                        tokens_cpu_flat = []
                        for triplet in sample_gold_tokens:
                            tokens_cpu_flat.extend(triplet)
                            tokens_cpu_flat.append(predictor.space)
                        
                        gold_text = [predictor.tokenizer[token] for token in tokens_cpu_flat if token not in [predictor.blank, predictor.sos, predictor.eos, predictor.pad]]
                        gold_text_str = ''.join(gold_text)
                        gold_text_str = gold_text_str.replace(predictor.tokenizer[predictor.space], ' ')
                        predicted_text_str = ''.join([t for t in predicted_text if t not in [predictor.blank, predictor.sos, predictor.eos, predictor.pad]])
                        predicted_text_str = predicted_text_str.replace(predictor.tokenizer[predictor.space], ' ')

                    if self.config['training']['type'] == "phoneme":
                        predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                        space_token = self.vocab.get("<space>")
                        predicted_text_str = predicted_text_str.replace(predictor.tokenizer[space_token], ' ')

                        gold_text_str = ''.join([predictor.tokenizer[token] for token in sample_gold_tokens if token != predictor.blank])
                        gold_text_str = gold_text_str.replace(predictor.tokenizer[space_token], ' ')
                    elif self.config['training']['type'] == "char":
                        predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                        gold_text_str = ''.join([predictor.tokenizer[token] for token in sample_gold_tokens if token != predictor.blank])
                    
                    all_gold_texts.append(gold_text_str)
                    all_predicted_texts.append(predicted_text_str)

                    wer_score = wer(gold_text_str, predicted_text_str)
                    cer_score = cer(gold_text_str, predicted_text_str)
                    results.append({
                        "gold": gold_text_str,
                        "predicted": predicted_text_str,
                        "WER": wer_score,
                        "CER": cer_score,
                    })
                    print(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}")
            
            total_wer = wer(all_gold_texts, all_predicted_texts)
            total_cer = cer(all_gold_texts, all_predicted_texts)

            results.append({
                "WER": total_wer,
                "CER": total_cer
            })

        json.dump(results, open(result_path, "w+"), ensure_ascii=False, indent=4)
    
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
