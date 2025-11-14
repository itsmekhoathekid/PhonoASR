import torch
from core import AcousticModel
import yaml
import os 
from dataset import causal_mask


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, vocab_len: int, device: torch.device, epoch : int):
    checkpoint_path = os.path.join(
        config['training']['save_path'],
        f"{config['model']['model_name']}_epoch_{epoch}"
    )
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = AcousticModel(
        config=config,
        vocab_size=vocab_len
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class GreedyPredictor:
    def __init__(self, model, vocab, device, max_len=100):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.blank = vocab.get_blank_token()
        self.pad = vocab.get_pad_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
    @torch.no_grad()
    def greedy_decode(self, src, src_mask):
        B = src.size(0)
        
        # Encode
        enc_out, src_mask , src_len = self.model.encode(src, src_mask)

        # Init decoder input: [B, 1]
        decoder_input = torch.full((B, 1), self.sos, dtype=torch.long, device=self.device)

        # Track finished sequences
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(self.max_len):
            tgt_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)
            dec_out = self.model.decode(decoder_input, enc_out, src_mask, tgt_mask)

            # logits for next token: [B, vocab]
            logits = dec_out[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)  # [B]

            # Replace next_tokens with BLANK for finished ones (force pad)
            next_tokens = next_tokens.masked_fill(finished | (next_tokens == self.sos) | (next_tokens == self.blank), self.blank)
            
            # Cập nhật finished
            finished |= (next_tokens == self.eos)

            # Thêm token vào input decoder
            next_tokens = next_tokens.unsqueeze(1)  # [B, 1]
            decoder_input = torch.cat([decoder_input, next_tokens], dim=1)

            if finished.all():
                break
    
        return decoder_input.cpu().numpy()



class GreedyMPStackPredictor:
    def __init__(self, model, vocab, device, max_len=150):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.pad = 0
        self.blank = vocab.get_blank_token()
        self.space = vocab.get_space_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
    
    def greedy_decode(self, src, src_mask):
        B = src.size(0)
        enc_out, src_mask, src_len = self.model.encode(src, src_mask)
        # decoder_input = torch.tensor([[[self.sos, self.sos, self.sos]]], dtype=torch.long).to(self.device)
        decoder_input = torch.full((B,1,3), self.sos, dtype= torch.long, device= self.device)

        break_flag = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(self.max_len):
            decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)
            dec_out = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
            initial = dec_out[0][:, -1, :]  # [B, vocab_size]
            rhyme = dec_out[1][:, -1, :]  # [B, vocab_size]
            tone = dec_out[2][:, -1, :]  # [B, vocab_size]

            _, initial_tokens = torch.max(initial, dim=1)  # [B]
            _, rhyme_tokens = torch.max(rhyme, dim=1)  # [B]
            _, tone_tokens = torch.max(tone, dim=1)  # [B]

            # Tạo next_token_tensor
            next_token_tensor = torch.stack([initial_tokens, rhyme_tokens, tone_tokens], dim=1).unsqueeze(1)  # [B,1,3]

            # Kiểm tra EOS cho từng sample trong batch
            eos_mask = ((initial_tokens == self.eos) | (rhyme_tokens == self.eos) | (tone_tokens == self.eos))
            
            # Chỉ thêm token cho những sample chưa gặp EOS
            new_break_flags = torch.zeros_like(break_flag)
            
            for b in range(B):
                if not break_flag[b]:  # Chỉ xử lý những sample chưa break
                    if eos_mask[b]:
                        # Nếu gặp EOS lần đầu, thay thế toàn bộ triplet bằng EOS
                        next_token_tensor[b, 0, :] = self.eos
                        new_break_flags[b] = True
                    # Nếu không gặp EOS, giữ nguyên next_token_tensor[b]
                else:
                    # Nếu đã break trước đó, thêm PAD token
                    next_token_tensor[b, 0, :] = self.pad
            
            # Cập nhật break_flag
            break_flag = break_flag | new_break_flags

            # Cập nhật decoder_input cho tất cả samples
            decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)
            
            # Kiểm tra xem tất cả samples đã hoàn thành chưa
            if break_flag.all():
                break
        # print("Output shape (B, seq_len, 3): ", decoder_input.shape)
        # print(decoder_input[0, :, :])
        # print('==============')
        # print(decoder_input[1, :, :])
        # exit(0)
        return decoder_input.cpu().numpy()  # Trả về [B, seq_len, 3]

import torch

# Self-Speculative Decoding 

class GreedyMutiplePredictor:
    def __init__(self, model, vocab, device, max_len=50, n_heads=3):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.blank = vocab.get_blank_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
        self.n_heads = n_heads


    @torch.no_grad()
    def greedy_decode(self, src, src_mask):
        # ===== 1. Encode =====
        enc_out, src_mask, src_len = self.model.encode(src, src_mask)
        decoder_input = torch.tensor([[self.sos]], dtype=torch.long, device=self.device)

        fwd_pred, fwd_verify = 0, 0

        for _ in range(self.max_len):
            # causal mask for current prefix
            decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)

            # ===== 2. Predict phase (multi-head) =====
            dec_out_heads = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
            fwd_pred += 1
            probs_heads = [torch.softmax(h[:, -1, :], dim=-1) for h in dec_out_heads]
            draft_tokens = [torch.argmax(p[0], dim=-1).item() for p in probs_heads]

            if draft_tokens[0] == self.eos:
                break

            # ===== 3. Verify phase (single forward, blockwise) =====
            # prefix + first (K−1) draft tokens
            seq_full = torch.cat(
                [decoder_input, torch.tensor([draft_tokens[:-1]], device=self.device)], dim=1
            )
            verify_mask = causal_mask(src.size(0), seq_full.size(1)).to(self.device)
            dec_out_verify = self.model.decode(seq_full, enc_out, src_mask, verify_mask)
            fwd_verify += 1

            # logits from main head p₁ for all time steps of seq_full
            p1_logits_all = dec_out_verify[0]            # [1, L+(K−1), V]
            L = decoder_input.size(1)
            p1_block = p1_logits_all[:, L-1 : L-1+self.n_heads, :]  # [1,K,V]
            

            # ===== 4. Find largest k̂ =====
            if self.tau is not None:
                p1_probs = torch.exp(p1_block[0])  # convert log-prob to prob
                ok = p1_probs[torch.arange(self.n_heads), torch.tensor(draft_tokens, device=self.device)] >= self.tau
                if ok.any():
                    false_idx = (~ok).nonzero(as_tuple=False)
                    k_hat = false_idx[0,0].item() if false_idx.numel() > 0 else self.n_heads
                else:
                    k_hat = 0
            else:
                p1_probs = torch.softmax(p1_block, dim=-1)[0]          # [K,V]
                p1_pred = torch.argmax(p1_probs, dim=-1)
                matches = (p1_pred == torch.tensor(draft_tokens, device=self.device))
                if matches.any():
                    false_idx = (~matches).nonzero(as_tuple=False)
                    k_hat = false_idx[0,0].item() if false_idx.numel() > 0 else self.n_heads
                else:
                    k_hat = 0

            # always accept head-1’s token
            if k_hat == 0:
                k_hat = 1
            accepted = draft_tokens[:k_hat]

            # ===== 5. Append accepted tokens =====
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([accepted], device=self.device)], dim=1
            )

            if self.eos in accepted:
                break

        total_fw = fwd_pred + fwd_verify
        print(f"🧮 Total forward calls: {total_fw}")
        print(f"   ├─ Predict phase : {fwd_pred}")
        print(f"   └─ Verify phase  : {fwd_verify}")
        print(f"📏 Output length   : {decoder_input.size(1)}")
        print(f"⚡ Avg forward/token: {round(total_fw / decoder_input.size(1), 3)}")
        
        return decoder_input.squeeze(0).cpu().numpy()

