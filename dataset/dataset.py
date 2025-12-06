import torch
from torch.utils.data import Dataset
import torch
from speechbrain.lobes.features import Fbank
import speechbrain as sb
import os
import librosa
import numpy as np
import torchaudio
import torch.nn as nn


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding= 'utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def get_blank_token(self):
        return self.stoi["<blank>"]
    def get_space_token(self):
        return self.stoi["<blank>"] # <blank>
    def __len__(self):
        return len(self.vocab)


class AudioPreprocessing(nn.Module):

    """Audio Preprocessing

    Computes mel-scale log filter banks spectrogram

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT frame size, creates n_fft // 2 + 1 frequency bins.
        win_length_ms: FFT window length in ms, must be <= n_fft
        hop_length_ms: length of hop between FFT windows in ms
        n_mels: number of mel filter banks
        normalize: whether to normalize mel spectrograms outputs
        mean: training mean
        std: training std

    Shape:
        Input: (batch_size, audio_len)
        Output: (batch_size, n_mels, audio_len // hop_length + 1)
    
    """

    def __init__(self, sample_rate, n_fft, win_length_ms, hop_length_ms, n_mels, normalize, mean, std):
        super(AudioPreprocessing, self).__init__()
        self.win_length = int(sample_rate * win_length_ms) // 1000
        self.hop_length = int(sample_rate * hop_length_ms) // 1000
        self.Spectrogram = torchaudio.transforms.Spectrogram(n_fft, self.win_length, self.hop_length)
        self.MelScale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min=0, f_max=8000, n_stft=n_fft // 2 + 1)
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, x, x_len):

        # Short Time Fourier Transform (B, T) -> (B, n_fft // 2 + 1, T // hop_length + 1)
        x = self.Spectrogram(x)

        # Mel Scale (B, n_fft // 2 + 1, T // hop_length + 1) -> (B, n_mels, T // hop_length + 1)
        x = self.MelScale(x)
        
        # Energy log, autocast disabled to prevent float16 overflow
        x = (x.float() + 1e-9).log().type(x.dtype)

        # Compute Sequence lengths 
        if x_len is not None:
            x_len = torch.div(x_len, self.hop_length, rounding_mode='floor') + 1

        # Normalize
        if self.normalize:
            x = (x - self.mean) / self.std

        return x


class AudioProcessor:
    def __init__(self, config):
        self.model_name = config['model']['enc'].get('type','None')
        if self.model_name == 'None':
            raise "Havent specified the model type, cant do shit"
        
        self.win_length = config['training'].get('win_length', 25)
        self.hop_length = config['training'].get('win_length', 10)
        self.n_fft = config['training'].get('n_fft', 512)
        self.n_mels = config['training'].get('n_mels', 80)
        self.sr = config['training'].get('sample_rate', 16000)
        self.fbank = Fbank(
            sample_rate=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length
        )
        self.gmvn_mean = None
        self.gmvn_std = None
        self.audio_preprocessing = AudioPreprocessing(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length_ms=self.win_length,
            hop_length_ms=self.hop_length,
            n_mels=self.n_mels,
            normalize=config['training'].get('normalize', False),
            mean=config['training'].get('mean', 0.0),
            std=config['training'].get('std', 1.0)
        )
        

    def stack_context(self, x, left=3, right=1):
        """x: (T, D) -> (T, (left+1+right)*D) | pad biên bằng replicate."""
        T, D = x.shape
        pads = []
        for off in range(-left, right + 1):
            idx = np.clip(np.arange(T) + off, 0, T - 1)
            pads.append(x[idx])
        return np.concatenate(pads, axis=1)

    def subsample(self, x, base_hop_ms=10, target_hop_ms=30):
        stride = target_hop_ms // base_hop_ms
        return x[::stride]
    
    def tr_tr_audio_process(self, wav_file):
        y, sr = librosa.load(wav_file, sr=self.sr)
        win_length = int(self.win_length / 1000 * sr)   # 25 ms
        hop_length = int(self.hop_length / 1000 * sr)   # 10 ms
        # n_fft = next power of 2 >= win_length
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length,
            window='hann', power=2.0, center=True
        )
        # log-mel (dB)
        x = librosa.power_to_db(S, ref=np.max).T   # (T, 40)
        
        mu = x.mean(axis=0, keepdims=True)
        sg = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mu) / sg
        x = self.stack_context(x, left=3, right=1) 
        return torch.tensor(self.subsample(x, 10, 30))

    def conv_rnnt_audio_process(self, wav_file, sr = 16000):
        # Load waveform
        y, sr = librosa.load(wav_file, sr= self.sr)

        # Window và hop size
        win_length = int(self.win_length / 1000 * sr)   # 25 ms
        hop_length = int(self.hop_length / 1000 * sr)   # 10 ms

        # STFT magnitude
        stft = librosa.stft(y, n_fft=self.n_fft, win_length=win_length, hop_length=hop_length, window='hamming')
        mag = np.abs(stft[:64, :])  # Lấy 64 bins đầu tiên (low frequencies)

        # Log magnitude
        log_mag = np.log1p(mag)  # log(1 + x)

        # Transpose: (64, T) -> (T, 64)
        log_mag = log_mag.T

        # Frame stacking: 3 frames, skip = 3
        stacked_feats = []
        for i in range(0, len(log_mag) - 6, 3):  # skip rate = 3
            stacked = np.concatenate([log_mag[i], log_mag[i+3], log_mag[i+6]])
            stacked_feats.append(stacked)

        stacked_feats = torch.tensor(np.array(stacked_feats), dtype=torch.float)
        mean_feats = stacked_feats.mean(dim=0, keepdim=True)
        std_feats = stacked_feats.std(dim=0, keepdim=True)

        # stacked_feats = (stacked_feats - self.gmvn_mean) / (self.gmvn_std + 1e-5)
        if self.gmvn_mean is not None and self.gmvn_std is not None:
            stacked_feats = (stacked_feats - self.gmvn_mean) / (self.gmvn_std + 1e-5)
        else:
            stacked_feats = (stacked_feats - mean_feats) / (std_feats + 1e-5)
        return stacked_feats 
    
    

    def normal_audio_process(self, wav_path):
        sig  = sb.dataio.dataio.read_audio(wav_path)
        
        features = self.fbank(sig.unsqueeze(0))
        features = features.squeeze(0)
        return features
    

    def comformer_audio_process(self, wav_file):
        y, sr = torchaudio.load(wav_file)
        audio = self.audio_preprocessing(y, None)
        # print(audio.squeeze(0).shape)  # [T, 80]
        return audio.squeeze(0).transpose(0,1)  # [T, 80]

    def extract_audio_feats(self, wav_path):
        if self.model_name == "TransformerTransducer":
            return self.tr_tr_audio_process(wav_file=wav_path)
        elif self.model_name == "ConvRNNT":
            return self.conv_rnnt_audio_process(wav_file=wav_path)
        elif self.model_name == "Conformer" or self.model_name == "ConvConformer":
            return self.comformer_audio_process(wav_file=wav_path)
        else:
            return self.normal_audio_process(wav_path=wav_path)





class Speech2Text(Dataset):
    def __init__(self, config, type, type_training = "ctc-kldiv"):
        super().__init__()
        self.config = config
        if type == 'train':
            json_path = config['training']['train_path']
        elif type == 'dev':
            json_path = config['training']['dev_path']
        elif type == 'test':
            json_path = config['training']['test_path']
            
        self.wave_path = config['training']["wave_path"]
        vocab_path = config['training']['vocab_path']
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()
        self.apply_spec_augment = config['training'].get('apply_spec_augment', False)
        self.audio_processor = AudioProcessor(config)
        self.type_training = type_training


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = os.path.join(self.wave_path, current_item["wav_path"])
        if self.type_training == "ce" and self.config['model']['dec']['k'] == 3:
            encoded_text = torch.tensor(current_item["encoded_text"] + [[self.eos_token, self.eos_token, self.eos_token]], dtype=torch.long)
            decoder_input = torch.tensor([[self.sos_token, self.sos_token, self.sos_token]] + current_item["encoded_text"], dtype=torch.long)
        elif self.type_training == 'transducer':
            encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
            decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"] + [self.pad_token], dtype=torch.long)
        else:
            encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
            decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
        tokens = torch.tensor(current_item["encoded_text"], dtype=torch.long)
        fbank = self.audio_processor.extract_audio_feats(wav_path).float()  # [T, 512]

        return {
            "text": encoded_text,
            "fbank": fbank,
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,
            "tokens": tokens,
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def causal_mask(batch_size, size):
    """Tạo mask cho decoder để tránh nhìn thấy tương lai"""
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, T]

def speech_collate_fn(batch):
    decoder_outputs = [item["decoder_input"].detach().clone() for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)
    tokens_lens = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)      # [B, T_text]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    # print(calculate_mask(text_lens, padded_texts.size(1)).shape)
    # print(causal_mask(padded_texts.size(0), padded_texts.size(1)).shape)
    
    text_mask= calculate_mask(text_lens, padded_texts.size(1)).unsqueeze(1) & causal_mask(padded_texts.size(0), padded_texts.size(1))  # [B, T_text, T_text]
    text_mask = text_mask.unsqueeze(1)  # [B, 1, T_text, T_text]
    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask,
        "tokens" : padded_tokens,
        "tokens_lens": tokens_lens
    }


import logging
import os 

def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )


def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def causal_mask(batch_size, size):
    """Tạo mask cho decoder để tránh nhìn thấy tương lai"""
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, T]