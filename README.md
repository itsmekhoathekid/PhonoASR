# 🎙️ PhonoASR — Phoneme-Aware End-to-End Speech Recognition

PhonoASR is an end-to-end Automatic Speech Recognition (ASR) framework designed for Vietnamese speech, with **phoneme-level modeling**, **multi-mode decoding**, and support for **CTC-KL**, **RNNT**, and **Cross-Entropy** training strategies.

This project enables flexible experimentation with phonetic supervision to improve speech recognition quality on Vietnamese datasets.

---

## 🚀 Features

- ✅ Phoneme-aware encoder–decoder architecture
- ✅ Multi-mode training:
  - CTC + KL-divergence
  - RNNT (Transducer)
  - Cross-Entropy phoneme decoder
- ✅ Configurable decoding:
  - Word-based
  - Character-based
  - Pure-phoneme mode
- ✅ Supports dynamic batch & variable-length input
- ✅ Plug-and-play architecture modules

---

## 📁 Configuration Guide

### 🎛 Training Settings

```yaml
training:
    ctc_weight:         # Weight for CTC loss (only in "ctc-kldiv" mode)

    type_training:      # Training objective
        # "ctc-kldiv"   → CTC + KL-divergence loss (decoder k = 1)
        # "transducer"  → RNN-Transducer (RNNT) loss
        # "ce"          → Cross-entropy phoneme decoder (decoder k = 3)

    epochs:
        # 0  → Train until early stopping
        # >0 → Train for a fixed number of epochs

    type:               # Inference mode / tokenizer
        # "word"     → Phoneme decoder → word output
        # "char"     → Character-level output
        # "phoneme"  → Pure phoneme mode (no phoneme decoder)
```

---

### 🎯 RNNT Loss Configuration

```yaml
rnnt_loss:
    blank:   # Blank token index (must match pad_id)
```

> ⚠️ Ensure `blank == pad_id` when using RNNT.

---

### 📌 Summary Table

| Mode | Description | Decoder Setting |
|------|------------|----------------|
`ctc-kldiv` | CTC + KL divergence | `k = 1` |
`transducer` | RNNT loss | — |
`ce` | Cross-entropy phoneme decoder | `k = 3` |

---


