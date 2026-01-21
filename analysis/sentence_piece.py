#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train SentencePiece and export vocabulary to JSON.

Outputs:
- <model_prefix>.model, <model_prefix>.vocab
- <model_prefix>_vocab.json                 (full model vocab: id, piece, score)
- <model_prefix>_corpus_piece_counts.json   (optional: token usage counts over the input corpus)

Usage:
  python spm_train_and_export_vocab.py \
      --input /home/anhkhoa/PhonoASR/saves/full_text.txt \
      --model_prefix m \
      --vocab_size 5000 \
      --model_type bpe \
      --hard_vocab_limit false \
      --export_corpus_counts

Notes:
- SentencePiece uses '▁' (U+2581) to represent whitespace.
- JSON export uses ensure_ascii=False to preserve Unicode pieces.
"""

import argparse
import json
import os
from collections import Counter

import sentencepiece as spm


def str2bool(v: str) -> bool:
    """Parse boolean-like CLI strings."""
    if isinstance(v, bool):
        return v
    v = v.lower().strip()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def train_sentencepiece(
    input_path: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    hard_vocab_limit: bool = True,
    user_defined_symbols=None,
) -> str:
    """
    Train a SentencePiece model and return the resulting model path.
    """
    if user_defined_symbols is None:
        user_defined_symbols = []

    # Compose trainer args string (SentencePieceTrainer API expects a string)
    args = [
        f"--input={input_path}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",  # unigram or bpe
        f"--character_coverage={character_coverage}",
        f"--hard_vocab_limit={'true' if hard_vocab_limit else 'false'}",
        "--bos_id=-1",  # disable BOS if you don't want it
        "--eos_id=-1",  # disable EOS if you don't want it
    ]

    # Add user defined symbols if provided
    for sym in user_defined_symbols:
        args.append(f"--user_defined_symbols={sym}")

    arg_str = " ".join(args)
    print(f"[INFO] Training SentencePiece with args:\n{arg_str}\n")

    spm.SentencePieceTrainer.train(arg_str)

    model_path = f"{model_prefix}.model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected model file not found: {model_path}")

    print(f"[INFO] Trained model saved: {model_path}")
    return model_path


def export_model_vocab_json(model_path: str, out_json_path: str) -> None:
    """
    Export full SentencePiece vocab (from model) to JSON.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    vocab = []
    for i in range(sp.get_piece_size()):
        vocab.append({
            "id": i,
            "piece": sp.id_to_piece(i),
            "score": float(sp.get_score(i)),
        })

    payload = {
        "model_path": model_path,
        "vocab_size": sp.get_piece_size(),
        "vocab": vocab,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Exported model vocab JSON: {out_json_path} ({len(vocab)} pieces)")


def export_corpus_piece_counts(model_path: str, input_path: str, out_json_path: str) -> None:
    """
    Encode the corpus and count how many times each piece id appears, then export to JSON.
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)

    cnt = Counter()
    total_lines = 0
    total_pieces = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            ids = sp.encode(line, out_type=int)
            total_pieces += len(ids)
            cnt.update(ids)

    pieces = []
    for pid, c in cnt.most_common():
        pid = int(pid)
        pieces.append({
            "id": pid,
            "piece": sp.id_to_piece(pid),
            "count": int(c),
        })

    payload = {
        "model_path": model_path,
        "input_path": input_path,
        "unique_pieces_used": len(cnt),
        "total_pieces": total_pieces,
        "total_lines": total_lines,
        "pieces": pieces,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Exported corpus piece counts JSON: {out_json_path}")
    print(f"[INFO] Lines={total_lines}, TotalPieces={total_pieces}, UniqueUsed={len(cnt)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to training text file (one sentence per line).")
    ap.add_argument("--model_prefix", default="m", help="Prefix for output model files (m.model, m.vocab).")
    ap.add_argument("--vocab_size", type=int, default=5000, help="SentencePiece vocab size.")
    ap.add_argument(
        "--model_type",
        default="unigram",
        choices=["unigram", "bpe"],
        help="SentencePiece model type."
    )
    ap.add_argument(
        "--character_coverage",
        type=float,
        default=1.0,
        help="Character coverage (use 1.0 for Vietnamese; 0.9995 common for JP/ZH)."
    )
    ap.add_argument(
        "--hard_vocab_limit",
        type=str2bool,
        default=True,
        help="If false, allow training to proceed even if requested vocab_size cannot be met (useful for small corpora). "
             "Examples: --hard_vocab_limit false"
    )
    ap.add_argument(
        "--export_corpus_counts",
        action="store_true",
        help="If set, export token usage counts on the input corpus."
    )
    ap.add_argument(
        "--out_vocab_json",
        default=None,
        help="Output JSON for model vocab. Default: <model_prefix>_vocab.json"
    )
    ap.add_argument(
        "--out_counts_json",
        default=None,
        help="Output JSON for corpus piece counts. Default: <model_prefix>_corpus_piece_counts.json"
    )
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    model_prefix = args.model_prefix
    model_path = f"{model_prefix}.model"

    # Train only if model doesn't exist (safe default)
    if not os.path.exists(model_path):
        model_path = train_sentencepiece(
            input_path=input_path,
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            hard_vocab_limit=args.hard_vocab_limit,
        )
    else:
        print(f"[INFO] Model already exists, skipping training: {model_path}")

    out_vocab_json = args.out_vocab_json or f"{model_prefix}_vocab.json"
    export_model_vocab_json(model_path, out_vocab_json)

    if args.export_corpus_counts:
        out_counts_json = args.out_counts_json or f"{model_prefix}_corpus_piece_counts.json"
        export_corpus_piece_counts(model_path, input_path, out_counts_json)


if __name__ == "__main__":
    main()
