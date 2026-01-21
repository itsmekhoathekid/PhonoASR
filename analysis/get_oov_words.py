#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter
import sentencepiece as spm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_word(w: str) -> str:
    # Giữ bảo thủ: không lower/punct-strip để tránh làm sai semantics.
    return w.strip()


def build_train_vocab(train_data, text_key="text"):
    vocab = set()
    for item in train_data:
        if not isinstance(item, dict) or text_key not in item:
            continue
        for w in str(item[text_key]).split():
            w = normalize_word(w)
            if w:
                vocab.add(w)
    return vocab


def extract_oov_words(train_vocab, test_data, text_key="text"):
    oov = set()
    for item in test_data:
        if not isinstance(item, dict) or text_key not in item:
            continue
        for w in str(item[text_key]).split():
            w = normalize_word(w)
            if w and w not in train_vocab:
                oov.add(w)
    return sorted(oov)


def load_sp(model_path: str) -> spm.SentencePieceProcessor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return spm.SentencePieceProcessor(model_file=model_path)


def seg(sp: spm.SentencePieceProcessor, word: str):
    pieces = sp.encode(word, out_type=str)
    ids = sp.encode(word, out_type=int)
    return pieces, ids


def main():
    ap = argparse.ArgumentParser(description="Compare BPE vs Unigram SentencePiece segmentation for OOV words.")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--bpe_model", required=True, help="Path to SentencePiece BPE .model")
    ap.add_argument("--uni_model", required=True, help="Path to SentencePiece Unigram .model")
    ap.add_argument("--out_json", required=True, help="Output JSON with side-by-side segmentations")
    ap.add_argument("--print_top", type=int, default=50, help="Print top N examples to console")
    args = ap.parse_args()

    train = load_json(args.train_json)
    test = load_json(args.test_json)

    train_vocab = build_train_vocab(train, text_key=args.text_key)
    oov_words = extract_oov_words(train_vocab, test, text_key=args.text_key)

    sp_bpe = load_sp(args.bpe_model)
    sp_uni = load_sp(args.uni_model)

    unk_bpe = sp_bpe.unk_id()
    unk_uni = sp_uni.unk_id()

    rows = []
    diff_count = 0
    both_unk = 0
    bpe_unk_only = 0
    uni_unk_only = 0

    # để xem xu hướng: số mảnh subword
    bpe_piece_lens = Counter()
    uni_piece_lens = Counter()

    for w in oov_words:
        bpe_pieces, bpe_ids = seg(sp_bpe, w)
        uni_pieces, uni_ids = seg(sp_uni, w)

        bpe_has_unk = unk_bpe in bpe_ids
        uni_has_unk = unk_uni in uni_ids

        if bpe_has_unk and uni_has_unk:
            both_unk += 1
        elif bpe_has_unk and not uni_has_unk:
            bpe_unk_only += 1
        elif (not bpe_has_unk) and uni_has_unk:
            uni_unk_only += 1

        bpe_piece_lens[len(bpe_pieces)] += 1
        uni_piece_lens[len(uni_pieces)] += 1

        same = (bpe_pieces == uni_pieces)
        if not same:
            diff_count += 1

        rows.append({
            "word": w,
            "bpe": {
                "pieces": bpe_pieces,
                "ids": bpe_ids,
                "n_pieces": len(bpe_pieces),
                "has_unk": bpe_has_unk
            },
            "unigram": {
                "pieces": uni_pieces,
                "ids": uni_ids,
                "n_pieces": len(uni_pieces),
                "has_unk": uni_has_unk
            },
            "same_segmentation": same
        })

    summary = {
        "num_oov_words": len(oov_words),
        "num_diff_segmentation": diff_count,
        "ratio_diff_segmentation": (diff_count / len(oov_words)) if oov_words else 0.0,
        "unk_stats": {
            "both_unk": both_unk,
            "bpe_unk_only": bpe_unk_only,
            "unigram_unk_only": uni_unk_only
        },
        "bpe_piece_count_distribution": dict(bpe_piece_lens),
        "unigram_piece_count_distribution": dict(uni_piece_lens),
        "bpe_model": args.bpe_model,
        "unigram_model": args.uni_model
    }

    save_json({"summary": summary, "oov_examples": rows}, args.out_json)

    # Print a few examples to console (prioritize different segmentations)
    print(f"[INFO] OOV words: {len(oov_words)}")
    print(f"[INFO] Different segmentations: {diff_count} ({summary['ratio_diff_segmentation']:.2%})")
    print(f"[INFO] Saved: {args.out_json}\n")

    # show top examples where segmentation differs
    shown = 0
    for r in rows:
        if not r["same_segmentation"]:
            print(f"WORD: {r['word']}")
            print(f"  BPE     ({r['bpe']['n_pieces']}): {' | '.join(r['bpe']['pieces'])}   (UNK={r['bpe']['has_unk']})")
            print(f"  UNIGRAM ({r['unigram']['n_pieces']}): {' | '.join(r['unigram']['pieces'])}   (UNK={r['unigram']['has_unk']})")
            print("-" * 80)
            shown += 1
            if shown >= args.print_top:
                break

    if shown == 0:
        print("[INFO] No differing segmentations found in the printed range.")


if __name__ == "__main__":
    main()
