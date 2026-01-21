#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import yaml
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

from jiwer import process_words


# -------------------------
# Logging
# -------------------------
def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


# -------------------------
# IO
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# jiwer alignment expansion
# -------------------------
def expand_chunks(word_output, sample_idx: int = 0) -> List[Tuple[Optional[str], Optional[str], str]]:
    """
    Convert jiwer WordOutput alignment chunks (AlignmentChunk) into per-token steps.
    Returns list of (ref_token|None, hyp_token|None, op) where op in:
      equal, substitute, insert, delete
    """
    ref_tokens = word_output.references[sample_idx]
    hyp_tokens = word_output.hypotheses[sample_idx]
    chunks = word_output.alignments[sample_idx]

    steps: List[Tuple[Optional[str], Optional[str], str]] = []
    for ch in chunks:
        op = ch.type
        r0, r1 = ch.ref_start_idx, ch.ref_end_idx
        h0, h1 = ch.hyp_start_idx, ch.hyp_end_idx

        if op in ("equal", "substitute"):
            L = min(r1 - r0, h1 - h0)
            for k in range(L):
                steps.append((ref_tokens[r0 + k], hyp_tokens[h0 + k], op))
            # Safety if unequal lengths (rare)
            if (r1 - r0) > L:
                for k in range(L, r1 - r0):
                    steps.append((ref_tokens[r0 + k], None, "delete"))
            if (h1 - h0) > L:
                for k in range(L, h1 - h0):
                    steps.append((None, hyp_tokens[h0 + k], "insert"))

        elif op == "delete":
            for k in range(r0, r1):
                steps.append((ref_tokens[k], None, "delete"))

        elif op == "insert":
            for k in range(h0, h1):
                steps.append((None, hyp_tokens[k], "insert"))

        else:
            raise ValueError(f"Unknown AlignmentChunk type: {op}")

    return steps


# -------------------------
# Vocab loader (optional but recommended)
# -------------------------
def load_vocab(vocab_path: str) -> set:
    """
    Supports:
    - list[str]
    - dict[token->id]
    - dict[id->token]
    """
    obj = load_json(vocab_path)

    if isinstance(obj, list):
        return set(obj)

    if isinstance(obj, dict):
        # token->id
        if all(isinstance(k, str) and isinstance(v, int) for k, v in obj.items()):
            return set(obj.keys())
        # id->token
        if all(isinstance(v, str) for v in obj.values()):
            return set(obj.values())

    raise ValueError(f"Unsupported vocab format: {vocab_path}")


# -------------------------
# Core metric
# -------------------------
def compute_oov_fill_accuracy_using_w2i_gold(
    test_word_path: str,
    vocab_word_path: str,
    result_test_word_path: str,   # w2i json; we will use its "gold" field (contains <unk>)
    result_fill_path: str,        # c2i/p2i/w2i json; use "predicted"
    unk_token: str = "<unk>",
    save: bool = False,
    save_path: Optional[str] = None,
    type_fill: str = "c2i",
    max_examples: int = 200,
) -> Dict[str, Any]:
    """
    Targets are reference tokens aligned to <unk> in w2i.gold.
    True-OOV filtering uses vocab_train (token not in vocab).
    Correct if fill prediction recovers exact token in ref-vs-fill alignment.
    """

    test_ref = load_json(test_word_path)            # list of {"text": ...}
    w2i = load_json(result_test_word_path)          # list of {"gold": "... <unk> ...", "predicted": ...}
    fill = load_json(result_fill_path)              # list of {"predicted": ...}
    vocab = load_vocab(vocab_word_path)

    n = min(len(test_ref), len(w2i), len(fill))
    if n == 0:
        raise ValueError("Empty inputs or mismatched lengths.")

    total_targets = 0
    correct = 0
    unique_correct = set()
    unique_missed = set()

    examples = []

    for i in range(n):
        ref = (test_ref[i].get("text") or "").strip()
        gold_unked = (w2i[i].get("gold") or "").strip()      # IMPORTANT: use gold to locate <unk>
        hyp_fill = (fill[i].get("predicted") or "").strip()

        if not ref or not gold_unked:
            continue

        # 1) Align ref vs w2i.gold (unked) => identify which ref tokens became <unk>
        out_rg = process_words(ref, gold_unked)
        print(out_rg)
        steps_rg = expand_chunks(out_rg, 0)

        targets: List[str] = []
        for r, h, _ in steps_rg:
            if r is None:
                continue
            if h == unk_token and (r not in vocab):
                targets.append(r)

        if not targets:
            continue

        # 2) Align ref vs fill.predicted => check recovery (exact match)
        out_rf = process_words(ref, hyp_fill)
        steps_rf = expand_chunks(out_rf, 0)

        recovered: List[str] = []
        for r, h, _ in steps_rf:
            if r is None:
                continue
            if (r not in vocab) and (r == h):
                recovered.append(r)

        # multiset match for duplicates
        bag: Dict[str, int] = {}
        for t in recovered:
            bag[t] = bag.get(t, 0) + 1

        local_missed = []
        for t in targets:
            total_targets += 1
            if bag.get(t, 0) > 0:
                correct += 1
                bag[t] -= 1
                unique_correct.add(t)
            else:
                unique_missed.add(t)
                local_missed.append(t)

        if local_missed and len(examples) < max_examples:
            examples.append({
                "idx": i,
                "targets": targets,
                "missed": local_missed,
                "ref": ref,
                "w2i_gold_unked": gold_unked,
                "fill_pred": hyp_fill,
            })

    acc = (correct / total_targets) if total_targets else 0.0

    summary = {
        "type_fill": type_fill,
        "unk_token": unk_token,
        "total_oov_targets": total_targets,
        "correct_recovered": correct,
        "accuracy": acc,
        "num_unique_correct": len(unique_correct),
        "num_unique_missed": len(unique_missed),
        "unique_correct_tokens": sorted(unique_correct),
        "unique_missed_tokens": sorted(unique_missed),
    }

    logging.info("=== OOV Fill Accuracy (targets from w2i.gold, alignment-based) ===")
    logging.info(f"type_fill: {type_fill}")
    logging.info(f"total_oov_targets: {total_targets}")
    logging.info(f"correct_recovered: {correct}")
    logging.info(f"accuracy: {acc*100:.3f}%")
    logging.info(f"unique_correct: {len(unique_correct)} | unique_missed: {len(unique_missed)}")

    if save and save_path:
        out_dir = os.path.join(save_path, "oov_fill", type_fill)
        os.makedirs(out_dir, exist_ok=True)
        save_json(os.path.join(out_dir, "summary.json"), summary)
        save_json(os.path.join(out_dir, "examples_missed.json"), examples)

    return summary


def main():
    p = argparse.ArgumentParser("Compute OOV-fill accuracy with config + model_name + type_fill")
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--model_name", required=True, help="e.g. SpeechTrans, TASA, ...")
    p.add_argument("--type_fill", required=True, choices=["c2i", "p2i", "w2i"], help="Evaluate result_{type_fill}_path")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.model_name not in cfg:
        raise KeyError(f"model_name '{args.model_name}' not found in config.")

    m = cfg[args.model_name]

    test_word_path = m["test_word_path"]
    vocab_word_path = m["vocab_word_path"]

    # Use result_test_word_path (same as w2i json) because it contains "gold" with <unk>
    result_test_word_path = m["result_test_word_path"]

    fill_key = f"result_{args.type_fill}_path"
    if fill_key not in m:
        raise KeyError(f"Missing '{fill_key}' in config for model {args.model_name}.")
    result_fill_path = m[fill_key]

    log_dir = m.get("log_path", m.get("save_path", "."))
    log_file = os.path.join(log_dir, f"oov_fill_{args.type_fill}.log")
    setup_logging(log_file)

    compute_oov_fill_accuracy_using_w2i_gold(
        test_word_path=test_word_path,
        vocab_word_path=vocab_word_path,
        result_test_word_path=result_test_word_path,
        result_fill_path=result_fill_path,
        unk_token="<unk>",
        save=bool(m.get("save", False)),
        save_path=m.get("save_path", None),
        type_fill=args.type_fill,
    )


if __name__ == "__main__":
    main()
