#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert ASR result .txt (Predict text / Ground truth blocks) to JSON
and compute WER/CER using jiwer.

Input format (per block):
  Predict text: ...
  Ground truth: ...
  ---------------

Output JSON:
  [
    {
      "gold": "...",
      "predicted": "...",
      "WER": 0.123,
      "CER": 0.045
    },
    ...
  ]

Usage:
  python convert_txt_to_json_with_jiwer.py \
    --txt /path/to/result.txt \
    --out /path/to/output.json

Notes:
- WER computed with jiwer.wer
- CER computed with jiwer.cer
- Optional: use --no_normalize to disable basic whitespace normalization
"""

from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Tuple, Optional

from jiwer import wer as jiwer_wer
from jiwer import cer as jiwer_cer


SEP_LINE_RE = re.compile(r"^\s*-{5,}\s*$")
PRED_RE = re.compile(r"^\s*Predict text:\s*(.*)\s*$")
GOLD_RE = re.compile(r"^\s*Ground truth:\s*(.*)\s*$")


def normalize_spaces(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_txt(path: str) -> List[Tuple[str, str]]:
    """
    Parse blocks from file into list of (gold, predicted).

    Assumption: Predict text and Ground truth are each on a single line,
    which matches your uploaded txt logs. If you later have multi-line
    utterances, tell me and I’ll adapt the parser to continuation lines.
    """
    pairs: List[Tuple[str, str]] = []
    current_pred: Optional[str] = None
    current_gold: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if SEP_LINE_RE.match(line):
                if current_pred is not None and current_gold is not None:
                    pairs.append((current_gold, current_pred))
                current_pred, current_gold = None, None
                continue

            m_pred = PRED_RE.match(line)
            if m_pred:
                current_pred = m_pred.group(1).strip()
                continue

            m_gold = GOLD_RE.match(line)
            if m_gold:
                current_gold = m_gold.group(1).strip()
                continue

    # flush at EOF
    if current_pred is not None and current_gold is not None:
        pairs.append((current_gold, current_pred))

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True, help="Input .txt ASR result file")
    ap.add_argument("--out", required=True, help="Output .json file path")
    ap.add_argument("--indent", type=int, default=4, help="JSON indent")
    ap.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable whitespace normalization (strip + collapse spaces).",
    )
    args = ap.parse_args()

    pairs = parse_txt(args.txt)
    if not pairs:
        raise SystemExit("No (gold, predicted) pairs parsed. Check input format.")

    items: List[Dict] = []
    sum_wer = 0.0
    sum_cer = 0.0

    for gold, pred in pairs:
        if not args.no_normalize:
            gold_eval = normalize_spaces(gold)
            pred_eval = normalize_spaces(pred)
        else:
            gold_eval = gold
            pred_eval = pred

        w = float(jiwer_wer(gold_eval, pred_eval))
        c = float(jiwer_cer(gold_eval, pred_eval))

        items.append(
            {
                "gold": gold,          # keep original text
                "predicted": pred,     # keep original text
                "WER": w,
                "CER": c,
            }
        )
        sum_wer += w
        sum_cer += c

    avg_wer = sum_wer / len(items)
    avg_cer = sum_cer / len(items)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=args.indent)

    print(f"Parsed utterances: {len(items)}")
    print(f"Average WER: {avg_wer:.6f}")
    print(f"Average CER: {avg_cer:.6f}")
    print(f"Wrote JSON: {args.out}")


if __name__ == "__main__":
    main()
