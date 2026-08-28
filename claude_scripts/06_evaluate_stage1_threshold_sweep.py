"""
Stage 1 (masculine-span detection, SPECIFICATION.md §3.2) evaluation:
precision at fixed recall, per §3.3's actual model-selection criterion
("Pick the winner on precision at fixed recall ... not raw F1 -- for an
autocorrect feature, false positives ... are more costly ... than false
negatives").

03_evaluate.py reports a single argmax-threshold operating point (like
seqeval always does). This script instead sweeps the BERT token
classifier's decision threshold on P(not-O) to trace out a precision/recall
curve, and reports precision at several fixed recall levels for direct
comparison against a second architecture (e.g. the OpenNMT-py greedy decoder
in opennmt_run_stage1/, which has no equivalent confidence knob and is
therefore reported as a single point on this same axis).

This script only makes sense for Stage 1 data specifically, because Stage 1
spans are, structurally, always exactly one token (every degender_match()
substitution in 01b_extract_masculine_spans.py is a single word with no
internal whitespace -- unlike Approach A/B's data, where a "paired"-pattern
gold span like "Lehrer und Lehrerinnen" covers 3 tokens including "und").
That means span-level and token-level precision/recall coincide exactly
here, so a simple per-token threshold sweep is a valid proxy for a span-level
PR curve -- this assumption is checked at runtime and the script aborts if
it doesn't hold.

Usage:
    python 06_evaluate_stage1_threshold_sweep.py data_dir/ model_dir/ [--out results.json]
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
O_ID = LABEL_LIST.index('O')


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def check_single_token_spans(examples):
    """Verify every gold positive span in `examples` is exactly one token
    (B-GENDER never followed by I-GENDER). Returns (ok, violation_count)."""
    violations = 0
    for ex in examples:
        labels = ex['labels']
        for i, lbl in enumerate(labels):
            if lbl == 'I-GENDER' and (i == 0 or labels[i - 1] not in ('B-GENDER', 'I-GENDER')):
                violations += 1
            elif lbl == 'B-GENDER' and i + 1 < len(labels) and labels[i + 1] == 'I-GENDER':
                violations += 1
    return violations == 0, violations


def predict_probs(model, tokenizer, batch_tokens, device):
    """Return (probs ndarray [batch, seq, 3], word_ids_list) for a batch."""
    encoding = tokenizer(
        batch_tokens, truncation=True, is_split_into_words=True,
        max_length=128, padding=True, return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        logits = model(**encoding).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    word_ids_list = [encoding.word_ids(i) for i in range(len(batch_tokens))]
    return probs, word_ids_list


def word_level_not_o_prob(probs_row, word_ids, n_words):
    """First-subword rule: P(not-O) per word, from per-subword-token probs."""
    word_prob = {}
    for pos, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_prob:
            word_prob[word_id] = 1.0 - probs_row[pos, O_ID]
    return [word_prob.get(i, 0.0) for i in range(n_words)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir')
    parser.add_argument('model_dir')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out', help='Write full results JSON here')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_examples = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))
    print(f"Test set: {len(test_examples):,} examples")

    ok, violations = check_single_token_spans(test_examples)
    print(f"Single-token-span assumption: {'HOLDS' if ok else 'VIOLATED'} "
          f"({violations} multi-token spans)")
    if not ok:
        print("ERROR: this script assumes every gold span is one token "
              "(true for Stage 1 data). Aborting rather than silently "
              "reporting a wrong metric.", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    all_scores = []  # per-token P(not-O)
    all_gold = []    # per-token 1 if B-GENDER else 0
    n_pos_tokens = 0

    for i in range(0, len(test_examples), args.batch_size):
        batch = test_examples[i:i + args.batch_size]
        batch_tokens = [ex['tokens'] for ex in batch]
        probs, word_ids_list = predict_probs(model, tokenizer, batch_tokens, device)

        for row_idx, (ex, word_ids) in enumerate(zip(batch, word_ids_list)):
            n_words = len(ex['labels'])
            scores = word_level_not_o_prob(probs[row_idx], word_ids, n_words)
            gold = [1 if l == 'B-GENDER' else 0 for l in ex['labels']]
            all_scores.extend(scores)
            all_gold.extend(gold)
            n_pos_tokens += sum(gold)

        done = min(i + args.batch_size, len(test_examples))
        if done % 1000 < args.batch_size:
            print(f"  {done:,}/{len(test_examples):,} ...")

    all_scores = np.array(all_scores)
    all_gold = np.array(all_gold)
    print(f"Total tokens: {len(all_gold):,}  positive tokens: {n_pos_tokens:,}")

    # ---- sweep thresholds, trace precision/recall ----
    thresholds = sorted(set(np.round(np.arange(0.01, 1.00, 0.01), 2)))
    curve = []
    for t in thresholds:
        pred_pos = all_scores >= t
        tp = int(np.sum(pred_pos & (all_gold == 1)))
        fp = int(np.sum(pred_pos & (all_gold == 0)))
        fn = int(np.sum(~pred_pos & (all_gold == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append({'threshold': t, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn})

    # ---- precision at fixed recall targets (nearest achieved recall >= target) ----
    targets = [0.95, 0.90, 0.85, 0.80, 0.70]
    precision_at_recall = {}
    for target in targets:
        candidates = [c for c in curve if c['recall'] >= target]
        if candidates:
            # among thresholds achieving at least `target` recall, take the
            # highest-precision one (== lowest threshold that still clears it)
            best = max(candidates, key=lambda c: c['precision'])
            precision_at_recall[target] = {
                'precision': best['precision'], 'recall': best['recall'], 'threshold': best['threshold'],
            }
        else:
            precision_at_recall[target] = None

    # ---- also report the standard argmax (threshold=0.5) operating point ----
    default_op = min(curve, key=lambda c: abs(c['threshold'] - 0.5))

    results = {
        'n_test_examples': len(test_examples),
        'n_tokens': int(len(all_gold)),
        'n_positive_tokens': int(n_pos_tokens),
        'default_threshold_0.5_operating_point': default_op,
        'precision_at_fixed_recall': precision_at_recall,
        'full_curve': curve,
    }

    print("\n=== Precision at fixed recall ===")
    for target in targets:
        r = precision_at_recall[target]
        if r:
            print(f"  recall >= {target:.2f}: precision = {r['precision']:.4f} "
                  f"(actual recall {r['recall']:.4f}, threshold {r['threshold']:.2f})")
        else:
            print(f"  recall >= {target:.2f}: UNREACHABLE")
    print(f"\n  default (threshold=0.5): precision={default_op['precision']:.4f} "
          f"recall={default_op['recall']:.4f}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
