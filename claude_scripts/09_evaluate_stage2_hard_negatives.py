"""
The targeted "did hard-negative mining actually work" comparison for Stage 2
(SPECIFICATION.md §3.2 / §8 milestone 3).

Overall F1 on a merged test set (from 03_evaluate.py) can't answer this on
its own -- it's diluted by the much larger Stage 1 portion of the data. This
script instead evaluates BOTH the Stage 1 model and the retrained Stage 2
model on exactly the held-out hard-negative examples
(07b_mine_stage2_hard_negatives.py's test split, source == 'stage2_hard_neg'
only -- i.e. real sentences where Stage 1 is *known* to have been wrong, by
construction) and reports each model's false-positive rate on that specific,
adversarial set. Success = Stage 2's rate is meaningfully lower than Stage
1's, without recall on ordinary positives collapsing (check that separately
via 03_evaluate.py on the full merged test set).

Usage:
    python 09_evaluate_stage2_hard_negatives.py \\
        --hard-neg-test ../claude_pipeline_output_stage2/hard_mined/test.jsonl \\
        --stage1-model ../claude_pipeline_output_stage1/model \\
        --stage2-model ../claude_pipeline_output_stage2/model \\
        --out ../claude_pipeline_output_stage2/hard_negative_before_after.json
"""

import os
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def predict_any_flag(model, tokenizer, batch_words, device):
    """Return, per example, True if the model flags ANY non-O token."""
    encoding = tokenizer(
        batch_words, truncation=True, is_split_into_words=True,
        max_length=128, padding=True, return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        logits = model(**encoding).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    flags = []
    for row_idx in range(len(batch_words)):
        word_ids = encoding.word_ids(row_idx)
        flagged = False
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                seen.add(wid)
                if ID2LABEL[preds[row_idx][pos]] != 'O':
                    flagged = True
        flags.append(flagged)
    return flags


def evaluate_model(model_dir, examples, device, batch_size, label):
    print(f"\nLoading {label} model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    n_still_flagged = 0
    examples_still_wrong = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        batch_words = [ex['tokens'] for ex in batch]
        flags = predict_any_flag(model, tokenizer, batch_words, device)
        for ex, flagged in zip(batch, flags):
            if flagged:
                n_still_flagged += 1
                if len(examples_still_wrong) < 10:
                    examples_still_wrong.append({
                        'sentence': ' '.join(ex['tokens']),
                        'originally_flagged_word': ex.get('stage1_flagged_word'),
                    })

    rate = n_still_flagged / len(examples) if examples else 0.0
    print(f"  {label}: still flags {n_still_flagged:,}/{len(examples):,} "
          f"({rate:.1%}) of known Stage 1 false positives.")
    del model
    torch.cuda.empty_cache()
    return {
        'n_examples': len(examples),
        'n_still_flagged': n_still_flagged,
        'false_positive_rate_on_hard_negatives': rate,
        'sample_still_wrong': examples_still_wrong,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hard-neg-test', required=True)
    parser.add_argument('--stage1-model', required=True)
    parser.add_argument('--stage2-model', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_examples = load_jsonl(args.hard_neg_test)
    hard_neg_examples = [ex for ex in all_examples if ex.get('source') == 'stage2_hard_neg']
    print(f"Held-out hard-negative test examples: {len(hard_neg_examples):,} "
          f"(of {len(all_examples):,} total in this split)")

    if not hard_neg_examples:
        print("ERROR: no source=='stage2_hard_neg' examples found in this file.")
        return

    results = {
        'n_hard_negative_test_examples': len(hard_neg_examples),
        'stage1_model': evaluate_model(args.stage1_model, hard_neg_examples, device, args.batch_size, 'Stage 1'),
        'stage2_model': evaluate_model(args.stage2_model, hard_neg_examples, device, args.batch_size, 'Stage 2'),
    }

    r1 = results['stage1_model']['false_positive_rate_on_hard_negatives']
    r2 = results['stage2_model']['false_positive_rate_on_hard_negatives']
    print(f"\n=== Summary ===")
    print(f"  Stage 1 false-positive rate on these known-hard cases: {r1:.1%}")
    print(f"  Stage 2 false-positive rate on these known-hard cases: {r2:.1%}")
    if r1 > 0:
        print(f"  Relative reduction: {(r1 - r2) / r1:.1%}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
