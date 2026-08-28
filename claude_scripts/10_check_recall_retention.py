"""
Check whether Stage 2's precision gain (09_evaluate_stage2_hard_negatives.py)
came at a recall cost on ORIGINAL Stage 1 positive examples -- the standard
precision/recall tradeoff sanity check hard-negative mining always needs,
since it's easy to "fix" false positives by just becoming more conservative
across the board rather than specifically on the hard cases.

Isolates test examples tagged source=='stage1' with a real B-GENDER label
(i.e. Stage 1's own original positive examples, not Stage 2's mined bonus
positives or hard negatives) from the merged Stage 2 test set, and compares
the Stage 1 and Stage 2 models' recall on exactly that fixed positive set.

Usage:
    python 10_check_recall_retention.py \\
        --test-set ../claude_pipeline_output_stage2/data_merged/test.jsonl \\
        --stage1-model ../claude_pipeline_output_stage1/model \\
        --stage2-model ../claude_pipeline_output_stage2/model \\
        --out ../claude_pipeline_output_stage2/original_positive_recall_retention.json
"""

import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def recall_on(model_dir, examples, device, batch_size, label):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    hit = 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        words = [ex['tokens'] for ex in batch]
        encoding = tokenizer(words, truncation=True, is_split_into_words=True,
                              max_length=128, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            preds = torch.argmax(model(**encoding).logits, dim=-1).cpu().tolist()
        for row_idx, ex in enumerate(batch):
            gold_idx = ex['labels'].index('B-GENDER')
            word_ids = encoding.word_ids(row_idx)
            pred_label = 'O'
            for pos, wid in enumerate(word_ids):
                if wid == gold_idx:
                    pred_label = ID2LABEL[preds[row_idx][pos]]
                    break
            if pred_label in ('B-GENDER', 'I-GENDER'):
                hit += 1

    r = hit / len(examples) if examples else 0.0
    print(f"  {label}: recall on original Stage 1 positives = {hit:,}/{len(examples):,} ({r:.1%})")
    del model
    torch.cuda.empty_cache()
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--stage1-model', required=True)
    parser.add_argument('--stage2-model', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out')
    args = parser.parse_args()

    examples = load_jsonl(args.test_set)
    orig_pos = [ex for ex in examples if ex.get('source') == 'stage1' and 'B-GENDER' in ex['labels']]
    print(f"Original Stage-1-sourced positive examples in this test set: {len(orig_pos):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r1 = recall_on(args.stage1_model, orig_pos, device, args.batch_size, 'Stage 1 model')
    r2 = recall_on(args.stage2_model, orig_pos, device, args.batch_size, 'Stage 2 model')
    print(f"\nRecall change: {r1:.1%} -> {r2:.1%} ({(r2 - r1) * 100:+.1f} points)")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'n_examples': len(orig_pos), 'stage1_recall': r1, 'stage2_recall': r2}, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
