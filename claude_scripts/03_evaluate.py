"""
Evaluate the fine-tuned gender annotation model on the test split.

Prints overall precision / recall / F1 and per-entity-type breakdown.

Usage:
    python 03_evaluate.py data/ model/
    python 03_evaluate.py data/ model/ --batch-size 64
"""

import os
import json
import argparse

import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def predict_batch(model, tokenizer, batch_tokens, device):
    """Return (predictions ndarray, list-of-word_ids) for a batch of word lists."""
    encoding = tokenizer(
        batch_tokens,
        truncation=True,
        is_split_into_words=True,
        max_length=128,
        padding=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits

    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    word_ids_list = [encoding.word_ids(i) for i in range(len(batch_tokens))]
    return predictions, word_ids_list


def align_predictions_to_words(preds, word_ids, n_words):
    """Map subword predictions back to word-level using first-subword rule."""
    word_preds = {}
    for pos, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_preds:
            word_preds[word_id] = ID2LABEL[preds[pos]]
    return [word_preds.get(i, 'O') for i in range(n_words)]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the gender annotation model on the test set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('data_dir',  help='Directory with test.jsonl')
    parser.add_argument('model_dir', help='Directory with fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    test_path = os.path.join(args.data_dir, 'test.jsonl')
    print(f"Loading {test_path} ...")
    test_examples = load_jsonl(test_path)
    print(f"Test set: {len(test_examples):,} examples")

    seqeval = evaluate.load('seqeval')
    all_true, all_pred = [], []

    for i in range(0, len(test_examples), args.batch_size):
        batch = test_examples[i:i + args.batch_size]
        batch_tokens = [ex['tokens'] for ex in batch]
        batch_labels = [ex['labels'] for ex in batch]

        preds, word_ids_list = predict_batch(model, tokenizer, batch_tokens, device)

        for pred_seq, word_ids, true_labels in zip(preds, word_ids_list, batch_labels):
            pred_labels = align_predictions_to_words(pred_seq, word_ids, len(true_labels))
            all_true.append(true_labels)
            all_pred.append(pred_labels)

        processed = min(i + args.batch_size, len(test_examples))
        if processed % 1000 < args.batch_size:
            print(f"  {processed:,}/{len(test_examples):,} processed ...")

    results = seqeval.compute(predictions=all_pred, references=all_true)

    print("\n=== Evaluation Results ===")
    print(f"Precision : {results['overall_precision']:.4f}")
    print(f"Recall    : {results['overall_recall']:.4f}")
    print(f"F1        : {results['overall_f1']:.4f}")
    print(f"Accuracy  : {results['overall_accuracy']:.4f}")

    # Per-entity breakdown (seqeval keys the entity type without B-/I- prefix)
    for key in sorted(results):
        if key.startswith('overall') or not isinstance(results[key], dict):
            continue
        print(f"\n  [{key}]")
        for metric, value in results[key].items():
            print(f"    {metric:12s}: {value:.4f}")


if __name__ == '__main__':
    main()
