"""
Mine hard negatives (and bonus positives) for Stage 2 -- SPECIFICATION.md
§3.2's "should this occurrence be gendered?" decision, §8 milestone 3.

Rationale (SPECIFICATION.md §10.3): Stage 1's own real false positives
(Schutzzölle, Devot, OSZE-Ministerratskonferenz, ...) are a better source of
hard negatives than a hand-designed proxy, because they are literally the
decision boundary Stage 2 needs to learn -- surface-plausible enough to fool
a 98%-F1 model is exactly what "hard" means. This script runs the trained
Stage 1 model over corpus text it never saw and checks every prediction
against a corpus-wide ground truth (07a_build_confirmed_stems.py's output):

    predicted stem IS  in the confirmed set -> bonus positive
        (a real candidate outside Stage 1's 10k-per-source training cap)
    predicted stem NOT in the confirmed set -> hard negative
        (Stage 1 was wrong to flag it -- keep the sentence with all-O gold
        labels so retraining teaches the model not to)

No-leakage guarantee: this reproduces 01b_extract_masculine_spans.py's
collect_articles() with IDENTICAL parameters to run_stage1_pipeline.sh's
actual invocation (--max-positive 10000 --neg-ratio 1.0 --seed 42), to get
the exact set of article paths Stage 1's training data came from, and skips
every one of them. This script imports 01b_extract_masculine_spans.py as a
module specifically to replicate that logic exactly rather than risk drift
from a re-copied version -- a deliberate, documented exception to this
project's usual "self-contained script" convention, justified because this
script's whole job is to reproduce another script's exact behavior.

Usage:
    python 07b_mine_stage2_hard_negatives.py \\
        --stage1-model ../claude_pipeline_output_stage1/model \\
        --confirmed-stems ../claude_pipeline_output_stage2/confirmed_stems.json \\
        --output-dir ../claude_pipeline_output_stage2/hard_mined \\
        --target-hard-negatives 2000
"""

import os
import sys
import json
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "extract_masculine_spans",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "01b_extract_masculine_spans.py"),
)
ems = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ems)

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def stage1_consumed_paths(corpus_dir, max_positive, neg_ratio, seed, all_files):
    """Reproduce 01b's collect_articles() with run_stage1_pipeline.sh's exact
    params and return the set of article paths it consumed."""
    random.seed(seed)
    max_neg = int(max_positive * neg_ratio)
    articles, total_pos, total_neg = ems.collect_articles(
        corpus_dir, max_positive, max_neg, all_files=all_files
    )
    return {a['path'] for a in articles}, total_pos, total_neg


def predict_spans(model, tokenizer, batch_word_lists, device):
    """Return, for each example, a list of (word_index, predicted_label)
    where predicted_label != 'O' (first-subword rule, same as
    04_annotate.py / 06_evaluate_stage1_threshold_sweep.py)."""
    if not batch_word_lists:
        return []
    encoding = tokenizer(
        batch_word_lists, truncation=True, is_split_into_words=True,
        max_length=128, padding=True, return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        logits = model(**encoding).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    results = []
    for row_idx, words in enumerate(batch_word_lists):
        word_ids = encoding.word_ids(row_idx)
        word_label = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in word_label:
                word_label[wid] = ID2LABEL[preds[row_idx][pos]]
        flagged = [(i, word_label.get(i, 'O')) for i in range(len(words))
                   if word_label.get(i, 'O') != 'O']
        results.append(flagged)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpora', nargs='+',
                        default=['corpora/taz.de', 'corpora/www.woz.ch'])
    parser.add_argument('--stage1-model', required=True)
    parser.add_argument('--confirmed-stems', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-hard-negatives', type=int, default=2000)
    parser.add_argument('--true-neg-fraction', type=float, default=0.2,
                        help='Also keep this fraction (relative to hard negatives) of '
                             'plain true-negative sentences (model predicted nothing)')
    parser.add_argument('--max-sentences', type=int, default=1_000_000,
                        help='Safety cap on fresh sentences scanned')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-positive', type=int, default=10000,
                        help='Must match run_stage1_pipeline.sh\'s value used to produce --stage1-model')
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--progress-every', type=int, default=2000)
    args = parser.parse_args()

    with open(args.confirmed_stems, 'r', encoding='utf-8') as f:
        confirmed_stems = set(json.load(f))
    print(f"Loaded {len(confirmed_stems):,} confirmed stems.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Stage 1 model from {args.stage1_model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.stage1_model)
    model = AutoModelForTokenClassification.from_pretrained(args.stage1_model).to(device)
    model.eval()

    articles_out = []  # list of {'path', 'hard_neg': [...], 'bonus_pos': [...], 'true_neg': [...]}
    n_hard_neg = 0
    n_bonus_pos = 0
    n_true_neg = 0
    n_sentences_scanned = 0

    for corpus_dir in args.corpora:
        source_name = os.path.basename(os.path.normpath(corpus_dir))
        sample_files = []
        for dirpath, _, filenames in os.walk(corpus_dir):
            for fn in filenames:
                sample_files.append(fn)
            if len(sample_files) >= 20:
                break
        all_files = not any(fn.lower().endswith('.html') for fn in sample_files[:20])

        print(f"\n=== {source_name} (all_files={all_files}) ===")
        print("Reproducing Stage 1's consumed-article set (no-leakage exclusion) ...")
        consumed, stage1_pos, stage1_neg = stage1_consumed_paths(
            corpus_dir, args.max_positive, args.neg_ratio, args.seed, all_files
        )
        print(f"  Stage 1 consumed {len(consumed):,} articles from this source "
              f"({stage1_pos:,} positive / {stage1_neg:,} negative examples).")

        random.seed(args.seed)  # re-seed for this source's fresh-sentence sampling determinism

        for dirpath, _, filenames in os.walk(corpus_dir):
            if n_hard_neg >= args.target_hard_negatives or n_sentences_scanned >= args.max_sentences:
                break
            for filename in filenames:
                if n_hard_neg >= args.target_hard_negatives or n_sentences_scanned >= args.max_sentences:
                    break
                path = os.path.join(dirpath, filename)
                if path in consumed:
                    continue
                if not all_files and not filename.lower().endswith('.html'):
                    continue
                if all_files and not filename.lower().endswith('.html') and not ems.is_html(path):
                    continue

                _, sentences = ems.extract_article_sentences(path)
                if not sentences:
                    continue

                art_hard_neg, art_bonus_pos, art_true_neg = [], [], []
                for i in range(0, len(sentences), args.batch_size):
                    batch_sents = sentences[i:i + args.batch_size]
                    batch_words = [s.split() for s in batch_sents]
                    flagged_batch = predict_spans(model, tokenizer, batch_words, device)
                    n_sentences_scanned += len(batch_sents)

                    for words, flagged in zip(batch_words, flagged_batch):
                        if not flagged:
                            if random.random() < args.true_neg_fraction:
                                art_true_neg.append({'tokens': words, 'labels': ['O'] * len(words), 'source': 'stage2_true_neg'})
                            continue
                        # Single-token-span assumption (verified for Stage 1 data in
                        # 06_evaluate_stage1_threshold_sweep.py) -- take the first
                        # flagged word as the candidate.
                        idx = flagged[0][0]
                        stem = words[idx]
                        labels = ['O'] * len(words)
                        if stem in confirmed_stems:
                            labels[idx] = 'B-GENDER'
                            art_bonus_pos.append({'tokens': words, 'labels': labels, 'source': 'stage2_bonus_pos'})
                        else:
                            art_hard_neg.append({'tokens': words, 'labels': ['O'] * len(words), 'source': 'stage2_hard_neg', 'stage1_flagged_word': stem})

                if art_hard_neg or art_bonus_pos or art_true_neg:
                    articles_out.append({
                        'path': path,
                        'hard_neg': art_hard_neg,
                        'bonus_pos': art_bonus_pos,
                        'true_neg': art_true_neg,
                    })
                    n_hard_neg += len(art_hard_neg)
                    n_bonus_pos += len(art_bonus_pos)
                    n_true_neg += len(art_true_neg)

                if n_sentences_scanned % args.progress_every < args.batch_size:
                    print(f"  scanned {n_sentences_scanned:,} sentences | "
                          f"hard_neg={n_hard_neg:,} bonus_pos={n_bonus_pos:,} true_neg={n_true_neg:,}")

    print(f"\nDone mining. Totals: hard_neg={n_hard_neg:,} bonus_pos={n_bonus_pos:,} "
          f"true_neg={n_true_neg:,} across {len(articles_out):,} articles "
          f"({n_sentences_scanned:,} sentences scanned).")

    random.shuffle(articles_out)
    n = len(articles_out)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    split_articles = {
        'train': articles_out[:n_train],
        'valid': articles_out[n_train:n_train + n_valid],
        'test':  articles_out[n_train + n_valid:],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for split, arts in split_articles.items():
        examples = []
        for a in arts:
            examples.extend(a['hard_neg'])
            examples.extend(a['bonus_pos'])
            examples.extend(a['true_neg'])
        random.shuffle(examples)
        out_path = os.path.join(args.output_dir, f'{split}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        n_hn = sum(len(a['hard_neg']) for a in arts)
        n_bp = sum(len(a['bonus_pos']) for a in arts)
        n_tn = sum(len(a['true_neg']) for a in arts)
        print(f"  [{split}] {len(examples):,} examples "
              f"(hard_neg={n_hn:,} bonus_pos={n_bp:,} true_neg={n_tn:,}) -> {out_path}")


if __name__ == '__main__':
    main()
