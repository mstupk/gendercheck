"""
Fine-tune a pre-trained German BERT model for gender term annotation
(token classification / NER with labels O, B-GENDER, I-GENDER).

Reads JSONL data from data_dir (train.jsonl, valid.jsonl).
Saves the best checkpoint (by F1) to model_dir.

Usage:
    python 02_train_model.py data/ model/
    python 02_train_model.py data/ model/ --base-model deepset/gbert-large --epochs 5
"""

import os
import json
import argparse
import numpy as np

import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def tokenize_and_align_labels(examples, tokenizer, label2id):
    """Tokenise word-split input and align BIO labels to subword tokens.

    Continuation subwords of a B-GENDER word receive I-GENDER so that the
    span boundaries remain correct during evaluation.  Special tokens ([CLS],
    [SEP], padding) receive -100 (ignored by the loss).
    """
    tokenized = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )

    all_label_ids = []
    for i, word_labels in enumerate(examples['labels']):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[word_labels[word_id]])
            else:
                # Continuation subword: propagate label, upgrading B→I.
                lbl = word_labels[word_id]
                if lbl == 'B-GENDER':
                    lbl = 'I-GENDER'
                label_ids.append(label2id[lbl])
            prev_word_id = word_id
        all_label_ids.append(label_ids)

    tokenized['labels'] = all_label_ids
    return tokenized


def make_compute_metrics(seqeval_metric, id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [id2label[l] for l in row if l != -100]
            for row in labels
        ]
        pred_labels = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        results = seqeval_metric.compute(predictions=pred_labels, references=true_labels)
        return {
            'precision': results['overall_precision'],
            'recall':    results['overall_recall'],
            'f1':        results['overall_f1'],
            'accuracy':  results['overall_accuracy'],
        }
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune German BERT for gender term annotation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('data_dir',  help='Directory with train.jsonl and valid.jsonl')
    parser.add_argument('model_dir', help='Directory to save the fine-tuned model')
    parser.add_argument('--base-model', default='deepset/gbert-base',
                        help='HuggingFace model name or local path')
    parser.add_argument('--epochs',        type=int,   default=3)
    parser.add_argument('--batch-size',    type=int,   default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-steps',  type=int,   default=500)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.model_dir, 'checkpoints')

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("Loading data ...")
    train_ds = Dataset.from_list(load_jsonl(os.path.join(args.data_dir, 'train.jsonl')))
    valid_ds = Dataset.from_list(load_jsonl(os.path.join(args.data_dir, 'valid.jsonl')))
    print(f"  Train: {len(train_ds):,}  Valid: {len(valid_ds):,}")

    fn_kwargs = {'tokenizer': tokenizer, 'label2id': LABEL2ID}
    train_ds = train_ds.map(
        tokenize_and_align_labels, batched=True, fn_kwargs=fn_kwargs,
        remove_columns=['tokens', 'labels'],
    )
    valid_ds = valid_ds.map(
        tokenize_and_align_labels, batched=True, fn_kwargs=fn_kwargs,
        remove_columns=['tokens', 'labels'],
    )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    seqeval = evaluate.load('seqeval')

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=100,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(seqeval, ID2LABEL),
    )

    print("Training ...")
    trainer.train()

    print(f"Saving model → {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print("Training complete.")


if __name__ == '__main__':
    main()
