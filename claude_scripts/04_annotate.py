"""
Annotate gendered German terms in text using the fine-tuned model.

Wraps each detected span with <gender>...</gender> tags.

Input is read line-by-line; each non-empty line is annotated independently.

Usage:
    echo "Die Lehrer:innen kommen morgen." | python 04_annotate.py model/
    python 04_annotate.py model/ --input raw.txt --output annotated.txt

Example output:
    Die <gender>Lehrer:innen</gender> kommen morgen.
"""

import sys
import re
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def predict_words(words, model, tokenizer, device):
    """Return a list of (word, label) pairs for a pre-split word list."""
    if not words:
        return []

    encoding = tokenizer(
        words,
        truncation=True,
        is_split_into_words=True,
        max_length=128,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits

    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    word_ids = encoding.word_ids(0)

    # First-subword rule: use the prediction of the first subword for each word.
    word_label = {}
    for pos, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_label:
            word_label[word_id] = ID2LABEL[preds[pos]]

    return [(words[i], word_label.get(i, 'O')) for i in range(len(words))]


def annotate_word_labels(word_label_pairs):
    """Insert <gender>...</gender> around contiguous B/I-GENDER spans.

    Handles edge cases:
    - Lone I-GENDER (no preceding B-GENDER): treated as start of a new span.
    - Empty input: returns empty string.
    """
    parts = []
    in_span = False

    for word, label in word_label_pairs:
        is_gender = label in ('B-GENDER', 'I-GENDER')
        starts_span = label == 'B-GENDER' or (label == 'I-GENDER' and not in_span)

        if starts_span:
            if in_span:
                parts.append('</gender>')
            if parts:
                parts.append(' ')
            parts.append('<gender>')
            parts.append(word)
            in_span = True
        elif is_gender:  # I-GENDER continuing a span
            parts.append(' ')
            parts.append(word)
        else:  # O
            if in_span:
                parts.append('</gender>')
                in_span = False
            if parts:
                parts.append(' ')
            parts.append(word)

    if in_span:
        parts.append('</gender>')

    return ''.join(parts)


def annotate_line(line, model, tokenizer, device):
    """Annotate a single line of text.

    The line is split into sentences at sentence-terminal punctuation so that
    each sentence fits within the model's 128-token context window.
    Sentences are annotated independently and reassembled.
    """
    sentences = re.split(r'(?<=[.!?])\s+', line.strip())
    annotated_sentences = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        words = sentence.split()
        pairs = predict_words(words, model, tokenizer, device)
        annotated_sentences.append(annotate_word_labels(pairs))

    return ' '.join(annotated_sentences)


def main():
    parser = argparse.ArgumentParser(
        description='Annotate gendered German terms with <gender>...</gender> tags.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('model_dir', help='Directory with fine-tuned model')
    parser.add_argument('--input',  help='Input text file (default: stdin)')
    parser.add_argument('--output', help='Output file (default: stdout)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model_dir} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    print("Model ready.", file=sys.stderr)

    in_stream  = open(args.input,  'r', encoding='utf-8') if args.input  else sys.stdin
    out_stream = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout

    try:
        for line in in_stream:
            stripped = line.rstrip('\n')
            if stripped.strip():
                out_stream.write(annotate_line(stripped, model, tokenizer, device) + '\n')
            else:
                out_stream.write('\n')
    finally:
        if args.input:
            in_stream.close()
        if args.output:
            out_stream.close()


if __name__ == '__main__':
    main()
