"""
Score the Approach B (OpenNMT-py seq2seq) gendered-span detector.

Compares opennmt_run/test.pred against opennmt_run/test.tgt (gold), using
opennmt_run/test.src as the token-position reference (verified identical to
tgt with <gender>/</gender> tags stripped, so src token index == the position
system for both gold and, when lengths match, predicted spans).

Reports:
  - corpus BLEU / chrF / TER (sacrebleu) -- the metric the training config's
    own header comment prescribes
  - UNK rate in predictions (out-of-vocabulary corruption)
  - length-alignment rate (how often pred word count == src word count,
    a precondition for position-level span comparison)
  - span-level exact-match precision/recall/F1 (a predicted span counts as a
    hit only if its [start,end) token range exactly matches a gold span)
  - token-level BIO precision/recall/F1 via seqeval, restricted to
    length-aligned examples
  - per-gendering-convention recall (colon/asterisk/underscore/binnen_i/
    paired), classified post-hoc by re-running the same regexes
    01_extract_training_data.py used to mine the corpus, against each gold
    span's own text

Reproducing test.pred in this environment
------------------------------------------
onmt_translate (from `pip install -e testruns/OpenNMT-py`) does not work here
out of the box: pyonmttok has no wheel for this Python/platform, and
torch>=2.6's default `weights_only=True` rejects this checkpoint's pickled
argparse.Namespace. Both are environment issues, not model issues -- see
SPECIFICATION.md §9.1 for why. Work around them with the compat shims in
opennmt_compat/ (a pyonmttok stand-in covering only what checkpoint loading
actually calls, verified against the OpenNMT-py 3.5.1 source) and a
weights_only=False torch.load patch (safe: this is a locally-produced,
trusted checkpoint), e.g.:

    import torch, sys
    torch.load = (lambda f, _orig=torch.load, **kw: _orig(f, **{**kw, "weights_only": False}))
    sys.path.insert(0, "claude_scripts/opennmt_compat")
    sys.path.insert(0, "testruns/OpenNMT-py")
    sys.argv = ["onmt_translate", "-model", "opennmt_run/model/gendercheck_step_29000.pt",
                "-src", "opennmt_run/test.src", "-output", "opennmt_run/test.pred",
                "-gpu", "-1", "--beam_size", "1", "--batch_size", "8"]
    from onmt.bin.translate import main
    main()

Then score:
    python3 claude_scripts/05_evaluate_opennmt.py \\
        opennmt_run/test.src opennmt_run/test.tgt opennmt_run/test.pred \\
        claude_pipeline_output/approach_b_eval_results.json
"""
import re
import sys
import json
import sacrebleu
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# ---------------------------------------------------------------------------
# Same regexes as claude_scripts/01_extract_training_data.py, used here only
# to classify (not detect) gold spans by convention.
_ALPHA = r'[a-zA-ZäöüÄÖÜß]'
_ALPHA_LOWER = r'[a-zäöüß]'
_ALPHA_UPPER = r'[A-ZÄÖÜ]'

GENDER_PATTERNS = {
    'paired':     re.compile(r'\b(' + _ALPHA + r'{3,})\s+und\s+\1(?:innen|Innen)\b'),
    'asterisk':   re.compile(r'\b' + _ALPHA_UPPER + r'[^\*\s]{2,}\*' + _ALPHA_LOWER + r'{2,}\b'),
    'colon':      re.compile(r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}:' + _ALPHA_LOWER + r'{2,}\b'),
    'underscore': re.compile(r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}_' + _ALPHA_LOWER + r'{2,}\b'),
    'binnen_i':   re.compile(r'\b' + _ALPHA_UPPER + _ALPHA_LOWER + r'{2,}I' + _ALPHA_LOWER + r'{2,}\b'),
}


def classify_convention(span_text):
    hits = [name for name, pat in GENDER_PATTERNS.items() if pat.search(span_text)]
    return hits[0] if hits else 'unclassified'


def extract_spans(tagged_words):
    """Given a whitespace-tokenised, <gender>/</gender>-tagged word list,
    return (stripped_words, [(start, end), ...]) with start/end token
    indices into stripped_words (end exclusive)."""
    stripped = []
    spans = []
    open_start = None
    for w in tagged_words:
        if w == '<gender>':
            open_start = len(stripped)
        elif w == '</gender>':
            if open_start is not None:
                spans.append((open_start, len(stripped)))
                open_start = None
        else:
            stripped.append(w)
    if open_start is not None:  # unterminated span (model never emitted </gender>)
        spans.append((open_start, len(stripped)))
    return stripped, spans


def bio_labels(n_tokens, spans):
    labels = ['O'] * n_tokens
    for start, end in spans:
        for i in range(start, min(end, n_tokens)):
            labels[i] = 'B-GENDER' if i == start else 'I-GENDER'
    return labels


def main(src_path, tgt_path, pred_path):
    src_lines = open(src_path, encoding='utf-8').read().splitlines()
    tgt_lines = open(tgt_path, encoding='utf-8').read().splitlines()
    pred_lines = open(pred_path, encoding='utf-8').read().splitlines()

    n = len(src_lines)
    assert len(tgt_lines) == n, f"tgt has {len(tgt_lines)} lines, expected {n}"
    if len(pred_lines) != n:
        print(f"WARNING: pred has {len(pred_lines)} lines, expected {n} "
              f"-- scoring only the first {min(len(pred_lines), n)} lines",
              file=sys.stderr)
    n = min(n, len(pred_lines))
    src_lines, tgt_lines, pred_lines = src_lines[:n], tgt_lines[:n], pred_lines[:n]

    # ---- sacrebleu (as prescribed by train_config.yaml's own instructions) ----
    bleu = sacrebleu.corpus_bleu(pred_lines, [tgt_lines])
    chrf = sacrebleu.corpus_chrf(pred_lines, [tgt_lines])
    ter = sacrebleu.corpus_ter(pred_lines, [tgt_lines])

    # ---- UNK rate ----
    total_pred_tokens = 0
    unk_pred_tokens = 0
    for line in pred_lines:
        toks = line.split()
        total_pred_tokens += len(toks)
        unk_pred_tokens += sum(1 for t in toks if t == '<unk>')

    # ---- per-example structural + span analysis ----
    aligned = 0
    misaligned = 0
    gold_bio_all, pred_bio_all = [], []

    gold_span_total = 0
    exact_match_total = 0
    pred_span_total = 0

    convention_counts = {}     # convention -> total gold spans
    convention_exact_hits = {} # convention -> exact-match hits

    for src, tgt, pred in zip(src_lines, tgt_lines, pred_lines):
        src_words = src.split()
        gold_stripped, gold_spans = extract_spans(tgt.split())
        assert gold_stripped == src_words, "tgt (minus tags) must equal src"

        pred_stripped, pred_spans = extract_spans(pred.split())

        gold_span_total += len(gold_spans)
        pred_span_total += len(pred_spans)

        # Classify + track exact-match per gold span, regardless of alignment
        # (exact match requires len(pred_stripped)==len(src_words) so indices
        # correspond; otherwise none of this sentence's spans can match).
        length_ok = len(pred_stripped) == len(src_words)
        pred_span_set = set(pred_spans) if length_ok else set()

        for span in gold_spans:
            span_text = ' '.join(src_words[span[0]:span[1]])
            conv = classify_convention(span_text)
            convention_counts[conv] = convention_counts.get(conv, 0) + 1
            if span in pred_span_set:
                exact_match_total += 1
                convention_exact_hits[conv] = convention_exact_hits.get(conv, 0) + 1

        if length_ok:
            aligned += 1
            gold_bio_all.append(bio_labels(len(src_words), gold_spans))
            pred_bio_all.append(bio_labels(len(src_words), pred_spans))
        else:
            misaligned += 1

    precision = exact_match_total / pred_span_total if pred_span_total else 0.0
    recall = exact_match_total / gold_span_total if gold_span_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    results = {
        'n_examples': n,
        'bleu': bleu.score,
        'chrf': chrf.score,
        'ter': ter.score,
        'unk_rate_pred_tokens_pct': 100 * unk_pred_tokens / total_pred_tokens if total_pred_tokens else 0,
        'length_aligned_examples': aligned,
        'length_misaligned_examples': misaligned,
        'length_misaligned_pct': 100 * misaligned / n,
        'gold_span_total': gold_span_total,
        'pred_span_total': pred_span_total,
        'exact_span_match_total': exact_match_total,
        'span_exact_precision': precision,
        'span_exact_recall': recall,
        'span_exact_f1': f1,
        'convention_breakdown': {
            conv: {
                'gold_spans': convention_counts[conv],
                'exact_matches': convention_exact_hits.get(conv, 0),
                'recall_pct': 100 * convention_exact_hits.get(conv, 0) / convention_counts[conv],
            }
            for conv in sorted(convention_counts)
        },
    }

    if gold_bio_all:
        results['seqeval_token_precision'] = precision_score(gold_bio_all, pred_bio_all)
        results['seqeval_token_recall'] = recall_score(gold_bio_all, pred_bio_all)
        results['seqeval_token_f1'] = f1_score(gold_bio_all, pred_bio_all)
        results['seqeval_report'] = classification_report(gold_bio_all, pred_bio_all, digits=4)

    print(json.dumps({k: v for k, v in results.items() if k != 'seqeval_report'}, indent=2, ensure_ascii=False))
    if 'seqeval_report' in results:
        print("\n--- seqeval classification report (length-aligned subset) ---")
        print(results['seqeval_report'])

    return results


if __name__ == '__main__':
    src_path, tgt_path, pred_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out_json = sys.argv[4] if len(sys.argv) > 4 else None
    results = main(src_path, tgt_path, pred_path)
    if out_json:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
