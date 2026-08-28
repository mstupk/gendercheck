"""
Local inference service for the Gendercheck LibreOffice plugin
(SPECIFICATION.md §4.2: "the plugin is Python running inside LibreOffice's
bundled interpreter under UNO -- it should NOT load a transformer model
in-process ... a small local inference service").

Stdlib-only HTTP server (no Flask/FastAPI dependency) so it's trivial to
launch from the extension's own bundled Python environment, separate from
LibreOffice's own UNO Python runtime.

Pipeline per SPECIFICATION.md's decided design:
  Stage 1/2 (claude_pipeline_output_stage2/model, the hard-negative-mined
  BERT classifier recommended in §11.5) finds candidate masculine spans.
  Stage 3 (claude_scripts/11b_stage3_inflect.py's inflect()) turns each
  candidate into a suggested surface form, defaulting to convention='colon'
  per §6.2's decision.

API:
    POST /check   {"text": "Die Lehrer kommen morgen."}
        -> {"candidates": [{"start": 4, "end": 10, "word": "Lehrer",
                             "suggestion": "Lehrer:innen", "confidence": 0.98}]}
    GET  /health  -> {"status": "ok"}

Usage:
    python inference_server.py --port 8765 \\
        --model ../../claude_pipeline_output_stage2/model \\
        --stage3 ../../claude_scripts

The suggest-not-auto-apply decision (§6.1) means this server's job is only
to SURFACE candidates with a confidence score -- the UNO Proofreader
component (python/gendercheck_proofreader.py) decides what to do with them
(currently: surface everything above --threshold, defaulting to the 0.5
operating point per §6.1's note that a suggest-only UX doesn't need the
auto-apply-grade >99%-precision threshold).
"""

import os
import sys
import json
import argparse
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ['O', 'B-GENDER', 'I-GENDER']
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

MODEL = None
TOKENIZER = None
DEVICE = None
INFLECT = None
THRESHOLD = 0.5
CONVENTION = 'colon'


def load_stage3(stage3_dir):
    path = os.path.join(stage3_dir, '11b_stage3_inflect.py')
    spec = importlib.util.spec_from_file_location('stage3_inflect', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.inflect


def find_candidates(text):
    """Return [(start_char, end_char, word, confidence), ...] for tokens
    scored above THRESHOLD -- word-level P(not-O), first-subword rule, same
    methodology as claude_scripts/06_evaluate_stage1_threshold_sweep.py."""
    words = text.split()
    if not words:
        return []

    # Track character offsets for each whitespace-split word.
    offsets = []
    pos = 0
    for w in words:
        start = text.find(w, pos)
        offsets.append((start, start + len(w)))
        pos = start + len(w)

    encoding = TOKENIZER(words, truncation=True, is_split_into_words=True,
                          max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = MODEL(**encoding).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    word_ids = encoding.word_ids(0)

    word_score = {}
    for pos_idx, wid in enumerate(word_ids):
        if wid is not None and wid not in word_score:
            word_score[wid] = 1.0 - float(probs[pos_idx][LABEL_LIST.index('O')])

    results = []
    for i, w in enumerate(words):
        score = word_score.get(i, 0.0)
        if score >= THRESHOLD:
            start, end = offsets[i]
            results.append((start, end, w, score))
    return results


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; this runs as a background service

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            self._send_json({'status': 'ok'})
        else:
            self._send_json({'error': 'not found'}, status=404)

    def do_POST(self):
        if self.path != '/check':
            self._send_json({'error': 'not found'}, status=404)
            return
        length = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            text = payload['text']
        except Exception as e:
            self._send_json({'error': f'bad request: {e}'}, status=400)
            return

        candidates = []
        for start, end, word, score in find_candidates(text):
            try:
                suggestion = INFLECT(word, CONVENTION)
            except ValueError:
                continue
            candidates.append({
                'start': start, 'end': end, 'word': word,
                'suggestion': suggestion, 'confidence': round(score, 4),
            })
        self._send_json({'candidates': candidates})


def main():
    global MODEL, TOKENIZER, DEVICE, INFLECT, THRESHOLD, CONVENTION

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='Path to the trained Stage 2 model directory')
    parser.add_argument('--stage3-dir', required=True, help='Path to claude_scripts/ (for 11b_stage3_inflect.py)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--convention', default='colon', choices=('colon', 'asterisk', 'underscore', 'binnen_i'))
    args = parser.parse_args()

    THRESHOLD = args.threshold
    CONVENTION = args.convention

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model} on {DEVICE} ...", file=sys.stderr)
    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    MODEL = AutoModelForTokenClassification.from_pretrained(args.model).to(DEVICE)
    MODEL.eval()
    INFLECT = load_stage3(args.stage3_dir)
    print("Model ready.", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Gendercheck inference server listening on http://{args.host}:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == '__main__':
    main()
