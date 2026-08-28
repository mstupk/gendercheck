# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Gendercheck** is a LibreOffice Writer plugin and ML data pipeline for gendering German text. It automatically converts gender-neutral German text to gendered forms (e.g., "Lehrer" → "Lehrer:innen"). The project has three distinct components:

1. **LibreOffice Plugin** (`apps/poc_app_1/`) — A UNO Python macro that acts as autocorrect for gender forms
2. **Translation pipeline** (`scripts/`) — Processes newspaper HTML to build OpenNMT training data for the masculine→gendered translation task
3. **Annotation pipeline** (`claude_scripts/`) — Trains models to annotate gendered terms in-place with `<gender>…</gender>` tags

## Corpora

| Source | Files | Format |
|---|---|---|
| `corpora/taz.de/` | ~77,000 | `.html` files in nested subdirs |
| `corpora/www.woz.ch/` | ~17,393 | **Extensionless** HTML files (no `.html` suffix) |

`www.woz.ch` files have no extension — any script that needs to process them must sniff content (check for `<!doctype` / `<html`) rather than relying on the file extension. All `claude_scripts/` tools handle this automatically via `--all-files`.

## Translation Pipeline (`scripts/`)

Produces parallel corpora for training an OpenNMT model that converts masculine-plural forms **back** to gendered forms (e.g., "Lehrer" → "Lehrer:innen").

The numbered script suffixes (`_1.py`, `_2.py`, …) are iterative refinements — always use the highest-numbered version for a given stage.

```bash
pip install beautifulsoup4 lxml pyyaml  # + subword-nmt for BPE

python scripts/regex_crawler_1.py corpora/taz.de/ out/gendered_terms.xml
python scripts/retranslator_1.py  out/gendered_terms.xml out/retranslated.xml --timeout 600
python scripts/create_dataset_1.py out/retranslated.xml \
  --outputs train 0.8 valid 0.1 test 0.1 --timeout 600 --seed 42
python scripts/generate_config_yaml_1.py
python scripts/bpe_corpus_tokenizer_1.py
# train from testruns/OpenNMT-py/
onmt_train -config config.yml
```

**Data flow:**
```
HTML → regex_crawler → XML (matches)
     → retranslator  → XML (src/trg pairs: masculine ↔ gendered)
     → create_dataset → .src / .trg text files
     → bpe_tokenizer  → BPE vocab + tokenised files
     → OpenNMT config → onmt_train
```

XML is stream-processed (output files can reach 9–227 GB). Stages can run concurrently via `--timeout` polling.

**XML formats:**
- Crawler: `<matches><entry file="…" path="…"><article>full text</article><match pattern="…">sentence</match>…</entry></matches>`
- Retranslator: `<entries><entry …><src_string>masculine</src_string><trg_string>gendered</trg_string></entry></entries>`

## Annotation Pipeline (`claude_scripts/`)

Produces data and models that annotate gendered terms in-place:
```
src:  Die Lehrer:innen kommen morgen .
tgt:  Die <gender> Lehrer:innen </gender> kommen morgen .
```

Two model approaches are supported:

### Approach A — HuggingFace token classifier (`run_pipeline.sh` / `run_corpora_pipeline.sh`)

Fine-tunes `deepset/gbert-base` for NER-style BIO token classification. Outputs a model usable via `04_annotate.py`.

```bash
cd claude_scripts/
pip install -r requirements.txt

# Single-corpus quick run:
./run_pipeline.sh [corpus_dir]

# Multi-corpus run with equal-quota balancing across sources:
./run_corpora_pipeline.sh [--max-positive 10000] [--skip-training]

# Step-by-step:
python3 01_extract_training_data.py ../corpora/taz.de data/ --max-positive 50000
python3 02_train_model.py data/ model/ --base-model deepset/gbert-base --epochs 3
python3 03_evaluate.py data/ model/
echo "Die Lehrer:innen kommen morgen." | python3 04_annotate.py model/
```

**JSONL format** (`data/*.jsonl`): `{"tokens": […], "labels": […]}` with BIO labels `O`, `B-GENDER`, `I-GENDER`.

### Approach B — OpenNMT-py seq2seq (`run_pipeline_3.sh`)

Produces `.src`/`.tgt` plain-text files and a token-based vocabulary for OpenNMT-py 3.x. `<gender>` and `</gender>` are plain whitespace-delimited tokens — **no BPE or subword transforms**.

```bash
cd claude_scripts/
./run_pipeline_3.sh [OPTIONS]
# key options: --max-positive N  --gpu N  --skip-extract  --skip-convert

# After the script finishes, train with:
cd ../testruns/OpenNMT-py && pip install -e .
onmt_train -config ../../opennmt_run/train_config.yaml
```

**`onmt_build_vocab` cannot be used** — `pyonmttok` is not installable in this environment. `build_opennmt_vocab.py` produces the identical `word\tcount` format instead and is called automatically by `run_pipeline_3.sh`.

### Script inventory

| Script | Purpose |
|---|---|
| `01_extract_training_data.py` | HTML → BIO-labeled JSONL, article-level splits |
| `02_train_model.py` | Fine-tune gbert-base for token classification |
| `03_evaluate.py` | seqeval precision/recall/F1 on test set |
| `04_annotate.py` | Inference: wraps gender terms with `<gender>…</gender>` |
| `convert_bio_to_opennmt.py` | BIO JSONL → `.src` / `.tgt` plain-text files |
| `build_opennmt_vocab.py` | Builds `word\tcount` vocab files (replaces `onmt_build_vocab`) |
| `_merge_splits.py` | Merges + shuffles per-source JSONL splits |
| `run_pipeline.sh` | Single-corpus HuggingFace pipeline |
| `run_corpora_pipeline.sh` | Multi-corpus HuggingFace pipeline (equal-quota) |
| `run_pipeline_3.sh` | Multi-corpus OpenNMT-py pipeline (equal-quota) |

### Key design decisions (shared by both approaches)

- **Umlaut-aware regexes**: use `[a-zA-ZäöüÄÖÜß]` not `\w` — `\w` matches digits and underscores, producing false positives in German newspaper text.
- **Paired-form backreference**: `\b(\w+)\s+und\s+\1(?:innen|Innen)\b` — the `\1` ensures the stem before "und" and after match exactly.
- **Article-level splitting**: train/valid/test splits are done at article granularity so no sentences from the same article appear in multiple splits (prevents data leakage).
- **Equal-quota balancing**: each corpus source is extracted with the same `--max-positive` cap before merging, so no single source dominates despite size differences (taz.de: 77k files vs. woz.ch: 17k).

## Generated Dataset (`opennmt_run/`)

The OpenNMT-py dataset has been generated and is ready for training:

| File | Lines |
|---|---|
| `opennmt_run/opennmt_data/train.src` / `.tgt` | 33,034 |
| `opennmt_run/opennmt_data/valid.src` / `.tgt` | 3,801 |
| `opennmt_run/opennmt_data/test.src` / `.tgt` | 3,165 |
| `opennmt_run/vocab/vocab.src` | 50,000 tokens |
| `opennmt_run/vocab/vocab.tgt` | 50,000 tokens (incl. `<gender>`, `</gender>`) |
| `opennmt_run/train_config.yaml` | Training config, ready for `onmt_train` |

Source breakdown: ~50% taz.de, ~50% woz.ch per split (10,000 positive examples each).

## OpenNMT-py Location

The OpenNMT-py 3.5.1 source tree is at `testruns/OpenNMT-py/`. It is **not installed as a package** (`pyonmttok` dependency is unresolvable in this environment). To install before training:

```bash
cd testruns/OpenNMT-py && pip install -e .
```

## LibreOffice Plugin (`apps/poc_app_1/`)

Implements `XKeyListener` UNO interface. On each keystroke, buffers input and detects configured patterns; when matched, moves cursor back, selects text, and replaces with the gendered form. The handler is stored in a global list to prevent garbage collection by the UNO runtime. Package as `.oxt` for distribution.
