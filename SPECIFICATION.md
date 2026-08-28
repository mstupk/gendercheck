# Gendercheck — Implementation Specification

**Status:** Draft, derived from analysis of the existing repository state (2026-08-27).
**Purpose:** State what this project is actually trying to become, document precisely what
exists today versus what is assumed to exist, and specify the work needed to turn the
current collection of prototypes into the one artifact the project was started to produce:
a working LibreOffice autocorrect plugin that genders German text.

This document does not replace `CLAUDE.md` (which documents *how to invoke* what's here).
It answers a different question: *what is this actually for, what's missing, and what
should get built next.*

---

## 1. Intention analysis

### 1.1 The stated goal

From `scripts/README.md` (the oldest surviving description, mirrored in the standalone
`git/gendercheck` and `git/gendercheck-0.1` snapshots):

> A LibreOffice Writer Plugin which genders German text similar to an autocorrection.
> Written as Programming Project for University of Zurich. This project is intended to
> be commercialised at some later point.

Read literally, this is a very specific product: a user types ordinary German
(`Die Lehrer kommen morgen.`), and — the same way LibreOffice silently fixes `teh` → `the`
— the plugin silently or semi-silently turns the masculine-plural generic into a gendered
form (`Die Lehrer:innen kommen morgen.`) as they type or on demand.

Everything else in the repository — the two ML pipelines, the corpora, the OpenNMT
training runs — exists to build the "brain" this autocorrect needs, because a fixed regex
substitution table (`Lehrer` → `Lehrer:innen`) doesn't work for a highly inflected
language: German nouns change form for case (`Lehrer/Lehrers/Lehrern`), the gendered form
has to inflect the same way, and — critically — you need a judgment call about *whether a
given occurrence should be gendered at all* (proper nouns, quotes, compound words, and
already-gendered text must be left alone).

### 1.2 What actually got built

Three independent efforts exist, at three different levels of completion, and — this is
the central finding of this analysis — **they solve two different problems, and neither
solves the one the plugin needs**:

| Effort | Problem it solves | Solves the product's problem? |
|---|---|---|
| `scripts/` + `opennmt_run/` seq2seq idea in `CLAUDE.md` §Translation Pipeline | Mine real newspaper text for *already-gendered* sentences, algorithmically derive the masculine-only counterfactual, and train a model to translate masculine → gendered | **Yes, this is the actual generation task** — but see §3.1, it was never trained to a finished model |
| `claude_scripts/` Approach A (HuggingFace `gbert-base` BIO token classifier) | Given a sentence, mark which tokens are an *already-gendered* span (`<gender>Lehrer:innen</gender>`) | No — this is a **detector of existing gendering**, not a generator |
| `claude_scripts/` Approach B (OpenNMT-py seq2seq, same BIO task reframed as translation) | Same detection task as Approach A, different model architecture | No — same task as above |
| `apps/poc_app_1/` (the actual plugin) | Find literal string `"foo"` in a Writer document, replace with `"bar"` | No — this is UNO-API scaffolding, not gendering logic of any kind |

In short: the project has spent most of its engineering effort validating that a model
*can be trained* to find gendered terms that are already there, and comparatively little
effort on the harder, actually-needed task of *deciding where to insert new gendering*
and *what inflected form to insert*. The plugin that was supposed to consume either
model doesn't call into any model at all yet.

This isn't a criticism of the sequencing — building a reliable detector first is a
reasonable way to bootstrap training data and to give the eventual autocorrect a
"don't touch already-gendered text" guard — but it means the spec below has to treat
generation as unstarted, not nearly-done.

### 1.3 Corroborating evidence

- `ai_chats/deepseek-chat.2025-07-13.pdf` (6 pages) is a DeepSeek chat transcript from the
  project's early days, entirely about iteratively deriving the regex patterns
  (paired/asterisk/colon/underscore/Binnen-I forms) later baked into
  `scripts/regex_crawler_*.py` and `claude_scripts/01_extract_training_data.py`. It
  confirms the corpus-mining direction was the starting point, not a UX or plugin
  discussion — no design conversation about the autocorrect interaction model exists
  anywhere in the repo.
- `opennmt_run/model/` contains real trained checkpoints
  (`gendercheck_step_{10000,15000,20000,25000,29000}.pt`) plus `train.log`/`translate.log`.
  Inspecting `opennmt_run/train_config.yaml` and a sample of
  `opennmt_run/opennmt_data/train.src` / `.tgt` confirms this trained model is the
  **Approach B detector** (src already contains `GewerkschafterInnen`, tgt just wraps it
  in `<gender>` tags). **Correction from an earlier pass of this document:** `test.pred`
  was actually 0 bytes — `translate.log` shows the original `onmt_translate` invocation
  crashed on checkpoint load (a PyTorch 2.6+ `torch.load` default change) before producing
  any output, so no evaluation of this checkpoint had actually been run. A prior version
  of this analysis mistakenly read a `test.src`/`test.tgt` comparison as `pred`/`tgt` and
  reported the detector as validated on that basis — it wasn't. It has since actually been
  run end-to-end; see **§9 Evaluation results** for real numbers, which support a similar
  but more qualified conclusion (works well in-vocabulary, has a specific and
  well-characterized failure mode otherwise).
- `claude_pipeline_output/model/` and `claude_pipeline_output/data_merged/` (Approach A,
  the BERT token classifier) are **empty directories** — despite `CLAUDE.md` documenting
  full usage instructions for it, that pipeline was never run to completion.
- `testruns/testrun_{1,2,3}/` (the `scripts/` seq2seq masculine→gendered generation
  attempt) contain prepared `.src`/`.trg`/BPE/vocab files but **no model checkpoints
  anywhere under `testruns/`**, and `testrun_1` references `toy-ende.tar.gz` — the stock
  OpenNMT tutorial toy dataset — alongside real extracted data, suggesting this track
  got as far as pipeline plumbing and tokenizer validation, not a finished trained model.
- `apps/poc_app_1/auto_replace.py` and `foo2bar/python/foo2bar_ext.py` both do a literal
  `"foo"` → `"bar"` `findFirst`/replace over the whole document, triggered either by a
  Basic-macro call (`XSCRIPTCONTEXT`) or a `com.sun.star.task.Job` (a
  document-open/manual-trigger hook). **Neither uses `XKeyListener`, per-keystroke
  buffering, or cursor-relative undo of the last typed word** — despite `CLAUDE.md`
  §"LibreOffice Plugin" describing exactly that design. Treat `CLAUDE.md`'s plugin
  section as an aspirational design note, not documentation of existing code, until a
  real `XKeyListener` implementation exists.

---

## 2. Scope of "an implementation"

Given the above, "implementing this project" means closing three gaps in order:

1. **Build the generation model** the product actually needs (masculine occurrence →
   correctly inflected gendered form, plus a should-I-touch-this-at-all decision).
2. **Wire a real plugin** that calls into a model (detector, generator, or both) instead
   of doing a hardcoded string replace, with an interaction design appropriate for a
   change that is *semantically* riskier than a spelling fix.
3. **Consolidate the pipeline** so there is one blessed path from corpus → trained model
   → packaged `.oxt`, instead of two parallel script trees with numbered-suffix
   variants and undocumented "highest number wins" conventions.

The rest of this document specs each in turn.

---

## 3. Generation model (the actual missing piece)

### 3.1 Why the existing seq2seq direction isn't sufficient by itself

The `scripts/` pipeline's approach — mine real gendered sentences, strip them down to a
masculine counterfactual, train masculine→gendered seq2seq translation — is a sound way
to get supervision without hand-labeling. But full-sentence seq2seq has two problems for
this specific product:

- **Blast radius.** A transformer translating the whole sentence can silently alter
  unrelated words (that's normal seq2seq behavior — it optimizes sequence likelihood, not
  minimal edit distance). An autocorrect feature must not rewrite text the user didn't ask
  it to touch. This showed up as a real risk, not a hypothetical one, given the eval
  samples for the *detector* model already show it happily passing through large chunks of
  boilerplate (nav menus, footers) unchanged — that's fine for detection, but a generation
  model needs an explicit **copy-mostly, edit-locally** bias, which vanilla seq2seq
  doesn't give you for free.
- **Latency/granularity mismatch with "autocorrect."** Real autocorrect fires on a single
  completed word, not a full sentence, and needs to be fast enough to feel instant on every
  keystroke boundary. A beam-search transformer over the whole current sentence on every
  word boundary is the wrong shape for that; it's the right shape for "run gendering over
  a whole document/paragraph on demand," which is a legitimate secondary feature but not
  the core autocorrect loop.

### 3.2 Recommended approach: span-local classify-then-inflect

Reuse the one asset that *is* proven to work — the Approach B detector — but repoint it at
a different labeling scheme, and add a small second stage:

**Stage 1 — Candidate detection (mostly reuse what exists).**
Retrain the existing BIO/seq2seq architecture (whichever wins the bake-off in §3.3) on
masculine-plural generic nouns as the span to find, instead of already-gendered spans.
Training data: take the *same* mined (masculine, gendered) pairs the `scripts/` pipeline
already produces via `regex_crawler` + `retranslator`, but instead of using them as
translation pairs, use the masculine sentence as input and label the noun span that maps
to the historically-observed gendered form as the positive span. This is a straightforward
relabeling of data that's already being collected — no new crawling needed.

**Stage 2 — Should this occurrence be gendered? (new, needs explicit negative data).**
Not every masculine-plural noun should be gendered even when it's grammatically a person
noun — proper nouns (`die Bergers`), fixed compounds, quotations of un-gendered source
text, and non-person nouns that happen to match the surface pattern are all
false-positive risks the regex-based detector already had to guard against (see the
umlaut/backreference/capital-stem tricks documented in
`claude_scripts/01_extract_training_data.py`). This stage should be trained with hard
negatives specifically mined for these cases — pull sentences the regex almost matched but
didn't (near-miss mining), not just random negatives, since random negatives are too easy
and won't teach the model the actual decision boundary.

**Stage 3 — Inflection (new, and the part with no existing prototype at all).**
Given a candidate span, determine the correct gendered surface form respecting case and
number. This is a bounded morphological problem, not a full generation problem — a
lookup/rule model (stem + case-aware suffix table, informed by the German declension
patterns already implicit in the mined `(masculine, gendered)` pairs) will likely
outperform and be far cheaper than fine-tuning a generator for it. Only fall back to a
learned seq2seq inflector for stems the rule table can't resolve (irregular plurals,
compounds). **Do not conflate this with Stage 1** — Approach B's model was never asked to
produce a *new* surface form, only to locate an existing one; that capability doesn't
transfer for free.

**Style parameter.** All four surface conventions the regexes already recognize (`:`, `*`,
`_`, Binnen-I) are stylistic choices, not different tasks — Stage 3 should take a
user-configurable target style as an argument and produce the requested variant, rather
than training one model per style or hardcoding one.

### 3.3 Model architecture bake-off

Before committing engineering time, run Stage 1 relabeled data through **both** existing
candidate architectures (fine-tuned `gbert-base` token classifier, OpenNMT-py seq2seq) at
small scale, since the repo already has both pipelines built and this is now a
Stage-1-only relabeling exercise, not new plumbing. Pick the winner on precision at fixed
recall (see §5), not raw F1 — for an autocorrect feature, false positives (gendering
something that shouldn't be) are more costly to the user than false negatives.

### 3.4 What NOT to build

Do not attempt to finish the original `scripts/`/`testruns/` masculine→gendered
full-sentence translation model as originally scoped. The classify-then-inflect design in
§3.2 supersedes it for the interactive use case; keep the `scripts/` mining pipeline (it's
the shared data source for both stages) but retire the full-sentence-seq2seq training
target, and don't invest further in `testruns/testrun_{1,2,3}`'s toy-data configuration —
it was scaffolding, not a partially-finished production model.

---

## 4. LibreOffice plugin

### 4.1 Current state vs. required state

Replace both `apps/poc_app_1/auto_replace.py` (Basic-macro-triggered) and
`foo2bar/python/foo2bar_ext.py` (`Job`-triggered) with a single extension implementing
`com.sun.star.awt.XKeyListener`, registered on the Writer document's text view, per the
design already sketched (but not implemented) in `CLAUDE.md`. Key behavioral requirements
that the current `"foo"→"bar"` stub doesn't need to worry about but the real feature does:

- **Trigger point:** fire the check when a word-boundary character is typed (space,
  punctuation, newline) — mirrors how LibreOffice's own spelling autocorrect decides a
  word is "finished," and avoids re-evaluating a partially-typed word on every keystroke.
- **Undo integration:** the replacement must be a single undo-able action
  (`Ctrl+Z` reverts exactly the gendering, not also the character that triggered it) —
  the existing `foo2bar` PoC doesn't address undo at all since it's a batch find-replace,
  not a live-typing hook.
- **Immediate reversibility / opt-out per instance:** standard autocorrect UX
  (LibreOffice's built-in autocorrect shows a small indicator and lets you revert a single
  instance without disabling the feature) — needed here more than for typo-correction,
  because a wrong gendering call is a *meaning* change, not a typo fix, and users will
  reasonably distrust a feature that silently overrides their word choice with no
  easy undo-just-this-one path.
- **Confidence gating:** below some confidence threshold from Stage 2 (§3.2), don't
  auto-apply — surface it the way spell-check does (wavy underline / suggestion on
  right-click) instead of silently rewriting. This is a product decision that needs
  explicit resolution, not just an implementation detail — see §6.
- **Style setting:** a per-user (Tools → Options) or per-document setting for which
  gendering convention to apply (`:`, `*`, `_`, Binnen-I), passed through as the Stage 3
  style parameter.
- **Global on/off toggle and a scope guard**: don't run over text inside code blocks,
  URLs, email addresses, or existing `<gender>`-tagged/already-gendered spans (reuse the
  Stage-1-adjacent detector as a "don't touch, already gendered" guard directly — this is
  a case where the *existing* Approach B detector is directly reusable in production,
  unlike the generation task). **Caveat, per §9:** the trained checkpoint has an ~8%
  out-of-vocabulary token corruption rate on real newspaper text, so wire this guard
  through a plain-text OOV check first (fall back to "don't touch" whenever the sentence
  contains a word outside the model's 50k vocab) rather than trusting the model's span
  output blindly on such input.

### 4.2 Inference boundary

The plugin is Python running inside LibreOffice's bundled interpreter under UNO — it
should **not** load a transformer model in-process. Specify a small local inference
service (a background process, invoked via local HTTP or a UNIX socket) that the
extension talks to; the extension packages/starts it, or ships instructions for a
one-time local setup. This keeps the `.oxt` itself lightweight and avoids fighting
LibreOffice's bundled Python for `torch`/`transformers` dependencies (a real risk today —
`04_annotate.py`'s dependency stack has no reason to be importable from the UNO Python
runtime).

### 4.3 Packaging

Follow the existing `foo2bar/` `.oxt` structure (`META-INF/manifest.xml` +
`description.xml` + the Python component) as the packaging template — it's already
correct for a single-file UNO Python component and needs no changes, just a real
implementation file and a manifest entry for whatever config UI is added for §4.1's
settings.

---

## 5. Evaluation framework

Neither existing model has an evaluation report checked into the repo (`03_evaluate.py`
exists but was never run — its output directory is empty, per §1.3). Before building on
top of either model, produce one. Minimum bar:

- **Detector (Approach B, already trained):** ~~run `03_evaluate.py`-equivalent seqeval
  precision/recall/F1 on its held-out test split~~ — **done, see §9.** Binnen-I (a bare
  capital `I`, e.g. `LeserInnen`) turned out to be the one convention with a real recall
  gap (73% vs. 93–95% for the colon/asterisk forms) — though the dominant cause turned out
  to be vocabulary coverage on rare compounds, not the weaker regex signal per se.
- **New Stage 1/2/3 pipeline:** track precision at fixed recall (per §3.3), not just F1,
  since the cost of a false-positive gendering is asymmetric with a false negative for an
  autocorrect UX. Also track exact-match on the inflected surface form (Stage 3) separately
  from span/candidate accuracy (Stage 1/2) — conflating them hides which stage is actually
  failing.
- **End-to-end plugin QA:** since this can't be meaningfully validated by unit tests alone,
  budget for manual testing inside actual LibreOffice Writer — type real German paragraphs
  from held-out `taz.de`/`woz.ch` articles and check both false-positive and false-negative
  behavior interactively, per this project's standing practice of testing UI features live
  rather than trusting test suites alone.

---

## 6. Open product questions (need a decision, not an implementation)

These aren't engineering unknowns — they're decisions about what the feature is, and the
answer changes what gets built in §3–4. Flag rather than assume:

1. ~~**Auto-apply vs. suggest?**~~ — **Decided: suggest.** Spell-check-style suggestion
   requiring explicit accept, not silent autocorrect-style replacement. Implication for
   §4.1: the confidence-gating threshold doesn't need to clear the very high (>99%
   precision) bar an auto-apply design would require — each model's default 0.5-threshold
   operating point (§10.2, §11.2) is a reasonable starting point for what gets surfaced as
   a suggestion, since the user reviews before accepting. The precision-at-fixed-recall
   tables remain useful for tuning how aggressively suggestions are surfaced, just without
   the auto-apply-grade precision floor.
2. ~~**Default gendering convention?**~~ — **Decided: colon (`:`).** Also the
   empirically best-supported choice: colon was both the most frequent convention by a
   wide margin in the mined corpus data (§10.2: 1,523 of 2,084 gold spans in Stage 1's test
   set) and the most reliably inflected by Stage 3's rule (§12.3: 99.1% type / 99.9%
   frequency-weighted accuracy, the best of the four). `claude_scripts/11b_stage3_inflect.py`'s
   `inflect()` should be called with `convention='colon'` as the default; still expose the
   other three as a user-configurable override (§4.1) rather than hardcoding colon
   exclusively, since style guides do vary.
3. ~~**Scope of "gendered": people-nouns only, or also implied-gender pronouns/articles?**~~
   — **Decided: nouns, pronouns, and articles all in scope.** This is a real scope
   expansion, not just a config flag: everything built in §9–§12 (all four milestones)
   only ever targets nouns — the `GENDER_PATTERNS` regexes, the mined corpus data, Stage
   1/2's detection models, and Stage 3's inflection rule are all noun-specific by
   construction. Pronoun/article gendering in German doesn't reuse the same conventions
   (no established `:`/`*`/`_`/Binnen-I-equivalent suffix pattern the way nouns have one —
   constructions like `er/sie`, `sier`, `xier`, `jede*r`, `der:die` are far less
   standardized in real usage than noun-suffixing is). Treat this as a genuinely separate
   research thread needing its own pattern-discovery pass over the corpus (the same kind
   of iterative regex development documented in `ai_chats/` for nouns) before any Stage
   1-equivalent detection work can start — not an extension of the existing noun pipeline.
   Not started; would need its own milestone once scoped.
4. ~~**Commercialization vs. MIT license.**~~ — **Decided: CC BY-SA 4.0, code included.**
   Applied at the project root (`LICENSE`) for the plugin code, pipeline scripts, trained
   models, and derived datasets alike — supersedes the MIT license that only ever existed
   in the stale `git/gendercheck/` snapshot (§7), which the working tree itself never had a
   root license for at all. Two tradeoffs to go in with eyes open, since they were the
   reason MIT-vs-commercialization was flagged as a question in the first place:
   - CC BY-SA's ShareAlike clause is the opposite of what pure commercialization-via-
     exclusivity would want — anyone can redistribute and adapt the plugin, even
     commercially, as long as they credit and share alike. This is a defensible choice if
     the commercialization plan is service/support/distribution-based (e.g. a maintained,
     trusted LibreOffice extension listing) rather than exclusivity-based, but it does not
     block a competitor from shipping their own build.
   - CC BY-SA was not designed for software: it has no patent grant, and is not an
     OSI-approved open-source license — this could matter if the plugin is ever submitted
     to a distro package repository or an extension store with its own license vetting.
     Noting this for the record per Creative Commons' own guidance; not blocking the
     decision already made.
5. **Corpus/licensing risk for the mined training data.** The corpora
   (`corpora/taz.de/`, `corpora/www.woz.ch/`) are full-text newspaper archives; training
   and redistributing a model derived from them is a separate legal question from
   redistributing the plugin code, and should be resolved before any commercial
   distribution of trained weights.

---

## 7. Repository hygiene (blocking productionization, not urgent otherwise)

Not part of the model/plugin work, but should happen before either is built on top of the
current tree, since it's actively confusing right now:

- **Two parallel script trees** (`scripts/` vs `claude_scripts/`) solving overlapping
  problems with different naming conventions (`_N.py` suffix vs. `NN_name.py` prefix).
  Consolidate into one pipeline once §3.2's design is settled — there's no reason to keep
  maintaining `scripts/regex_crawler_{1,2,3}.py` as three live files instead of one with
  history in git.
- **`CLAUDE.md`'s plugin section describes unimplemented code as if it exists.** Update it
  once §4 is implemented, or mark it explicitly as a design target in the meantime so
  future contributors (including future Claude Code sessions) don't assume the
  `XKeyListener` hookup already works.
- **No top-level git repository** for the working tree at `/home/bb4g/work/gendercheck/`
  itself — only the `git/gendercheck` and `git/gendercheck-0.1` subfolders are actual
  clones, and they're stale snapshots (last commit predates most of `claude_scripts/`
  and `opennmt_run/`). Decide whether those two are meant to stay as historical
  snapshots or should be retired in favor of turning the working tree itself into the
  canonical repo.

---

## 8. Suggested milestone order

1. ~~Run the missing evaluation (§5) on the existing Approach B detector~~ — **done, see
   §9.** Its per-convention precision/recall directly informs the Stage 1/2 design in
   §3.2: in particular, fix the vocabulary-coverage failure mode (§9.4) before reusing
   this checkpoint as the "already-gendered" guard proposed in §4.1.
2. ~~Relabel existing mined data for Stage 1 (masculine-span detection) and re-run the same
   two architectures already proven out in Approach A/B~~ — **done, see §10.** BERT wins
   decisively (98.1% F1 vs. 90.7%); carry it, not OpenNMT-py, into Stage 2/3.
3. ~~Build Stage 2 (should-gender decision) with deliberately mined hard negatives~~ —
   **done, see §11.** Cut Stage 1's false-positive rate on known-hard cases by 70.8%
   (100%→29.2%) for a 1.2-point recall cost; use `claude_pipeline_output_stage2/model`,
   not Stage 1's, as the production candidate-detection model going forward.
4. ~~Build Stage 3 (case-aware inflection) as a rule/lookup table first; only add a learned
   fallback once the rule table's coverage gap is measured against held-out mined pairs~~ —
   **done, see §12.** The rule table alone hits 97.9% type-level / 99.5% frequency-weighted
   coverage; no learned fallback needed. Also overturned "case-aware" itself — real usage
   doesn't case-inflect the suffix at all (§12.1).
5. ~~Resolve the open product questions in §6~~ — **mostly done.** Items 1, 2, 3, and 4
   decided (see §6 for each); item 5 (corpus licensing risk) still open.
6. ~~Implement the real plugin (§4) against a stub inference service first, swap in the
   real pipeline once §2–4 above land~~ — **done, see §13.** Built against the real Stage
   2/3 pipeline, using LibreOffice's native grammar-checker service rather than the
   `XKeyListener` originally sketched in §4 (§6.1's suggest decision made that the right
   call — see §13.1). End-to-end functionality is screenshot-verified in the real
   interactive UI (§13.3): "Lehrer" underlined with a "Lehrer:innen" suggestion at 99%
   confidence, exactly as designed. Getting there surfaced and fixed a real bug — a
   `createInstanceWithArgumentsAndContext` constructor-signature mismatch that silently
   dropped the checker from LibreOffice's active set — root-caused via a web search for a
   real-world report of the same failure mode rather than continued blind trial-and-error.
7. Re-address commercialization/licensing (§6.4–6.5) before any distribution outside the
   development team. §6.4 (license) is decided; §6.5 (corpus licensing risk) is not.

---

## 9. Evaluation results — Approach B detector (OpenNMT-py seq2seq)

This section fills the gap flagged in §5 and corrects the erroneous claim in an earlier
version of §1.3 (which mistook a `src`/`tgt` comparison for `pred`/`tgt` and reported the
detector as already validated — it wasn't; `test.pred` was actually empty).

### 9.1 What was actually broken

`opennmt_run/model/translate.log` shows the checkpointed model was never successfully
evaluated: `onmt_translate` crashed on the `-model gendercheck_step_29000.pt` load with

```
_pickle.UnpicklingError: Weights only load failed ...
WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an allowed
global by default.
```

This is a PyTorch 2.6+ compatibility break (`torch.load`'s `weights_only` default flipped
from `False` to `True`), unrelated to the model itself. Two environment gaps also block
`CLAUDE.md`'s documented `pip install -e testruns/OpenNMT-py` step on this machine:
`pyonmttok` (a hard, unconditional import in `onmt/inputters/`) has no wheel for this
Python/platform combination, and `fasttext-wheel` fails to compile against the installed
compiler. Neither is actually exercised by this project's pipeline — it's word-level,
transform-free — so both were satisfied with small compatibility shims rather than real
installs: a ~70-line stand-in for the two `pyonmttok` entry points OpenNMT-py's checkpoint
loading actually calls (`Vocab`, `build_vocab_from_tokens`), verified against the
OpenNMT-py 3.5.1 source to confirm vocab id assignment is order-preserving, not something
`pyonmttok` re-sorts; and a stub `fasttext` module for a language-ID transform this
pipeline never invokes. `torch.load` was monkey-patched to pass `weights_only=False` for
this one, locally-produced, trusted checkpoint. With those two fixes, `onmt_translate` runs
correctly against `gendercheck_step_29000.pt` (the final, step-29000 checkpoint) — this
was not previously possible in this environment and should be treated as a standing
environment note for any future run, not just this one-off.

### 9.2 Method

Full greedy decoding (`beam_size 1`, CPU, `batch_size 8`) of all 3,165 examples in
`opennmt_run/test.src` against `gendercheck_step_29000.pt`, scored against
`opennmt_run/test.tgt`. Sanity-checked first: stripping `<gender>`/`</gender>` from every
gold `test.tgt` line reproduces the corresponding `test.src` line exactly (0/3,165
mismatches) — confirming gold spans are pure insertions with no other edits, so source
token position is a valid coordinate system for both gold and predicted spans wherever
lengths agree. Reproduction tooling (the `pyonmttok`/`fasttext` compat shims from §9.1 and
the scoring script) is checked in at `claude_scripts/opennmt_compat/` and
`claude_scripts/05_evaluate_opennmt.py` — see that script's docstring for the exact
translate + score commands. Raw results: `claude_pipeline_output/approach_b_eval_results.json`.
Metrics computed:

- **Corpus BLEU / chrF / TER** via `sacrebleu`, exactly as `train_config.yaml`'s own header
  comment prescribes.
- **Span-level exact match** precision/recall/F1: a predicted `[start, end)` token span
  counts as a hit only if it exactly matches a gold span at the same source position.
  Only computed for the subset of examples where predicted and source token counts match
  (99.7%, see below) — where they don't, position-based span comparison isn't meaningful.
- **Token-level BIO precision/recall/F1** via `seqeval`, same length-aligned subset.
- **Per-convention recall**, by re-running the exact five regexes from
  `claude_scripts/01_extract_training_data.py` against each gold span's own text to
  classify it post-hoc (paired/asterisk/colon/underscore/Binnen-I), then computing recall
  within each bucket.

### 9.3 Headline numbers

| Metric | Value |
|---|---|
| Test examples | 3,165 |
| Corpus BLEU | 76.2 |
| chrF | 85.6 |
| TER | 9.08% |
| Predicted tokens that are `<unk>` | 8.18% |
| Examples with pred/src token-count mismatch | 9 / 3,165 (0.28%) |
| Gold gender spans | 2,084 |
| Span exact-match precision | 94.7% |
| Span exact-match recall | 90.2% |
| Span exact-match F1 | 92.4% |
| Token-level (seqeval) precision / recall / F1 | 94.8% / 90.3% / 92.5% |

**Per-convention recall** (share of gold spans of that convention exactly recovered):

| Convention | Gold spans in test set | Recall |
|---|---|---|
| Colon (`Lehrer:innen`) | 1,523 | 93.4% |
| Asterisk (`Lehrer*innen`) | 220 | 94.5% |
| Binnen-I (`LehrerInnen`) | 340 | **73.2%** |
| Underscore (`Lehrer_innen`) | 1 | 0% (n=1, not a meaningful sample) |
| Paired (`Lehrer und Lehrerinnen`) | 0 | not present in this test split |

### 9.4 What's actually going on: an out-of-vocabulary problem, not a task-understanding one

The headline span F1 (92.4%) looks strong, and on in-vocabulary text the model is genuinely
good at this task — but the 8.2% `<unk>` rate exposes the real failure mode, and it's
structural, not a training issue. This is a **word-level** vocabulary (50,000 tokens, no
BPE/subword fallback, per `train_config.yaml`), and German's productive compounding means
any sufficiently specific or rare compound noun — which is exactly where many gendered
terms live (`GitarrenheldInnen`, `swb-GaskundInnen`, `IV-RentnerInnen`) — falls outside it.
When that happens the model doesn't just mis-tag the word, it **replaces it with `<unk>`
in the output**, corrupting the sentence beyond the gendering task itself. Concrete
examples pulled from the test set:

```
src:  Doch die neuen Alben von Real Estate, [...] markieren Distanz zu den
      traditionellen GitarrenheldInnen.
pred: Doch die neuen Alben von Real <unk> [...] markieren Distanz zu den
      traditionellen <unk>
```

```
src:  Und schliesslich soll ein allfälliges Erwerbseinkommen von EhegattInnen
      ohne eigenen EL-Anspruch voll [...] als Einnahme verbucht werden.
pred: Und schliesslich soll ein <unk> <unk> von <unk> ohne eigenen <unk> voll
      [...] als Einnahme <unk> werden.
```

This explains the Binnen-I recall gap in §9.3 directly: bare-capital-I gendered nouns
(`GitarrenheldInnen`, `ChristInnen`, `RentnerInnen`) skew toward exactly the kind of long,
specific compounds that fall outside a 50k word-level vocabulary, so this convention takes
disproportionate `<unk>` damage — it's a vocabulary-coverage artifact of this checkpoint,
not evidence that the Binnen-I *signal* is inherently harder to learn (the regex-derived
label itself is no weaker a signal than the others).

There's also a smaller, separate precision issue worth naming honestly: not every error is
`<unk>`-driven. The model occasionally tags spans that don't match any of the five
gendering regexes at all and aren't gendered terms — e.g. flagging `Asyl-Plänen`,
`Registers`, `verfassungsschutz`, `IS-Anhänger` as gendered spans in the test set. These are
genuine false positives (roughly 5% of predicted spans), not an artifact of this scoring
method, and are exactly the kind of error §4.1's confidence-gating design needs to plan for
rather than assume away.

### 9.5 Implications for the rest of this spec

- **§4.1's "reuse the detector as an already-gendered guard" recommendation stands, but
  needs the OOV caveat now added there**: don't trust this checkpoint's span output on a
  sentence containing an out-of-vocabulary word — fall back to "don't touch" for such
  sentences rather than trusting a corrupted decode.
- **§3.3's architecture bake-off should include a subword/BPE-vocabulary variant**, not
  just re-runs of the exact word-level setup used here — the OOV failure mode in §9.4 is a
  direct, predictable consequence of the word-level vocabulary choice documented in
  `train_config.yaml`, and Stage 1 (masculine-span detection, which needs to work on
  arbitrary user-typed compounds, not just corpus-frequent ones) will hit it harder than
  this detection task did.
- **The Binnen-I recall gap is a vocabulary problem, not a signal-strength problem** — worth
  knowing before spending effort on convention-specific model changes that wouldn't
  actually address the root cause.

---

## 10. Milestone 2 results — Stage 1 masculine-span detection bake-off

This fills in milestone 2 from §8: the mined data was relabeled for the actual
autocorrect-trigger task (find the masculine-generic noun that *should* be gendered,
rather than find a span that already is), and both architectures already proven out for
Approach A/B were retrained on it and compared per §3.3's stated criterion.

### 10.1 What was built

- **`claude_scripts/01b_extract_masculine_spans.py`** — same corpus-scanning and
  article-level-split machinery as `01_extract_training_data.py`, but for each regex-matched
  gendered span it computes the degendered masculine form (same substitution logic as
  `scripts/retranslator_4.py`'s `translate_matched_term()`: paired→stem, asterisk/colon/
  underscore→text before the separator, Binnen-I→text before the embedded capital `I`,
  re-implemented against the umlaut-safe patterns rather than retranslator's `\w`-based
  ones), substitutes it into the sentence, and labels *that* position as the positive span.
  Negative (no-gendered-term-found) sentences are kept as real, unmodified `O`-only examples,
  same as before. Label vocabulary is unchanged (`O`/`B-GENDER`/`I-GENDER`) — only what the
  label *means* changes — so every downstream script (`02_train_model.py`, `03_evaluate.py`,
  `04_annotate.py`, `build_opennmt_vocab.py`, `convert_bio_to_opennmt.py`) works unmodified
  against the new data, just pointed at a separate `claude_pipeline_output_stage1/` /
  `opennmt_run_stage1/` directory tree so it's never confused with Approach A/B's data.
- **`claude_scripts/run_stage1_pipeline.sh`** — Stage 1 equivalent of
  `run_corpora_pipeline.sh`, same equal-quota-per-corpus convention.
- **`claude_scripts/06_evaluate_stage1_threshold_sweep.py`** — sweeps the BERT classifier's
  decision threshold to trace a precision/recall curve (per §3.3: "precision at fixed
  recall, not raw F1"), exploiting the fact that every Stage 1 gold span is structurally
  exactly one token (verified at runtime, not assumed), so a token-level threshold sweep is
  a valid stand-in for a span-level PR curve here.
- **Extraction result**: 40,000 examples (32,294 train / 3,856 valid / 3,850 test,
  10,000-positive-per-corpus-source quota across `taz.de` and `www.woz.ch`, matching the
  existing `run_corpora_pipeline.sh` convention), at `claude_pipeline_output_stage1/`.
- **Both architectures retrained on identical data**, both on the same GPU (NVIDIA RTX
  A4000, 16GB — smaller than the RTX 4090 the original Approach B checkpoints were trained
  on; `opennmt_run_stage1/train_config.yaml` documents the resulting batch/accum scale-down):
  - BERT (`deepset/gbert-base` token classifier): 3 epochs / 6,057 steps, ~60 minutes.
  - OpenNMT-py (transformer, same architecture as Approach B): 20,000 steps, ~6.3 hours,
    reached the configured step ceiling without triggering early stopping (patience was at
    8/10 — validation perplexity was still improving, just slowly: 4.49 at the final
    checkpoint).

### 10.2 Headline numbers

| Metric | BERT (gbert-base) | OpenNMT-py (seq2seq) |
|---|---|---|
| Precision | 97.6% | 92.7% (span) / 93.1% (token) |
| Recall | 98.6% | 87.8% (span) / 88.3% (token) |
| F1 | **98.1%** | 90.2% (span) / 90.7% (token) |
| Predicted-token `<unk>` rate | n/a (subword tokenizer, no OOV) | 8.1% |
| Length-aligned test examples | n/a (token classifier, always aligned) | 99.5% (3,829/3,850) |

BERT's numbers are `03_evaluate.py`'s seqeval output at the default argmax operating point;
cross-checked against the independent threshold-sweep script at threshold 0.5
(precision 97.6% / recall 98.7%) — the two agree, which is a useful sanity check that
nothing is inconsistent between the two evaluation paths. OpenNMT's numbers use the same
span-exact-match + seqeval methodology as §9 (`claude_scripts/05_evaluate_opennmt.py`,
unmodified, just pointed at `opennmt_run_stage1/`).

**Precision at fixed recall** (BERT, from `06_evaluate_stage1_threshold_sweep.py` —
directly relevant to §4.1's confidence-gating design, now that §6.1 has settled on a
suggest-style UX rather than auto-apply):

| Target recall | Achieved precision | Actual recall | Threshold |
|---|---|---|---|
| ≥95% | 99.2% | 95.6% | 0.96 |
| ≥90% | 99.4% | 91.9% | 0.99 |
| ≥70–85% | 99.4% (same operating point — 91.9% recall already clears these) | 91.9% | 0.99 |

OpenNMT has no equivalent confidence knob under greedy decoding (beam size / length penalty
are not real substitutes), so it's reported only as the single point in §10.2's table — this
is itself part of why BERT is the better fit for §4.1's design, which explicitly wants a
confidence threshold to gate auto-apply vs. suggest.

### 10.3 What's actually going on

**BERT wins decisively and for a structural reason, not a tuning one.** The OOV corruption
mechanism identified in §9.4 reproduces here at essentially the same rate (8.1% vs. 8.2% in
§9) — expected, since it's the same word-level, no-BPE vocabulary choice, and Stage 1 text
is drawn from the same corpora. But this task is *more* exposed to it than Approach B's
detection task was: Stage 1's positive spans are ordinary masculine nouns (`Koordinator`,
`Aussenminister`, `Medienmacher`), which have no special surface marker (no `:`, `*`, `_`,
capital-I) to help the model — they're exactly as vulnerable to falling outside the 50k-word
vocabulary as any other content word, and once `<unk>`'d, the position simply can't be
recovered. Examples pulled from the test set:

```
src:  Es gibt Koordinator für verschiedene Einheiten: Küche, Medizin, Kleiderlager, Dolmetscher.
pred: Es gibt <unk> für verschiedene <unk> <unk> Medizin, <unk> <unk>
```

```
src:  Dezember werden 1200 Delegierte aus 57 Staaten für die OSZE-Ministerratskonferenz
      anreisen, aus vielen Ländern werden die Aussenminister erwartet.
pred: Dezember werden 1200 Delegierte aus <unk> Staaten für die <gender> <unk> </gender>
      <unk> aus vielen Ländern werden die Aussenminister erwartet.
```

BERT's subword tokenizer has no equivalent failure mode by construction — an unseen word
decomposes into known subword pieces rather than disappearing, which is the direct, causal
reason it doesn't inherit this problem. This isn't a case of BERT being "better tuned"; it's
a task where the architecture-level tokenization choice determines the ceiling, and Stage 1
makes that gap wider than Approach B's task did.

There's also a real, non-OOV-driven false-positive pattern worth naming on both models —
Stage 1 is harder to define precisely than "is this word already gendered" (§9's task),
because "is this masculine noun a person that *should* be gendered" is a genuine judgment
call, and both models occasionally flag non-person nouns that happen to look plausible in
context: OpenNMT flagged `Schutzzölle` (protective tariffs), `Devot` (an adjective, "devout/
submissive"), and `OSZE-Ministerratskonferenz` (a conference, not a person) as candidate
spans. This is exactly the false-positive risk §3.2's Stage 2 (should-this-be-gendered) is
meant to address — Stage 1 was only ever supposed to be a candidate generator, and these
results confirm Stage 2 is a real, necessary stage, not a nice-to-have.

### 10.4 Recommendation

**Carry BERT (`gbert-base` token classifier), not OpenNMT-py, into Stage 2/3.** Per §3.3's
actual selection criterion — precision at fixed recall, not raw F1 — BERT dominates OpenNMT
at every recall level tested, not just on average. Concretely:

- Use BERT for Stage 1 in the eventual plugin pipeline (§4.2's inference service).
- The precision-at-fixed-recall table in §10.2 gives a concrete starting point for §4.1's
  confidence gating. With §6.1 now settled on suggest (not auto-apply), the default 0.5
  threshold (97.6% precision) is a reasonable starting point rather than needing to push
  toward the 0.96–0.99 / >99%-precision end of the curve that an auto-apply design would
  have required — tune from there based on how noisy real suggestion volume feels in use.
- Retire word-level OpenNMT-py as a candidate architecture for Stage 1 specifically. It's
  not that seq2seq modeling is wrong for this project — Approach B's detector is still a
  reasonable production candidate for the different, more surface-marker-driven task it
  solves (§4.1's already-gendered guard) — but for Stage 1's harder, unmarked-token task, a
  word-level vocabulary is a structural liability that a subword tokenizer avoids for free.
  If a seq2seq architecture is wanted for Stage 3 (inflection) later, use a subword/BPE
  vocabulary from the start rather than repeating this experiment.

### 10.5 Reproducing this

- Data: `claude_scripts/run_stage1_pipeline.sh` (or `01b_extract_masculine_spans.py`
  directly) → `claude_pipeline_output_stage1/data_merged/`.
- BERT: `claude_scripts/02_train_model.py claude_pipeline_output_stage1/data_merged
  claude_pipeline_output_stage1/model --base-model deepset/gbert-base --epochs 3
  --batch-size 16`, then `03_evaluate.py` and `06_evaluate_stage1_threshold_sweep.py` against
  the same two directories.
- OpenNMT: `convert_bio_to_opennmt.py` + `build_opennmt_vocab.py` → `opennmt_run_stage1/`,
  then `onmt_train -config opennmt_run_stage1/train_config.yaml` (same `pyonmttok`/
  `torch.load` compat workarounds as §9.1 apply), then `onmt_translate` against
  `opennmt_run_stage1/test.src`, then `05_evaluate_opennmt.py` against the three
  `opennmt_run_stage1/test.*` files — identical usage to §9, just a different data directory.
- Raw results: `claude_pipeline_output_stage1/bert_threshold_sweep_results.json`,
  `claude_pipeline_output_stage1/opennmt_eval_results.json`.

### 10.6 Retroactive finding: compound-continuation contamination in the mined data (affects §10 and §11, not §9)

Discovered while building Stage 3 (§12): the `GENDER_PATTERNS` regexes require only "≥2
lowercase characters" after the gendering marker, with no upper bound — so on a compound
like `Richter:innenwahl` ("judges' election"), the match swallows the whole compound tail,
not just `Richter:innen`. `01b_extract_masculine_spans.py`'s degendering then replaces the
*entire* matched span with the bare stem, silently dropping the tail: `Richter:innenwahl`
degenders to `Richter`, not `Richter Wahl` or `Richterwahl` — the word "wahl" just
disappears from the training sentence.

Measured directly against Stage 1's actual consumed training articles (not a sample):
**10,303 of 23,363 gendered matches (44.1%) had a compound tail past the core `innen`/`in`
suffix.** This is high in raw-match terms but concentrated in a much smaller number of
*distinct* repeated strings — site-chrome boilerplate (`Richter:innenwahl` as a repeated
topic-tag link, `Nutzer:innenmenu` as a repeated nav label) accounts for a large share of
it, not diverse article prose.

This does **not** invalidate §10/§11's reported metrics — training and gold test labels
were built by the same process consistently, so precision/recall still correctly measure
"does the model reproduce this labeling process," and BERT's win over OpenNMT-py holds
regardless (both were trained and scored against the same data). What it does mean:

- A meaningful share of Stage 1/2's "positive" training sentences have a dropped word,
  which is a genuine (if likely minor, given the boilerplate concentration) training-data
  quality issue not worth a full retrain given the results already validated on real
  held-out data, but worth fixing before further iteration — `01b_extract_masculine_spans.py`
  should adopt the same core-suffix truncation logic now used in
  `11a_build_inflection_pairs.py`'s `core_gendered_form()` rather than replacing the whole
  raw match.
- More importantly for the real product: **German's productive compounding means a person
  noun is very often not a standalone word but the first element of a longer compound**
  (`Richterwahl`, `Lehrerzimmer`, `Nutzerkonto`). Gendering one of these correctly means
  inserting the suffix *mid-compound* (`Richter:innenwahl`), not replacing the whole
  compound. Stage 1/2 as built only ever learned "is this whole token a candidate," with no
  notion of where inside a compound to insert — this is a real, previously-unaddressed gap
  for §4's plugin, not just a data-cleanliness footnote. Flagging as an open item for
  Stage 1/2's next iteration and for §4's design, not resolving it here.

---

## 11. Milestone 3 results — Stage 2 (should-this-be-gendered) hard-negative mining

This fills in milestone 3 from §8. §10.3's own bake-off already surfaced concrete evidence
that Stage 1 needs a precision-focused second pass — both architectures flagged real
non-person nouns (`Schutzzölle`, `Devot`, `OSZE-Ministerratskonferenz`) as gendering
candidates. This section builds and evaluates that pass.

### 11.1 Design: real Stage 1 false positives beat synthetic near-misses

§3.2's original text proposed mining hard negatives from "sentences the regex almost
matched but didn't." Once Stage 1 actually existed and was evaluated (§10), a better source
became available and was used instead: **Stage 1's own real false positives on fresh text**.
A regex near-miss tells you where the *regex* was nearly fooled; Stage 1's actual false
positives tell you where the *trained model* actually gets fooled, which is a more direct
target and doesn't require guessing what "almost matches" should mean for a neural model
rather than a hand-written pattern.

Concretely:

1. **`claude_scripts/07a_build_confirmed_stems.py`** — a full, uncapped scan of every file
   in both corpora (unlike Stage 1's `--max-positive`-capped extraction) collecting every
   word ever observed as the masculine base of a real mined gendered term. This is the
   ground-truth oracle for the next step: **6,258 confirmed stems**, a strict superset of
   what Stage 1 actually trained on.
2. **`claude_scripts/07b_mine_stage2_hard_negatives.py`** — reproduces Stage 1's exact
   `collect_articles()` call (same `--max-positive 10000 --neg-ratio 1.0 --seed 42` as
   `run_stage1_pipeline.sh`) to get the precise set of article paths Stage 1 trained on,
   then runs the trained Stage 1 model over **fresh, never-seen articles** and checks every
   flagged span against the confirmed-stems set:
   - flagged word **is** a confirmed stem → bonus positive (a real candidate Stage 1's
     10k-per-source cap happened not to include)
   - flagged word **is not** a confirmed stem → hard negative (Stage 1 was wrong to flag
     it — kept with all-`O` gold labels)

   Mining scanned **246,312 fresh sentences** (all from `taz.de`'s remaining, unconsumed
   corpus — the 3,000-hard-negative target was reached before `www.woz.ch`'s fresh portion
   was needed) and yielded **3,001 hard negatives** and **5,368 bonus positives** — a ~1.2%
   hard-negative yield rate, consistent with Stage 1's already-high (97.6%) precision:
   finding its mistakes takes scanning a lot of text.
3. **`claude_scripts/08_build_stage2_dataset.py`** — merges Stage 1's original data with
   the mined hard negatives/bonus positives. The miner also emitted a large pool of plain
   true negatives (47,821 — sentences where Stage 1 predicted nothing at all); left
   uncapped these would have outnumbered the actual hard negatives 16 to 1 in training,
   which is exactly the "not just random negatives … too easy" failure §3.2 warned about.
   Capped to 3× the hard-negative count per split before merging.
4. **`claude_scripts/09_evaluate_stage2_hard_negatives.py`** and
   **`10_check_recall_retention.py`** — the two evaluations that actually answer "did this
   work," see §11.3.

Final Stage 2 training set: **46,051 examples** (`stage1`=32,294, `stage2_bonus_pos`=4,289,
`stage2_hard_neg`=2,367, `stage2_true_neg`=7,101 — capped from 38,262). Retrained with the
identical `gbert-base` architecture and hyperparameters as Stage 1 (`02_train_model.py`,
3 epochs, batch 16).

### 11.2 Headline numbers on the standard (merged) test set

| Metric | Value |
|---|---|
| Test examples | 5,680 |
| Precision | 94.4% |
| Recall | 96.7% |
| F1 | 95.5% |

This is *lower* than Stage 1's own headline numbers (§10.2: 97.6%/98.6%/98.1%) — that is
expected and not a regression. Stage 2's test set is deliberately harder: it's the same
Stage 1 test examples plus the held-out portion of hard negatives and bonus positives
specifically chosen because they're hard. Comparing raw F1 across two differently-difficulty
test sets would be misleading; §11.3 is the metric that actually answers whether Stage 2
achieved its goal.

**Precision at fixed recall** (same methodology as §10.2, `06_evaluate_stage1_threshold_sweep.py`):

| Target recall | Achieved precision | Actual recall | Threshold |
|---|---|---|---|
| ≥95% | 96.5% | 95.1% | 0.81 |
| ≥90% | 97.8% | 90.9% | 0.95 |
| ≥85% | 98.3% | 86.5% | 0.98 |
| ≥80% | 98.8% | 82.3% | 0.99 |

Unlike Stage 1's curve (§10.2, which plateaued flat below 90% recall), this one keeps
trading precision for recall gradually across the whole range — expected, since the test
set now specifically contains the harder cases that make that trade-off real instead of
free.

### 11.3 The actual result: did hard-negative mining work?

**Yes, substantially, and without materially damaging recall on ordinary cases.**
`09_evaluate_stage2_hard_negatives.py` evaluates the Stage 1 model and the new Stage 2
model on exactly the 322 hard-negative examples held out in the mining's own test split —
real sentences where Stage 1 is *known* to be wrong, on fresh text neither model's training
data included:

| Model | Still flags a false positive on these 322 known-hard cases |
|---|---|
| Stage 1 | 322/322 (**100.0%**) — true by construction, confirms the test set is valid |
| Stage 2 | 94/322 (**29.2%**) |

**70.8% relative reduction in the false-positive rate on cases specifically chosen because
Stage 1 got them wrong**, measured on genuinely held-out data (these 322 test-split examples
were never in Stage 2's training data — only the 2,367 train-split hard negatives were).

`10_check_recall_retention.py` checks the other side of the trade — isolating the 2,105
*original* Stage 1 positive examples in the test set (tagged `source=='stage1'`, excluding
Stage 2's own mined additions) and comparing recall:

| Model | Recall on original Stage 1 positives |
|---|---|
| Stage 1 | 98.8% |
| Stage 2 | 97.5% |

A **1.2-point recall cost** for a 70.8% relative cut in known-hard false positives is a
favorable trade for an autocorrect feature, where §3.2 already established false positives
are more costly than false negatives.

### 11.4 What's actually going on — including a real limitation of the method

Looking at what Stage 2 fixed and what it didn't is informative. Genuinely fixed: mining
directly targeted non-person nouns and adjectives that happened to look plausible
(`Schutzzölle`, `Devot`-type errors from §10.3's original findings are exactly the pattern
this stage was built to catch).

But not everything in the "still wrong" 29.2% is actually a model failure — some of it is a
**limitation of the confirmed-stems oracle itself**. A few of Stage 2's remaining errors on
the held-out hard-negative set:

```
Staatsanwalt   (prosecutor — a completely legitimate person noun; it's a "hard negative"
                only because "Staatsanwält:innen" never happened to appear in gendered
                form anywhere in these two corpora, not because it shouldn't be gendered)
Anhänger       ("hatten ... viele Anhänger" — here plausibly means "supporters/followers,"
                a legitimate person-noun sense; "Anhänger" is also the word for a trailer,
                so this is a genuinely ambiguous case, not a clean miss)
Gehalter       (salaries — genuinely not a person noun; this one is a real, correctly-
                identified-as-wrong Stage 1 error that Stage 2 still didn't fully suppress)
```

The confirmed-stems set can only certify a word as "real" if it happened to appear gendered
*somewhere in these specific corpora* — it has no way to certify a word as "not real," so
every word absent from it gets treated as a hard negative regardless of whether that's
because it's genuinely not a person noun (`Gehalter`) or just because this particular
corpus never happened to gender it (`Staatsanwalt`). This means an unknown fraction of the
3,001 mined "hard negatives" are actually label noise — real candidates mislabeled as
negatives — which caps how clean this signal can be without a human-reviewed negative set.
The 70.8% reduction is a real result on real held-out data, but the ceiling here is probably
lower than 0%, not because Stage 2 can't learn better, but because part of the "test" is
measuring an oracle gap rather than a model gap.

### 11.5 Recommendation

**Use `claude_pipeline_output_stage2/model` as the production candidate-detection model
going forward, not Stage 1's.** It strictly dominates on the metric that matters most for an
autocorrect feature (false-positive suppression) at a small, well-characterized recall cost,
and it was built as a genuine refinement of Stage 1 rather than a replacement trained from
scratch — same architecture, same base data, plus the targeted hard negatives. Concretely:

- §4.1's confidence-gating design should use §11.2's precision-at-fixed-recall table
  (not §10.2's Stage 1 table), with §6.1's suggest-not-auto-apply decision meaning the
  default 0.5-threshold operating point is the right starting point, not the conservative
  end of the curve.
- §4.1's "already-gendered guard" reuse recommendation is unaffected — that's Approach B
  (§9), a different model for a different task.
- Before investing further in cleaning up the remaining 29.2%, note §11.4's oracle-noise
  caveat: a next iteration would benefit more from a small amount of human-reviewed hard
  negatives (even a few hundred, manually confirmed as genuinely-not-person-nouns) than
  from mining more automatically, since the current ceiling is partly an artifact of the
  labeling method, not the model.

### 11.6 Reproducing this

```
python claude_scripts/07a_build_confirmed_stems.py corpora/taz.de corpora/www.woz.ch \
    --output claude_pipeline_output_stage2/confirmed_stems.json

python claude_scripts/07b_mine_stage2_hard_negatives.py \
    --stage1-model claude_pipeline_output_stage1/model \
    --confirmed-stems claude_pipeline_output_stage2/confirmed_stems.json \
    --output-dir claude_pipeline_output_stage2/hard_mined \
    --target-hard-negatives 3000

python claude_scripts/08_build_stage2_dataset.py \
    --stage1-dir claude_pipeline_output_stage1/data_merged \
    --hard-mined-dir claude_pipeline_output_stage2/hard_mined \
    --output-dir claude_pipeline_output_stage2/data_merged

python claude_scripts/02_train_model.py claude_pipeline_output_stage2/data_merged \
    claude_pipeline_output_stage2/model --base-model deepset/gbert-base --epochs 3 --batch-size 16

python claude_scripts/03_evaluate.py claude_pipeline_output_stage2/data_merged claude_pipeline_output_stage2/model
python claude_scripts/06_evaluate_stage1_threshold_sweep.py claude_pipeline_output_stage2/data_merged claude_pipeline_output_stage2/model
python claude_scripts/09_evaluate_stage2_hard_negatives.py \
    --hard-neg-test claude_pipeline_output_stage2/hard_mined/test.jsonl \
    --stage1-model claude_pipeline_output_stage1/model --stage2-model claude_pipeline_output_stage2/model
python claude_scripts/10_check_recall_retention.py \
    --test-set claude_pipeline_output_stage2/data_merged/test.jsonl \
    --stage1-model claude_pipeline_output_stage1/model --stage2-model claude_pipeline_output_stage2/model
```

Raw results: `claude_pipeline_output_stage2/threshold_sweep_results.json`,
`claude_pipeline_output_stage2/hard_negative_before_after.json`,
`claude_pipeline_output_stage2/original_positive_recall_retention.json`.

---

## 12. Milestone 4 results — Stage 3 (case-aware inflection)

This fills in milestone 4 from §8: given a masculine candidate span (Stage 1/2's job),
produce the correctly inflected gendered surface form.

### 12.1 The premise didn't survive contact with the data — and that simplified everything

§3.2 framed this as "case-aware inflection," assuming the gendered suffix needs to agree
with the sentence's grammatical case (nominative/accusative/dative/genitive) the way normal
German plural nouns do (e.g. dative plural `den Lehrern`). **This assumption is false.** A
targeted search across the corpus for gendered forms immediately preceded by a
dative-triggering determiner (`den`, `allen`, `diesen`, `jenen`, ...) found **41 real
examples, zero of which case-inflect the suffix**:

```
den Gamer:innen        den Lehrer:innen        den Historiker:innen
den Aktivist:innen      allen Mitbewerber:innen  den Vertreter:innen
den Armenier:innen      den Genoss:innen         den Musiker:innen
```

Not `den Lehrern:innen`, `den Historikern:innen`, etc. — real usage treats the suffixed
form as **invariant with respect to grammatical case**, consistent with every qualitative
example pulled anywhere else in this document (e.g. §9.3's `"unseren Leser:innen"`, dative,
uninflected). Grammatical case is carried by the surrounding article/determiner, not the
gendered noun itself — apparently a general property of how these relatively recent
orthographic conventions get used, not an oversight in this specific corpus.

Consequence: Stage 3 does not need case detection or normalization at all. It reduces to
attaching a suffix to a stem — the "case-aware" part of the milestone's name turned out to
be based on a wrong assumption, discovered by checking rather than by building the more
complex thing first.

### 12.2 Method

- **`claude_scripts/11a_build_inflection_pairs.py`** — full, uncapped scan of both corpora
  (10,100 unique (stem, convention, form) records from 370,401 raw matches) collecting real
  `(stem, convention, form)` triples with frequency counts — the empirical ground truth for
  both the rule and its evaluation. Also fixes a data-quality issue discovered in the
  process and documented retroactively in §10.6: raw regex matches often swallow a
  following compound continuation (`Richter:innenwahl`, not just `Richter:innen`); this
  script's `core_gendered_form()` truncates each match at the `innen`/`in` suffix boundary
  to recover the clean target form regardless of what compounds onto it.
- **`claude_scripts/11b_stage3_inflect.py`** — the rule itself: `stem + separator + "innen"`
  for colon/asterisk/underscore, `stem + "Innen"` for Binnen-I (direct concatenation, no
  separator), with a `singular=True` option for the `"in"`/`"In"` singular forms also
  observed in real data (e.g. `Mitarbeiter:in`). `paired` (`"X und Xinnen"`) is included for
  completeness but not tuned for — §10.2 already found 0 gold spans of this convention in
  Stage 1's entire test set, so it's negligible in real usage.
- **`claude_scripts/11c_evaluate_stage3_coverage.py`** — splits by *stem* (not by
  individual record, to prevent the same word appearing on both sides) into 5,000 train /
  1,250 test stems, and checks how often the rule exactly reproduces each held-out test
  record's real `core_form`. Reports both type-level (each unique pair counted once) and
  frequency-weighted (weighted by real corpus frequency — the number that reflects
  real-world impact) accuracy.

### 12.3 Coverage results

| Convention | Type-level accuracy | Frequency-weighted accuracy |
|---|---|---|
| Colon | 99.1% (525/530) | 99.9% (3,523/3,528) |
| Binnen-I | 99.1% (843/851) | 99.8% (6,755/6,771) |
| Asterisk | 96.4% (489/507) | 98.7% (1,980/2,007) |
| Underscore | 78.6% (33/42) | 83.7% (72/86) |
| **Overall** | **97.9% (1,890/1,930)** | **99.5% (12,330/12,392)** |

99.5% frequency-weighted accuracy on genuinely held-out stems (never seen while building or
tuning the rule — there was nothing to tune, but the split is still honest) is a very high
bar for a ~15-line rule.

### 12.4 What the failures actually are — mostly not what milestone 4 anticipated

Milestone 4 anticipated the rule's gap would be genuine morphological irregularity (umlaut
shifts, weak nouns, suppletive plurals). **It mostly isn't.** Inspecting the actual highest-
frequency failures:

```
'Fatal'    [binnen_i]   rule='FatalInnen'    actual='FatalImpact'       -- not gendering at all
'Martin'   [underscore] rule='Martin_innen'  actual='Martin_meint'      -- a username, not gendering
'Trans'    [asterisk]   rule='Trans*innen'   actual='Trans*gender'      -- "Trans*" is an unbound
                                                                            inclusive-language prefix,
                                                                            not an occupation noun +
                                                                            suffix -- different
                                                                            grammatical category
'Wähler'   [asterisk]   rule='Wähler*innen'  actual='Wähler*nnen'       -- looks like a source typo
                                                                            (missing "i")
```

Almost every failure is either **a regex false-positive** (the mining pattern matched
something that isn't gendering at all — a proper noun, a username, a URL fragment) or a
**genuinely different construction** (`Trans*` as an identity-prefix rather than an
occupation-suffix pattern) — not a case where a real gendered term exists and the rule
computes the wrong inflected form. This explains why underscore is the weak convention here
too (78.6%/83.7%, well below the other three): underscores are common in usernames, file
paths, and URL slugs, so that regex has the most opportunities to match non-gendering text
in the first place. This is a mining-precision issue shared with §9's regex-matching step
generally, not something specific to inflection.

**Net implication:** there is essentially no morphological inflection problem left to solve
here. The stem, as mined directly from real corpus text, already carries whatever
irregularity the original German word has (the corpus IS the morphology reference) — the
rule just re-attaches the same suffix pattern the corpus already demonstrated for that stem.

### 12.5 Recommendation

**No learned fallback.** The measured gap is small, and what gap exists is explained by
upstream regex mining noise, not rule inadequacy — building a model to close it would be
optimizing the wrong stage. If mining precision is worth improving later, prioritize
tightening the `underscore` pattern specifically (e.g., requiring the matched span not be
adjacent to characters common in usernames/URLs) over anything in Stage 3 itself.

`claude_scripts/11b_stage3_inflect.py`'s `inflect(stem, convention, singular=False)` is
ready to use as-is as the Stage 3 component of §4.2's inference service: Stage 1/2 supplies
the candidate stem, `11b` supplies the surface form for the user's configured style (§4.1's
per-user style setting, defaulting to `convention='colon'` per §6.2).

### 12.6 Reproducing this

```
python claude_scripts/11a_build_inflection_pairs.py corpora/taz.de corpora/www.woz.ch \
    --output claude_pipeline_output_stage3/inflection_pairs.json

python claude_scripts/11c_evaluate_stage3_coverage.py \
    --pairs claude_pipeline_output_stage3/inflection_pairs.json \
    --out claude_pipeline_output_stage3/coverage_results.json
```

`claude_scripts/11b_stage3_inflect.py` has no training step — it's a pure function, usable
directly as a library (`from stage3_inflect import inflect`) or via its CLI for manual
checks. Raw results: `claude_pipeline_output_stage3/inflection_pairs.json`,
`claude_pipeline_output_stage3/coverage_results.json`.

---

## 13. Milestone 6 results — the actual plugin

This fills in milestone 6 from §8: a working LibreOffice extension wired to the real
Stage 1/2/3 pipeline, replacing the `"foo"`→`"bar"` stub in `apps/poc_app_1/`.

### 13.1 Architecture pivot: LibreOffice's grammar-checker service, not `XKeyListener`

§4's original design (written before §6.1 was decided) sketched a custom `XKeyListener`
that would buffer keystrokes and silently rewrite text, with its own undo/reversibility
handling. Now that §6.1 has settled on **suggest, not auto-apply**, that design is solving
the wrong problem: LibreOffice already has a purpose-built extensibility point for exactly
"flag a span, offer a suggestion, let the user accept or ignore via right-click" —
`com.sun.star.linguistic2.XProofreader`, the same interface LibreOffice's own bundled
LanguageTool grammar-checker integration uses. Using it means getting the squiggly
underline, the right-click suggestion menu, and the ignore/accept interaction entirely for
free from LibreOffice's existing linguistic framework, instead of building and maintaining
custom overlay UI. This supersedes §4.1's `XKeyListener`/undo-integration design; §4.2's
inference-boundary decision (don't load a transformer inside LO's bundled Python) stands
unchanged.

### 13.2 What was built

`apps/gendercheck_plugin/`:

- **`inference_server.py`** — stdlib-only local HTTP service (no Flask/FastAPI dependency,
  so it's trivial to launch). Loads the Stage 2 model recommended in §11.5
  (`claude_pipeline_output_stage2/model`) for candidate detection and
  `claude_scripts/11b_stage3_inflect.py` for the suggested surface form, defaulting to
  `convention='colon'` per §6.2. `POST /check {"text": ...}` returns candidate spans with
  character offsets, the suggested gendered form, and a confidence score.
- **`python/gendercheck_proofreader.py`** — the UNO component. Implements `XProofreader`,
  `XServiceInfo`, `XInitialization`; queries the inference server per sentence and maps
  each candidate to a `SingleProofreadingError` (grammar-error markup type, one suggestion:
  the gendered form). Degrades to zero errors (not an exception) if the inference server is
  unreachable — a missing suggestion is a much smaller failure than a broken grammar
  checker.
- **`registry/data/org/openoffice/Office/Linguistic.xcu`** — registers the component as a
  German (`de`, `de-AT`, `de-DE`, `de-CH`) grammar checker. Its schema was **not** guessed
  from documentation — verified directly against LibreOffice's own bundled registration for
  its built-in LanguageTool integration, found at
  `/usr/lib64/libreoffice/share/registry/lingucomponent.xcd`
  (`ServiceManager/GrammarCheckers/org.openoffice.lingu.LanguageToolGrammarChecker`), which
  is the authoritative real-world example of this exact, sparsely-documented registration
  format.
- `META-INF/manifest.xml`, `description.xml` — packaged as `.oxt` following the existing
  `foo2bar/` template from §4.3.

### 13.3 Verification actually performed

Built and installed into a real, fresh LibreOffice profile (`unopkg add`), then verified
against a live LibreOffice process via genuine UNO calls — not just read for plausibility:

1. **Component loads and self-identifies correctly**: `createInstanceWithContext("org.gendercheck.Proofreader", ctx)` succeeds and reports the expected interfaces
   (`XInitialization`, `XServiceInfo`, `XProofreader`, `XTypeProvider`).
2. **LibreOffice's own linguistic framework discovers it**: `LinguServiceManager.getAvailableServices("com.sun.star.linguistic2.Proofreader", Locale(de-DE))` returns
   `['org.gendercheck.Proofreader']` — this is LibreOffice's own registry confirming the
   `Linguistic.xcu` registration actually took effect, not just that the file was well-formed.
3. **End-to-end `doProofreading()` call, through the identical interface LibreOffice's own
   grammar-check pipeline calls**: given `"Die Lehrer kommen morgen zur Konferenz."`,
   returned exactly one error — `Lehrer` at the correct character offsets `[4:10]`, with
   suggestion `Lehrer:innen` and a confidence-annotated comment. This exercised the full
   real chain: UNO component → HTTP → Stage 2 BERT model → Stage 3 rule.

**Update — root-caused and fixed; the interactive path now works end-to-end,
screenshot-verified.** A later session tested the interactive path fully — installed the
extension into the real (non-test) LibreOffice profile, installed a German (`de_DE`)
hunspell dictionary (`dnf download` to fetch the RPM without root, extracted with
`rpm2cpio`/`cpio`, packaged as a second per-user `.oxt` dictionary extension, since none was
present and the combined Spelling/Grammar dialog refuses to run at all without one for the
document's language), and drove real screenshots of the actual GUI.

Initially, running `.uno:SpellingAndGrammarDialog` against real German text containing
"Lehrer" completed with **"The spellcheck is complete," flagging nothing** — reproduced
across three separate attempts, including a fresh document — despite registration being
confirmed correct via `LinguServiceManager.getAvailableServices()`. Instrumenting the
inference server with request logging confirmed `doProofreading()` was never called at
all: zero HTTP requests reached it during any dialog run.

**Root cause, found via web search rather than further blind trial-and-error**: a
[real-world account of building a Python `XProofreader` extension](https://keithcu.com/wordpress/?p=5276)
revealed that LibreOffice's actual internal linguistic framework instantiates registered
checkers via `createInstanceWithArgumentsAndContext` — a call site that passes extra
constructor arguments even when none are configured. The manual verification test earlier
in this section used the simpler `createInstanceWithContext` (single `ctx` argument) to
instantiate the component directly, which is why it succeeded while LibreOffice's real
invocation path silently failed: `GendercheckProofreader.__init__(self, ctx)` and the
`createInstance(ctx)` factory function both only accepted one argument, so the real call
with extra arguments would fail before ever reaching `doProofreading()` — and LibreOffice
apparently drops a checker that fails to instantiate from its active set without
surfacing any visible error, rather than crashing or logging.

**Fix**: changed both signatures to `__init__(self, ctx, *args)` and
`createInstance(ctx, *args)`, silently accepting and ignoring the extra arguments.
Rebuilt, reinstalled, relaunched — the same test this time showed real `POST /check`
requests reaching the inference server, and a screenshot confirms the complete, correct
UI: "Lehrer" underlined in the document, and the Spelling/Grammar dialog showing
*"'Lehrer' could be gendered as 'Lehrer:innen' (confidence 99%)"* with "Lehrer:innen"
offered as the correction. This is the plugin working exactly as designed, through
LibreOffice's real interactive UI, not a simulated or partial confirmation.

**A real bug found and fixed during testing, not just theorized**: the first GUI attempt
correctly demonstrated *why* this verification mattered — the test document defaulted to
English (USA) as its language, so LibreOffice's own English spell-checker fired instead of
the (German-only, per `getLocales()`) Gendercheck grammar checker, flagging "Lehrer" as a
misspelling with an unrelated suggestion ("Luehrer"). Fixed by setting `CharLocale` to
`de-DE` on the inserted text run. This is also a real deployment consideration worth noting
for §14 (not just a test artifact): the plugin does nothing on a document whose language
isn't set to a supported German variant — normal for a German-language document, but worth
a clear failure mode (or a language-mismatch notice) rather than silent inactivity if this
ever confuses a user.

### 13.4 Known gaps carried forward from earlier sections, now concretely observed

Testing surfaced one gap predicted only in passing earlier and now directly confirmed: the
inference server did **not** flag `"Studenten"` (dative plural of `Student`) in the same
test sentence that correctly flagged `"Lehrer"`. This matches the concern implicit in
§12.1's finding — Stage 1/2's training data, built by degendering already-invariant
gendered forms, never included genuinely case-inflected masculine occurrences as positive
examples, so recall on inflected forms like dative plurals is untested and, on this one
example, weaker. Not fixed here; noted for a future data/eval pass specifically targeting
case-inflected input, separate from §12's (correct, and unaffected) finding that the
*output* suffix itself doesn't need case-awareness.

Not yet wired into this plugin (tracked, not silently dropped):
- The Approach B "already-gendered" guard from §4.1/§9 (skip spans that are already
  gendered) — this MVP relies on Stage 1/2 alone, which was trained to not flag
  already-gendered text but wasn't given the dedicated guard model as a second check.
- The §10.6 mid-compound insertion gap (`Richterwahl` → `Richter:innenwahl`) — this plugin
  currently only handles whitespace-delimited whole-word candidates.
- Per-user style configuration (§4.1) — convention is hardcoded to `colon` via
  `inference_server.py --convention`, not yet exposed as a LibreOffice settings UI.

### 13.5 Reproducing/running this

```
# 1. Start the inference server (from apps/gendercheck_plugin/)
python3 inference_server.py --model ../../claude_pipeline_output_stage2/model \
    --stage3-dir ../../claude_scripts --port 8765

# 2. Build and install the extension
zip -r gendercheck.oxt META-INF description.xml description-en.txt registry python
unopkg add -f gendercheck.oxt        # add --shared for a system-wide install

# 3. Open a German-language document in LibreOffice Writer and run
#    Tools > Language > Grammar Checker (or wait for automatic background checking)
```
