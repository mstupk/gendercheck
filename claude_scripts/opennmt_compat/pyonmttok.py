"""Minimal stand-in for the pyonmttok C++ extension (unavailable on this platform:
no prebuilt wheel for this Python/torch combination, see CLAUDE.md).

Only implements what onmt.inputters actually needs for loading a checkpoint's
already-built, order-preserving word-level vocabulary and doing plain
whitespace tokenisation. Verified against OpenNMT-py 3.5.1 source
(onmt/inputters/inputter.py: _read_vocab_file / dict_to_vocabs / build_vocab)
which treats vocab files/checkpoint vocab lists as pre-ordered token lists,
not something pyonmttok re-sorts -- id == list index.
"""


class Vocab:
    def __init__(self, tokens):
        self.ids_to_tokens = list(tokens)
        self._token_to_id = {t: i for i, t in enumerate(self.ids_to_tokens)}
        self.default_id = self._token_to_id.get("<unk>", 0)

    def __len__(self):
        return len(self.ids_to_tokens)

    def __contains__(self, token):
        return token in self._token_to_id

    def __getitem__(self, token):
        return self._token_to_id.get(token, self.default_id)

    def __iter__(self):
        return iter(self.ids_to_tokens)

    def lookup_token(self, token):
        return self._token_to_id.get(token, self.default_id)

    def lookup_index(self, idx):
        return self.ids_to_tokens[idx]

    def __call__(self, tokens):
        return [self._token_to_id.get(t, self.default_id) for t in tokens]

    def add_token(self, token):
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self.ids_to_tokens)
            self.ids_to_tokens.append(token)
        return self


def build_vocab_from_tokens(tokens, maximum_size=0, special_tokens=None, minimum_frequency=0):
    toks = list(tokens)
    seen = set(toks)
    for t in (special_tokens or []):
        if t not in seen:
            toks.append(t)
            seen.add(t)
    if maximum_size and maximum_size > 0:
        toks = toks[:maximum_size]
    return Vocab(toks)


class Tokenizer:
    """Whitespace tokenizer stand-in -- sufficient since train_config.yaml
    specifies word-level tokenisation with no transforms."""

    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, text):
        tokens = text.split()
        return tokens, None

    def detokenize(self, tokens, *args, **kwargs):
        return " ".join(tokens)
