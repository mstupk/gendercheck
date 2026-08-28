"""Stub: the real fasttext-wheel package fails to build here (old C++ source
incompatible with the system compiler). Only onmt/transforms/clean.py imports
this at module level, for a language-ID transform this project's word-level,
transform-free pipeline never uses."""

def load_model(*args, **kwargs):
    raise NotImplementedError("fasttext stub -- language-ID transform not used by this pipeline")
