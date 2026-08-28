#!/usr/bin/env python3
import os
import argparse
import yaml
from pathlib import Path

def find_bpe_files(data_dir, language):
    """Find BPE-processed files for a given language"""
    lang_dir = Path(data_dir) / language
    if not lang_dir.exists():
        return None, None, None

    train_files = list(lang_dir.glob('*train*.BPE'))
    valid_files = list(lang_dir.glob('*valid*.BPE'))
    test_files = list(lang_dir.glob('*test*.BPE'))

    train = str(train_files[0]) if train_files else None
    valid = str(valid_files[0]) if valid_files else None
    test = str(test_files[0]) if test_files else None

    return train, valid, test

def main():
    parser = argparse.ArgumentParser(description='Generate YAML config for NMT training')
    parser.add_argument('--data_dir', required=True,
                        help='Directory with BPE-processed data')
    parser.add_argument('--languages', required=True,
                        help='Comma-separated languages (e.g., src,trg)')
    parser.add_argument('--output', default='config.yaml',
                        help='Output YAML filename')
    parser.add_argument('--run_name', default='example_run',
                        help='Name for this training run')

    args = parser.parse_args()
    languages = args.languages.split(',')

    if len(languages) < 2:
        raise ValueError("At least two languages required (source and target)")

    src_lang = languages[0]
    tgt_lang = languages[1]

    # Create output directory structure
    run_dir = Path(args.run_name) / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "model"
    model_dir.mkdir(exist_ok=True)

    # Find BPE-processed files
    src_train, src_valid, _ = find_bpe_files(args.data_dir, src_lang)
    tgt_train, tgt_valid, _ = find_bpe_files(args.data_dir, tgt_lang)

    if not src_train or not tgt_train:
        raise FileNotFoundError("Could not find training files in BPE output directory")

    # Vocabulary files
    src_vocab = str(Path(args.data_dir) / f"vocab.{src_lang}")
    tgt_vocab = str(Path(args.data_dir) / f"vocab.{tgt_lang}")

    # Build configuration dictionary
    config = {
        'save_data': str(run_dir / "example"),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'overwrite': False,
        'data': {
            'corpus_1': {
                'path_src': src_train,
                'path_tgt': tgt_train
            },
            'valid': {
                'path_src': src_valid if src_valid else "",
                'path_tgt': tgt_valid if tgt_valid else ""
            }
        },
        'world_size': 1,
        'gpu_ranks': [0],
        'save_model': str(model_dir / "model"),
        'save_checkpoint_steps': 500,
        'train_steps': 1000,
        'valid_steps': 500
    }

    # Write YAML file
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration file generated: {args.output}")
    print(f"Run directory created at: {run_dir}")

if __name__ == "__main__":
    main()
