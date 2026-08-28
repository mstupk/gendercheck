#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path

def apply_bpe(input_file, output_file, codes_file, vocab_file, threshold=50):
    """Apply BPE processing to a file using subword-nmt"""
    try:
        with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
            result = subprocess.run(
                [
                    'subword-nmt', 'apply-bpe',
                    '-c', codes_file,
                    '--vocabulary', vocab_file,
                    '--vocabulary-threshold', str(threshold)
                ],
                stdin=inf,
                stdout=outf,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error processing {input_file}:")
                print(result.stderr)
                return False

        return True
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Apply BPE to source and target files')
    parser.add_argument('--input_dir', required=True, help='Directory containing source/target files')
    parser.add_argument('--codes_file', required=True, help='BPE codes file')
    parser.add_argument('--src_vocab', required=True, help='Source vocabulary file')
    parser.add_argument('--tgt_vocab', required=True, help='Target vocabulary file')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Vocabulary threshold (default: 50)')

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a valid directory")
        return

    # Process all source and target files
    processed_count = 0
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            # Determine file type and vocabulary
            if file_path.suffix == '.src':
                vocab_file = args.src_vocab
            elif file_path.suffix == '.trg':
                vocab_file = args.tgt_vocab
            else:
                continue  # Skip non-source/target files

            # Create output filename (add .bpe before extension)
            output_file = file_path.with_name(
                f"{file_path.stem}.bpe{file_path.suffix}"
            )

            # Apply BPE
            if apply_bpe(
                str(file_path),
                str(output_file),
                args.codes_file,
                vocab_file,
                args.threshold
            ):
                print(f"Processed: {file_path.name} → {output_file.name}")
                processed_count += 1

    print(f"\nProcessing complete! Created {processed_count} BPE files.")

if __name__ == "__main__":
    main()
