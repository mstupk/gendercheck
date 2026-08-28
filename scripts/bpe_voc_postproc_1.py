#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def process_file(file_path):
    """Process a single file: remove first column from each line"""
    try:
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Process each line: split, remove first column, rejoin
        processed_lines = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                processed_lines.append(line)
                continue

            # Split into columns and remove first entry
            columns = line.split()
            if len(columns) > 1:
                processed_lines.append(' '.join(columns[1:]) + '\n')
            else:
                # Keep lines with only one column? (problem says three, but being safe)
                processed_lines.append(line)

        # Write processed content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python remove_first_column.py <directory_path>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    # Process all files in the directory
    processed_count = 0
    for file_path in directory.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            if process_file(file_path):
                print(f"Processed: {file_path.name}")
                processed_count += 1

    print(f"\nProcessing complete! Modified {processed_count} files.")

if __name__ == "__main__":
    main()
