import os
import re
import time
import random
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import sys

def parse_output_specs(spec_list):
    """Parse output specifications into (base_name, probability) tuples"""
    specs = []
    total_prob = 0.0

    # Process arguments in pairs
    for i in range(0, len(spec_list), 2):
        base_name = spec_list[i]
        try:
            prob = float(spec_list[i+1])
            if prob <= 0:
                raise ValueError("Probability must be positive")
            specs.append((base_name, prob))
            total_prob += prob
        except (IndexError, ValueError) as e:
            raise argparse.ArgumentTypeError(f"Invalid specification: {e}")

    # Normalize probabilities if they don't sum to 1.0
    if abs(total_prob - 1.0) > 1e-5:
        if total_prob <= 0:
            raise argparse.ArgumentTypeError("Total probability must be positive")
        # Normalize probabilities
        specs = [(name, prob/total_prob) for name, prob in specs]
        print(f"Normalized probabilities to sum to 1.0 (original sum: {total_prob:.4f})")

    return specs

def create_output_files(specs):
    """Create output file handles based on specifications"""
    handles = []
    for base_name, prob in specs:
        src_file = f"{base_name}-1.src"
        trg_file = f"{base_name}-1.trg"
        try:
            src_handle = open(src_file, 'w', encoding='utf-8', buffering=1)
            trg_handle = open(trg_file, 'w', encoding='utf-8', buffering=1)
            handles.append((src_handle, trg_handle, prob))
            print(f"Created output files: {src_file}, {trg_file} (probability: {prob:.4f})")
        except IOError as e:
            raise IOError(f"Error creating output files for {base_name}: {str(e)}")
    return handles

def choose_output(handles):
    """Randomly select an output set based on probabilities"""
    r = random.random()
    cumulative = 0.0
    for i, (src_handle, trg_handle, prob) in enumerate(handles):
        cumulative += prob
        if r < cumulative:
            return src_handle, trg_handle
    # Fallback to last handle if probabilities don't cover full range
    return handles[-1][0], handles[-1][1]

def clean_sentence(sentence):
    """Clean and normalize sentence text"""
    if not sentence:
        return ""
    return sentence.strip().replace('\n', ' ').replace('\r', '')

def process_xml(input_file, output_handles, timeout):
    """Process XML file incrementally and write to output files"""
    processed_sentences = set()
    last_activity = time.time()
    buffer = ""
    last_size = 0
    entry_count = 0
    sentence_count = 0

    print(f"Monitoring {input_file} for new entries (timeout: {timeout}s)")

    while (time.time() - last_activity) < timeout:
        try:
            # Check file existence
            if not os.path.exists(input_file):
                time.sleep(1)
                continue

            # Check file size
            current_size = os.path.getsize(input_file)

            # Handle file truncation or rotation
            if current_size < last_size:
                buffer = ""
                last_size = 0
                print("Input file was reset - clearing buffer")

            # Read new data if available
            if current_size > last_size:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    in_f.seek(last_size)
                    new_data = in_f.read(current_size - last_size)
                    buffer += new_data
                    last_size = current_size
                    last_activity = time.time()  # Reset timeout on new data

            # Process complete entries in buffer
            entries_processed = False
            while True:
                # Find next complete entry
                start_idx = buffer.find("<entry")
                if start_idx == -1:
                    break

                end_idx = buffer.find("</entry>", start_idx)
                if end_idx == -1:
                    break

                end_idx += len("</entry>")
                entry_xml = buffer[start_idx:end_idx]
                buffer = buffer[end_idx:]

                try:
                    # Parse the entry
                    entry_elem = ET.fromstring(entry_xml)
                    path = entry_elem.get('path', '')
                    pattern = entry_elem.get('pattern', '')
                    entry_count += 1

                    # Process all sentences in this entry
                    sentences = []

                    # Handle both single-sentence and multi-sentence entries
                    if entry_elem.find('sentence') is not None:
                        # Multi-sentence entry structure
                        for sent_elem in entry_elem.findall('sentence'):
                            src_elem = sent_elem.find('src_string')
                            trg_elem = sent_elem.find('trg_string')

                            if src_elem is None or trg_elem is None:
                                continue

                            src_text = src_elem.text or ''
                            trg_text = trg_elem.text or ''

                            sentences.append((src_text, trg_text))
                    else:
                        # Single-sentence entry structure
                        src_elem = entry_elem.find('src_string')
                        trg_elem = entry_elem.find('trg_string')

                        if src_elem is not None and trg_elem is not None:
                            src_text = src_elem.text or ''
                            trg_text = trg_elem.text or ''
                            sentences.append((src_text, trg_text))

                    # Process each sentence pair
                    for src_text, trg_text in sentences:
                        src_clean = clean_sentence(src_text)
                        trg_clean = clean_sentence(trg_text)

                        if not src_clean or not trg_clean:
                            continue

                        # Create unique key for this sentence pair
                        sentence_key = (path, pattern, src_clean, trg_clean)
                        if sentence_key in processed_sentences:
                            continue

                        processed_sentences.add(sentence_key)
                        sentence_count += 1
                        entries_processed = True

                        # Select output files based on probability
                        src_handle, trg_handle = choose_output(output_handles)

                        # Write to output files
                        src_handle.write(src_clean + '\n')
                        trg_handle.write(trg_clean + '\n')

                    # Reset timeout on successful processing
                    if entries_processed:
                        last_activity = time.time()

                    # Print progress periodically
                    if entry_count % 100 == 0:
                        print(f"Processed {entry_count} entries, {sentence_count} sentences")

                except (ParseError, TypeError, ValueError) as e:
                    # Skip malformed entries
                    continue

            # Sleep briefly to avoid busy waiting
            time.sleep(0.5)

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            time.sleep(1)

    # Close all output files
    for src_handle, trg_handle, _ in output_handles:
        src_handle.close()
        trg_handle.close()

    print(f"Processing complete. Total: {entry_count} entries, {sentence_count} sentences")

def main():
    parser = argparse.ArgumentParser(
        description='Create OpenNMT dataset from XML with gendered German terms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file', help='Path to input XML file')
    parser.add_argument('--outputs', nargs='+', required=True,
                        help='Output specifications: base_name1 prob1 base_name2 prob2 ...')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds for waiting on new input')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible dataset splitting')

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    try:
        # Parse output specifications
        output_specs = parse_output_specs(args.outputs)

        # Create output files
        output_handles = create_output_files(output_specs)

        # Process XML file
        process_xml(args.input_file, output_handles, args.timeout)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
