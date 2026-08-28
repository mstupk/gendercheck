import os
import re
import time
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

# Updated regex patterns including all 7 patterns
regex_patterns = {
    'regex1': re.compile(r'\b(\w+)\b\s+und\s+\b\1(?:innen|Innen)\b', re.IGNORECASE),
    'regex2': re.compile(r'\b\w*\*\w*\b', re.IGNORECASE),          # Asterisk gender forms
    'regex3': re.compile(r'\b\w*\:\w*\b', re.IGNORECASE),          # Colon gender forms
    'regex4': re.compile(r'\b\w*\_\w*\b', re.IGNORECASE),          # Underscore gender forms
    'regex5': re.compile(r'(\w+)\s+(?:und|oder)\s+\1innen\w+', re.IGNORECASE),    # Gendered composita
    'regex6': re.compile(r'\b[A-ZÄÖÜ][a-zäöüß]+I(?:[a-zäöüß]+)?\b', re.IGNORECASE),  # Binnen-I terms
    'regex7': re.compile(r'\b\w+(?:innen|Innen|in)\w*\b|\b\w+er\w*\b', re.IGNORECASE)
}

def translate_matched_term(match_obj, pattern_name):
    """Translate a matched gendered term back to masculine plural form with verbose output."""
    if pattern_name == 'regex1':
        base_term = match_obj.group(1)
    elif pattern_name in ('regex2', 'regex3', 'regex4'):
        separator = {
            'regex2': '*',
            'regex3': ':',
            'regex4': '_'
        }[pattern_name]
        base_term = match_obj.group(0).split(separator, 1)[0]
    elif pattern_name == 'regex5':
        base_term = match_obj.group(1)
    elif pattern_name == 'regex6':
        term = match_obj.group(0)
        # Find first occurrence of 'I' or 'i'
        idx = None
        for i, char in enumerate(term):
            if char == 'I' or char == 'i':
                idx = i
                break
        base_term = term[:idx] if idx is not None else term
    elif pattern_name == 'regex7':
        term = match_obj.group(0)
        lower_term = term.lower()
        if lower_term.endswith('innen'):
            base_term = term[:-5]
        elif lower_term.endswith('in'):
            base_term = term[:-2]
        else:
            base_term = term
    else:
        base_term = match_obj.group(0)

    return base_term

def process_xml_file(input_file, output_file, timeout, mode, monitor_time):
    """Process XML file incrementally, handling incomplete entries"""
    # Create output file with root element
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out_f.write('<entries>\n')

    processed_entries = set()
    start_time = time.time()
    last_size = 0
    buffer = ""
    chunk_size = 4096  # 4KB chunks for large files
    last_print_time = 0  # For monitor-time functionality

    print(f"Processing mode: {'sentences' if mode == 'sentence' else 'corpora'}")
    if monitor_time > 0:
        print(f"Monitoring terms every {monitor_time} seconds")

    while (time.time() - start_time) < timeout:
        try:
            # Check if file exists
            if not os.path.exists(input_file):
                time.sleep(1)
                continue

            # Get current file size
            current_size = os.path.getsize(input_file)

            # Reset if file was truncated or rotated
            if current_size < last_size:
                buffer = ""
                last_size = 0
                processed_entries.clear()
                print("Input file was reset - clearing state")

            # Read new data in chunks if file has grown
            if current_size > last_size:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    in_f.seek(last_size)
                    remaining = current_size - last_size

                    while remaining > 0:
                        # Read in chunks
                        read_size = min(chunk_size, remaining)
                        new_data = in_f.read(read_size)
                        buffer += new_data
                        last_size += len(new_data)
                        remaining -= len(new_data)

                        # Process complete entries in buffer
                        processed_entry = False
                        while True:
                            # Find the next complete entry
                            start_tag = buffer.find("<entry")
                            if start_tag == -1:
                                break

                            end_tag = buffer.find("</entry>", start_tag)
                            if end_tag == -1:
                                break

                            end_tag += len("</entry>")
                            entry_xml = buffer[start_tag:end_tag]
                            buffer = buffer[end_tag:]

                            try:
                                # Parse the entry element
                                entry_elem = ET.fromstring(entry_xml)
                                file_path = entry_elem.get('path')

                                # Process each match in the entry
                                for match_elem in entry_elem.findall('match'):
                                    pattern_name = match_elem.get('pattern')
                                    original_sentence = match_elem.text.strip() if match_elem.text else ""

                                    # Skip duplicate entries
                                    entry_key = (file_path, pattern_name, original_sentence)
                                    if entry_key in processed_entries:
                                        continue
                                    processed_entries.add(entry_key)

                                    # Find matching pattern and retranslate
                                    pattern = regex_patterns.get(pattern_name)
                                    if pattern:
                                        match_obj = pattern.search(original_sentence)
                                        if match_obj:
                                            base_term = translate_matched_term(match_obj, pattern_name)

                                            # Verbose output with monitor-time control
                                            current_time = time.time()
                                            if monitor_time == 0 or (current_time - last_print_time) >= monitor_time:
                                                print(f"Processing pattern {pattern_name}:")
                                                print(f"  Original term: '{match_obj.group(0)}'")
                                                print(f"  Translated to: '{base_term}'")
                                                print(f"  Context: '{original_sentence[:30]}...'")
                                                last_print_time = current_time

                                            retranslated_sentence = (
                                                original_sentence[:match_obj.start()] +
                                                base_term +
                                                original_sentence[match_obj.end():]
                                            )
                                        else:
                                            retranslated_sentence = original_sentence
                                    else:
                                        retranslated_sentence = original_sentence

                                    # Create output XML structure
                                    out_entry = ET.Element('entry')
                                    out_entry.set('file', os.path.basename(file_path))
                                    out_entry.set('path', file_path)
                                    out_entry.set('pattern', pattern_name)

                                    src_elem = ET.SubElement(out_entry, 'src_string')
                                    src_elem.text = retranslated_sentence

                                    trg_elem = ET.SubElement(out_entry, 'trg_string')
                                    trg_elem.text = original_sentence

                                    # Append to output file
                                    xml_str = ET.tostring(out_entry, encoding='unicode')
                                    with open(output_file, 'a', encoding='utf-8') as out_f:
                                        out_f.write(xml_str + '\n')

                                    processed_entry = True
                            except ParseError:
                                # Malformed XML - skip this entry
                                continue

                        # Reset timeout if we processed any entries
                        if processed_entry:
                            start_time = time.time()

            # Sleep briefly before checking again
            time.sleep(0.5)

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            time.sleep(1)

    # Close root element in output file
    with open(output_file, 'a', encoding='utf-8') as out_f:
        out_f.write('</entries>\n')

def main():
    parser = argparse.ArgumentParser(
        description='Process incomplete XML files with gendered German terms and convert to masculine plural.'
    )
    parser.add_argument('input_file', help='Path to input XML file being written incrementally')
    parser.add_argument('output_file', help='Path to output XML file')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds (default: 300)')
    parser.add_argument('--mode', choices=['sentence', 'corpus'], default='sentence',
                        help='Processing mode: "sentence" for individual sentences, "corpus" for whole documents (default: sentence)')
    parser.add_argument('--monitor-time', type=int, default=0,
                        help='Display terms every n seconds (0=show all, default=0)')

    args = parser.parse_args()

    print(f"Monitoring {args.input_file} for new entries (timeout: {args.timeout}s)")
    process_xml_file(args.input_file, args.output_file, args.timeout, args.mode, args.monitor_time)
    print(f"Processing complete. Results written to {args.output_file}")

if __name__ == '__main__':
    main()
