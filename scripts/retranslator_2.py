import os
import re
import time
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

# Regex patterns from the original script
regex_patterns = {
    'regex1': re.compile(r'\b(\w+)\b\s+und\s+\b\1(?:innen|Innen)\b', re.IGNORECASE),
    'regex2': re.compile(r'\b\w*\*\w*\b', re.IGNORECASE),
    'regex3': re.compile(r'\b\w*\:\w*\b', re.IGNORECASE),
    'regex4': re.compile(r'\b\w*\_\w*\b', re.IGNORECASE)
}

def translate_matched_term(match_obj, pattern_name):
    """Translate a matched gendered term back to masculine plural form."""
    if pattern_name == 'regex1':
        return match_obj.group(1)
    elif pattern_name in ('regex2', 'regex3', 'regex4'):
        separator = {
            'regex2': '*',
            'regex3': ':',
            'regex4': '_'
        }[pattern_name]
        return match_obj.group(0).split(separator, 1)[0]
    return match_obj.group(0)

def process_xml_file(input_file, output_file, timeout):
    """Process XML file incrementally, handling incomplete entries"""
    # Create output file with root element
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out_f.write('<entries>\n')

    processed_entries = set()
    start_time = time.time()
    last_size = 0
    buffer = ""

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

            # Read new data if file has grown
            if current_size > last_size:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    in_f.seek(last_size)
                    new_data = in_f.read(current_size - last_size)
                    buffer += new_data
                    last_size = current_size

            # Process all complete entries in buffer
            entries_processed = False
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

                        entries_processed = True
                except ParseError:
                    # Malformed XML - skip this entry
                    continue

            # Reset timeout timer if we processed new entries
            if entries_processed:
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

    args = parser.parse_args()

    print(f"Monitoring {args.input_file} for new entries (timeout: {args.timeout}s)")
    process_xml_file(args.input_file, args.output_file, args.timeout)
    print(f"Processing complete. Results written to {args.output_file}")

if __name__ == '__main__':
    main()
