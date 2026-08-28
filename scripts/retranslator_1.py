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
    """Process XML file, retranslate terms, and write output with timeout handling"""
    processed_keys = set()
    start_time = time.time()
    last_modified = 0

    # Create output file with root element
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out_f.write('<entries>\n')

    while (time.time() - start_time) < timeout:
        try:
            # Check file existence and modification time
            if not os.path.exists(input_file):
                time.sleep(1)
                continue

            current_mtime = os.path.getmtime(input_file)
            if current_mtime <= last_modified:
                time.sleep(1)
                continue

            last_modified = current_mtime
            modified = False

            with open(input_file, 'r', encoding='utf-8') as in_f:
                content = in_f.read()

            # Skip if file doesn't end with closing tag
            if not content.strip().endswith('</matches>'):
                time.sleep(1)
                continue

            try:
                root = ET.fromstring(content)
            except ParseError:
                time.sleep(1)
                continue

            # Process each entry in the XML
            for entry_elem in root.findall('entry'):
                file_path = entry_elem.get('path')
                for match_elem in entry_elem.findall('match'):
                    pattern_name = match_elem.get('pattern')
                    original_sentence = match_elem.text.strip() if match_elem.text else ""

                    # Skip if we've processed this exact match before
                    key = (file_path, pattern_name, original_sentence)
                    if key in processed_keys:
                        continue

                    processed_keys.add(key)
                    modified = True

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

            # If we processed new data, reset the timeout timer
            if modified:
                start_time = time.time()

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            time.sleep(1)

    # Close root element in output file
    with open(output_file, 'a', encoding='utf-8') as out_f:
        out_f.write('</entries>\n')

def main():
    parser = argparse.ArgumentParser(
        description='Process XML file with gendered German terms and convert to masculine plural.'
    )
    parser.add_argument('input_file', help='Path to input XML file')
    parser.add_argument('output_file', help='Path to output XML file')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds (default: 300)')

    args = parser.parse_args()

    print(f"Monitoring {args.input_file} for new entries (timeout: {args.timeout}s)")
    process_xml_file(args.input_file, args.output_file, args.timeout)
    print(f"Processing complete. Results written to {args.output_file}")

if __name__ == '__main__':
    main()
