import os
import re
import argparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from xml.dom import minidom

def process_html_file(file_path, regexes):
    """Process a single HTML file and return matches for all regexes"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get all text and split into sentences (very simple approach)
            text = soup.get_text(separator=' ', strip=True)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

            matches = []
            for sentence in sentences:
                for name, pattern in regexes.items():
                    if pattern.search(sentence):
                        matches.append({
                            'sentence': sentence.strip(),
                            'pattern': name
                        })
            return matches
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def write_xml_entry(xml_file, file_path, matches):
    """Write matches for a single file to the XML file"""
    if not matches:
        return

    # Create entry element
    entry = ET.Element('entry', {'file': os.path.basename(file_path), 'path': file_path})

    for match in matches:
        match_elem = ET.SubElement(entry, 'match', {'pattern': match['pattern']})
        match_elem.text = match['sentence']

    # Write to file
    xml_str = ET.tostring(entry, encoding='unicode')
    xml_file.write(xml_str + '\n')

def process_directory(root_dir, output_file, regexes):
    """Process all HTML files in directory and subdirectories"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write XML header
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<matches>\n')

        # Process each HTML file
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.html'):
                    full_path = os.path.join(dirpath, filename)
                    matches = process_html_file(full_path, regexes)
                    write_xml_entry(f, full_path, matches)

        # Close root element
        f.write('</matches>\n')

def main():
    # Define your regex patterns here
    regex_patterns = {
        'regex1': re.compile(r'\b(\w+)\b\s+und\s+\b\1(?:innen|Innen)\b', re.IGNORECASE),
        'regex2': re.compile(r'\b\w*\*\w*\b', re.IGNORECASE),          # Asterisk gender forms
        'regex3': re.compile(r'\b\w*\:\w*\b', re.IGNORECASE),          # Colon gender forms
        'regex4': re.compile(r'\b\w*\_\w*\b', re.IGNORECASE),          # Underscore gender forms
        'regex5': re.compile(r'(\w+)\s+(?:und|oder)\s+\1innen\w+'),    # Gendered composita
        'regex6': re.compile(r'\b[A-ZÄÖÜ][a-zäöüß]+I(?:[a-zäöüß]+)?\b'),  # Binnen-I terms
        'regex7': re.compile(r'\b\w+(?:innen|Innen|in)\w*\b|\b\w+er\w*\b')

    }

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process German newspaper HTML files and extract sentences matching given patterns.')
    parser.add_argument('input_dir', help='Directory containing HTML files')
    parser.add_argument('output_file', help='XML output file path')

    args = parser.parse_args()

    print(f"Processing directory: {args.input_dir}")
    print(f"Output will be written to: {args.output_file}")

    process_directory(args.input_dir, args.output_file, regex_patterns)
    print("Processing complete.")

if __name__ == '__main__':
    main()
