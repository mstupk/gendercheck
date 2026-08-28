import os
import re
import argparse
from bs4 import BeautifulSoup
import time
import sys

def count_html_files(root_dir):
    """Count all HTML files in directory and subdirectories"""
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.html'):
                count += 1
    return count

def extract_sentences(html_file_path):
    """Extract cleaned text sentences from an HTML file"""
    try:
        with open(html_file_path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f, 'html.parser')

            # Remove unnecessary elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                element.decompose()

            # Get clean text
            text = soup.get_text(separator=' ', strip=True)
            # Improved sentence splitting
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        print(f"\nError processing {html_file_path}: {str(e)}", file=sys.stderr)
        return []

def display_stats(stats, current, total_files):
    """Display real-time statistics in htop-like format"""
    # Calculate derived statistics
    elapsed = time.time() - stats['start_time']
    processed_percent = (current / total_files * 100) if total_files > 0 else 0
    articles_match_percent = (stats['articles_with_match'] / stats['total_articles'] * 100) if stats['total_articles'] > 0 else 0
    sentences_match_percent = (stats['sentences_with_match'] / stats['total_sentences'] * 100) if stats['total_sentences'] > 0 else 0
    avg_sentences = stats['total_sentences'] / stats['total_articles'] if stats['total_articles'] > 0 else 0
    time_per_file = elapsed / current if current > 0 else 0
    eta = (total_files - current) * time_per_file if current > 0 else 0
    files_per_sec = current / elapsed if elapsed > 0 else 0

    # Prepare progress bar
    progress_width = 50
    filled = int(progress_width * current / total_files) if total_files > 0 else 0
    progress_bar = '[' + '■' * filled + ' ' * (progress_width - filled) + ']'

    # Build display
    display = f"""
German Newspaper Corpus - Real-time Analysis (Ctrl+C to exit)
====================================================================
[Progress]  {progress_bar} {current}/{total_files} ({processed_percent:.1f}%)
            Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s
            Files/s: {files_per_sec:.1f} | Time/file: {time_per_file:.3f}s

[Overview]  Articles: {stats['total_articles']} processed, {stats['articles_with_match']} with matches ({articles_match_percent:.1f}%)
            Sentences: {stats['total_sentences']} total, {stats['sentences_with_match']} with matches ({sentences_match_percent:.1f}%)
            Avg. sentences: {avg_sentences:.1f} | Max: {stats['max_sentences']}

[Pattern Matches]
"""
    # Add pattern counts
    for pattern in stats['pattern_counts'].keys():
        count = stats['pattern_counts'][pattern]
        display += f"  {pattern}: {count}\n"

    display += "====================================================================\n"

    # Update display
    sys.stdout.write("\033[H\033[J")  # Clear screen
    sys.stdout.write(display)
    sys.stdout.flush()

def generate_report(stats, report_file, total_files):
    """Generate and save a statistics report"""
    # Calculate derived statistics
    articles_match_percent = (stats['articles_with_match'] / stats['total_articles'] * 100) if stats['total_articles'] > 0 else 0
    sentences_match_percent = (stats['sentences_with_match'] / stats['total_sentences'] * 100) if stats['total_sentences'] > 0 else 0
    avg_sentences = stats['total_sentences'] / stats['total_articles'] if stats['total_articles'] > 0 else 0
    elapsed_time = time.time() - stats['start_time']

    report = f"""German Newspaper Corpus Analysis - Gendered Language Patterns
====================================================================
Processing time: {elapsed_time:.1f} seconds
Files processed: {stats['processed_files']}/{total_files}
Files/s: {stats['processed_files'] / elapsed_time:.1f}

[Overview]
Total articles processed: {stats['total_articles']}
Articles with matches: {stats['articles_with_match']} ({articles_match_percent:.1f}% of total)
Total sentences processed: {stats['total_sentences']}
Sentences with matches: {stats['sentences_with_match']} ({sentences_match_percent:.1f}% of total)

[Pattern Statistics]
"""
    # Add pattern counts
    for pattern, count in stats['pattern_counts'].items():
        report += f"- {pattern}: {count} matches\n"

    report += f"""
[Article Statistics]
Average sentences per article: {avg_sentences:.1f}
Maximum sentences in an article: {stats['max_sentences']}

[Pattern Descriptions]
1. regex1: Standard gendered pairs (e.g., "Leser und Leserinnen")
2. regex2: Gender asterisk (e.g., "Leser*innen")
3. regex3: Gender colon (e.g., "Leser:innen")
4. regex4: Gender underscore (e.g., "Leser_innen")
5. regex5: Gendered composita (e.g., "Leser und Leserinnenkommentar")
6. regex6: Binnen-I terms (e.g., "LeserInnen")

====================================================================
Report generated on {time.ctime()}
"""
    # Save to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {os.path.abspath(report_file)}")

def main():
    # Define regex patterns for German gendered language
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
        description='Analyze German newspaper HTML files for gendered language patterns with real-time display')
    parser.add_argument('input_dir', help='Directory containing HTML files')
    parser.add_argument('report_file', help='Report output file path')

    args = parser.parse_args()

    print(f"Scanning directory: {args.input_dir}")
    start_time = time.time()

    # Get list of all HTML files
    all_files = []
    for dirpath, _, filenames in os.walk(args.input_dir):
        for filename in filenames:
            if filename.lower().endswith('.html'):
                all_files.append(os.path.join(dirpath, filename))

    total_files = len(all_files)
    if total_files == 0:
        print("No HTML files found in the specified directory.")
        return

    print(f"Found {total_files} HTML files. Starting analysis...")

    # Initialize statistics
    stats = {
        'start_time': start_time,
        'total_articles': 0,
        'articles_with_match': 0,
        'total_sentences': 0,
        'sentences_with_match': 0,
        'pattern_counts': {name: 0 for name in regex_patterns.keys()},
        'max_sentences': 0,
        'processed_files': 0
    }

    # Initial display
    sys.stdout.write("\033[H\033[J")  # Clear screen
    display_stats(stats, 0, total_files)

    last_display_update = 0
    processed_count = 0

    try:
        for file_path in all_files:
            # Process file
            sentences = extract_sentences(file_path)
            processed_count += 1

            if sentences:
                stats['total_articles'] += 1
                num_sentences = len(sentences)
                stats['total_sentences'] += num_sentences
                stats['max_sentences'] = max(stats['max_sentences'], num_sentences)

                article_has_match = False
                for sentence in sentences:
                    sentence_has_match = False
                    for pattern_name, pattern in regex_patterns.items():
                        if pattern.search(sentence):
                            stats['pattern_counts'][pattern_name] += 1
                            sentence_has_match = True

                    if sentence_has_match:
                        stats['sentences_with_match'] += 1
                        article_has_match = True

                if article_has_match:
                    stats['articles_with_match'] += 1

            stats['processed_files'] = processed_count

            # Update display at least once per second
            current_time = time.time()
            if current_time - last_display_update >= 0.5:  # Update twice per second
                display_stats(stats, processed_count, total_files)
                last_display_update = current_time

        # Final display update
        display_stats(stats, processed_count, total_files)
        print("\nProcessing complete!")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user. Finalizing report...")
        display_stats(stats, processed_count, total_files)

    # Generate final report
    generate_report(stats, args.report_file, total_files)

if __name__ == '__main__':
    main()
