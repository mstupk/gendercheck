import os
import re
import argparse
from bs4 import BeautifulSoup

def extract_sentences(html_file_path):
    """Extract cleaned text sentences from an HTML file"""
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

            # Remove unnecessary elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                element.decompose()

            # Get clean text and split into sentences
            text = soup.get_text(separator=' ', strip=True)
            # Improved sentence splitting regex
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        print(f"Error processing {html_file_path}: {str(e)}")
        return []

def collect_statistics(root_dir, regexes):
    """Collect statistics from all HTML files in directory and subdirectories"""
    stats = {
        'total_articles': 0,
        'articles_with_match': 0,
        'total_sentences': 0,
        'sentences_with_match': 0,
        'pattern_counts': {name: 0 for name in regexes.keys()},
        'sentences_per_article': []
    }

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.html'):
                full_path = os.path.join(dirpath, filename)
                sentences = extract_sentences(full_path)
                if not sentences:
                    continue

                stats['total_articles'] += 1
                num_sentences = len(sentences)
                stats['total_sentences'] += num_sentences
                stats['sentences_per_article'].append(num_sentences)

                article_has_match = False
                article_sentence_matches = 0

                for sentence in sentences:
                    sentence_has_match = False
                    for pattern_name, pattern in regexes.items():
                        # Count all matches of this pattern in the sentence
                        matches = list(pattern.finditer(sentence))
                        if matches:
                            stats['pattern_counts'][pattern_name] += len(matches)
                            sentence_has_match = True

                    if sentence_has_match:
                        stats['sentences_with_match'] += 1
                        article_sentence_matches += 1

                if article_sentence_matches > 0:
                    stats['articles_with_match'] += 1

    # Calculate derived statistics
    stats['avg_sentences'] = stats['total_sentences'] / stats['total_articles'] if stats['total_articles'] > 0 else 0
    stats['max_sentences'] = max(stats['sentences_per_article']) if stats['sentences_per_article'] else 0

    return stats

def generate_report(stats, report_file):
    """Generate and save a statistics report"""
    report = f"""German Newspaper Corpus Analysis - Gendered Language Patterns
====================================================================

[Overview]
Total articles processed: {stats['total_articles']}
Articles with matches: {stats['articles_with_match']} ({stats['articles_with_match']/stats['total_articles']*100:.1f}% of total)
Total sentences processed: {stats['total_sentences']}
Sentences with matches: {stats['sentences_with_match']} ({stats['sentences_with_match']/stats['total_sentences']*100:.1f}% of total)

[Pattern Statistics]
"""
    # Add pattern counts
    for pattern, count in stats['pattern_counts'].items():
        report += f"- {pattern}: {count} matches\n"

    report += f"""
[Article Statistics]
Average sentences per article: {stats['avg_sentences']:.1f}
Maximum sentences in an article: {stats['max_sentences']}

[Pattern Descriptions]
1. regex1: Standard gendered pairs (e.g., "Leser und Leserinnen")
2. regex2: Gender asterisk (e.g., "Leser*innen")
3. regex3: Gender colon (e.g., "Leser:innen")
4. regex4: Gender underscore (e.g., "Leser_innen")
5. regex5: Gendered composita (e.g., "Leser und Leserinnenkommentar")
6. regex6: Binnen-I terms (e.g., "LeserInnen")

====================================================================
Report generated on {os.path.basename(report_file)}
"""
    # Save to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # Also print to console
    print(report)
    print(f"Report saved to: {os.path.abspath(report_file)}")

def main():
    # Define regex patterns for German gendered language
    regex_patterns = {
        'regex1': re.compile(r'\b(\w+)\b\s+und\s+\b\1(?:innen|Innen)\b', re.IGNORECASE),
        'regex2': re.compile(r'\b\w*\*\w*\b', re.IGNORECASE),          # Asterisk gender forms
        'regex3': re.compile(r'\b\w*\:\w*\b', re.IGNORECASE),          # Colon gender forms
        'regex4': re.compile(r'\b\w*\_\w*\b', re.IGNORECASE),          # Underscore gender forms
        'regex5': re.compile(r'(\w+)\s+(?:und|oder)\s+\1innen\w+'),    # Gendered composita
        'regex6': re.compile(r'\b[A-ZÄÖÜ][a-zäöüß]+I(?:[a-zäöüß]+)?\b')  # Binnen-I terms
    }

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze German newspaper HTML files for gendered language patterns')
    parser.add_argument('input_dir', help='Directory containing HTML files')
    parser.add_argument('report_file', help='Report output file path')

    args = parser.parse_args()

    print(f"Processing directory: {args.input_dir}")
    stats = collect_statistics(args.input_dir, regex_patterns)
    generate_report(stats, args.report_file)
    print("Analysis complete.")

if __name__ == '__main__':
    main()
