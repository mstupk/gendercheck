# gendercheck
A Libre Office Writer Plugin which genders German text similar to an autocorrection. Written as Programming Project for University of Zurich. This project is intended to be commercialisec at some later point.

# German Gender-Term Machine Translation Pipeline

This repository contains a pipeline for processing gendered German terms in newspaper articles and creating datasets for training machine translation models to convert gender-neutral terms back to masculine plural forms.

## Pipeline Overview

1. **Crawling & Extraction**: Scan HTML files to find gendered German terms
2. **Retranslation**: Convert gendered terms back to masculine plural form
3. **Dataset Creation**: Create parallel corpora for machine translation training

```mermaid
graph LR
A[HTML Files] --> B(Crawler & Extraction)
B --> C[Raw XML]
C --> D(Retranslation)
D --> E[Retranslated XML]
E --> F(Dataset Creation)
F --> G[Train/Valid/Test Sets]

Script Descriptions
1. regex_crawler_1.py
Purpose:
Scans HTML files in a directory structure, extracts sentences containing gendered German terms using regex patterns, and outputs an XML file with matches.

Key Features:

Processes HTML files recursively

Removes script/style tags before processing

Uses regex patterns to identify gendered terms:

Paired terms (e.g., "Lehrer und Lehrerinnen")

Asterisk forms (e.g., "Lehrer*innen")

Colon forms (e.g., "Lehrer:innen")

Underscore forms (e.g., "Lehrer_innen")

Usage:

bash
python regex_crawler_1.py <input_dir> <output.xml>
Arguments:

input_dir: Directory containing HTML files

output.xml: Output XML file path

Output XML Structure:

xml
<matches>
  <entry file="..." path="...">
    <match pattern="...">Full sentence containing gendered term</match>
  </entry>
</matches>
2. retranslate_xml.py
Purpose:
Processes the XML output from the crawler, retranslates gendered terms to masculine plural form, and outputs a new XML file with both versions.

Key Features:

Handles incomplete XML files (actively being written)

Retranslates four types of gendered terms:

Paired terms → Base word (e.g., "Lehrer und Lehrerinnen" → "Lehrer")

Asterisk forms → Text before * (e.g., "Lehrer*innen" → "Lehrer")

Colon forms → Text before : (e.g., "Lehrer:innen" → "Lehrer")

Underscore forms → Text before _ (e.g., "Lehrer_innen" → "Lehrer")

Configurable timeout for processing actively written files

Usage:

bash
python retranslate_xml.py <input.xml> <output.xml> [--timeout 300]
Arguments:

input.xml: XML file from regex_crawler_1.py

output.xml: Output XML file path

--timeout: Seconds to wait for new entries (default: 300)

Output XML Structure:

xml
<entries>
  <entry file="..." path="..." pattern="...">
    <src_string>Masculine plural version</src_string>
    <trg_string>Original gendered version</trg_string>
  </entry>
</entries>
3. create_dataset.py
Purpose:
Creates parallel corpora datasets for machine translation from the retranslated XML file.

Key Features:

Handles incomplete XML files

Supports both single and multi-sentence entries

Splits data into multiple sets with specified probabilities

Creates properly formatted source/target files for OpenNMT

Removes duplicates and cleans text

Usage:

bash
python create_dataset.py <input.xml> \
  --outputs <base1> <prob1> <base2> <prob2> ... \
  [--timeout 300] \
  [--seed <random_seed>]
Arguments:

input.xml: XML file from retranslate_xml.py

--outputs: Space-separated list of output specifications (base_name probability pairs)

--timeout: Seconds to wait for new input (default: 300)

--seed: Random seed for reproducible splits

Output Files:
For each base_name in outputs:

{base_name}-1.src: Masculine plural versions

{base_name}-1.trg: Original gendered versions

Example:

bash
python create_dataset.py retranslated.xml \
  --outputs train 0.8 valid 0.1 test 0.1 \
  --timeout 600 \
  --seed 42
Pipeline Execution Workflow
Step 1: Extract Gendered Terms
bash
python regex_crawler_1.py newspaper_articles/ gendered_terms.xml
Step 2: Retranslate to Masculine Plural
bash
python retranslate_xml.py gendered_terms.xml retranslated.xml --timeout 600
Step 3: Create Training Datasets
bash
python create_dataset.py retranslated.xml \
  --outputs train 0.8 valid 0.1 test 0.1 \
  --timeout 600 \
  --seed 42
Step 4: Train OpenNMT Model
bash
onmt_train \
  -config config.yml \
  -data data/processed-1 \
  -save_model models/gender_model
Example config.yml:

yaml
# config.yml
data:
    train_features: train-1.src
    train_labels: train-1.trg
    valid_features: valid-1.src
    valid_labels: valid-1.trg
    save_data: data/processed-1/vocab

# Model parameters
model: transformer
optim: adam
learning_rate: 2.0
warmup_steps: 8000
encoder_type: transformer
decoder_type: transformer
word_vec_size: 512
hidden_size: 512
layers: 6
heads: 8
Technical Notes
Incomplete File Handling:

Both retranslate_xml.py and create_dataset.py can process actively written files

Use the --timeout parameter to control how long to wait for new content

Scripts automatically detect file truncation/rotation

Data Formats:

All scripts use UTF-8 encoding

XML files are stream-processed for memory efficiency

Output datasets are clean, parallel text files

Customization:

Modify regex patterns in regex_crawler_1.py for different gendered forms

Adjust dataset splits with different probability values

Add preprocessing steps as needed for specific requirements

Requirements
Python 3.7+

Required packages:

beautifulsoup4

lxml

argparse

Install dependencies with:

bash
pip install beautifulsoup4 lxml
Future Enhancements
Add progress bars for long-running operations

Support additional gendered term patterns

Implement incremental model training

Add Docker support for containerized execution

Integrate with cloud storage for input/output files

