#!/bin/bash

# Default values
VOCAB_THRESHOLD=50
TRAIN_DROPOUT=0
SEED=$RANDOM

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -l|--languages)
            IFS=',' read -ra LANGUAGES <<< "$2"
            shift 2
            ;;
        -s|--num_operations)
            NUM_OPERATIONS="$2"
            shift 2
            ;;
        -t|--vocab_threshold)
            VOCAB_THRESHOLD="$2"
            shift 2
            ;;
        --train_dropout)
            TRAIN_DROPOUT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DATA_DIR" || -z "${LANGUAGES[@]}" || -z "$NUM_OPERATIONS" ]]; then
    echo "Usage: $0 --data_dir DATA_DIR --languages LANG1,LANG2,... --num_operations NUM_OPS"
    echo "Optional: [--vocab_threshold THRESHOLD] [--train_dropout PROB] [--seed SEED]"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$(pwd)/data"
mkdir -p "$OUTPUT_DIR"

# Prepare input files for joint BPE training
TRAIN_FILES=()
VOCAB_PATHS=()
for lang in "${LANGUAGES[@]}"; do
    # Find the first training file with this language extension
    file=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.$lang" | grep -iE 'train|training' | head -1)
    if [[ -z "$file" ]]; then
        echo "Error: No training file found for language $lang"
        exit 1
    fi
    TRAIN_FILES+=("$file")
    VOCAB_PATHS+=("$OUTPUT_DIR/vocab.$lang")
done

# Step 1: Learn joint BPE and vocabulary
subword-nmt learn-joint-bpe-and-vocab \
    --input "${TRAIN_FILES[@]}" \
    -s "$NUM_OPERATIONS" \
    -o "$OUTPUT_DIR/codes_file" \
    --write-vocabulary "${VOCAB_PATHS[@]}"

# Verify that vocab files were created
for vocab_path in "${VOCAB_PATHS[@]}"; do
    if [[ ! -f "$vocab_path" ]]; then
        echo "Error: Vocabulary file $vocab_path was not created"
        exit 1
    fi
done

# Process files for each language
for lang in "${LANGUAGES[@]}"; do
    # Set language-specific vocabulary path
    VOCAB_PATH="$OUTPUT_DIR/vocab.$lang"

    # Create directory for language-specific outputs
    mkdir -p "$OUTPUT_DIR/$lang"

    # Process all files with this language extension
    while IFS= read -r -d $'\0' file; do
        base=$(basename "$file")
        out_file="$OUTPUT_DIR/$lang/${base}.BPE"

        # Determine if this is a training file
        if [[ "$base" =~ [Tt][Rr][Aa][Ii][Nn] ]]; then
            # Training file - apply dropout if specified
            subword-nmt apply-bpe \
                -c "$OUTPUT_DIR/codes_file" \
                --vocabulary "$VOCAB_PATH" \
                --vocabulary-threshold "$VOCAB_THRESHOLD" \
                ${TRAIN_DROPOUT:+--dropout $TRAIN_DROPOUT} \
                ${SEED:+--seed $SEED} \
                < "$file" > "$out_file"
        else
            # Non-training file - no dropout
            subword-nmt apply-bpe \
                -c "$OUTPUT_DIR/codes_file" \
                --vocabulary "$VOCAB_PATH" \
                --vocabulary-threshold "$VOCAB_THRESHOLD" \
                < "$file" > "$out_file"
        fi

        echo "Processed $file → $out_file"
    done < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.$lang" -print0)
done

echo "Processing complete. Output files saved to: $OUTPUT_DIR"
echo "BPE codes: $OUTPUT_DIR/codes_file"
echo "Vocabulary files:"
for lang in "${LANGUAGES[@]}"; do
    echo "  $lang: $OUTPUT_DIR/vocab.$lang"
done
