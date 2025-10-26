"""
Prepare dataset: align sentences to character-level IPA
Input: text<TAB>ipa (sentence-level)
Output: text<TAB>ipa (character-level aligned)
"""
import argparse
from pathlib import Path
from tqdm import tqdm
from src.alignment import align_word
from src.preprocess import normalize_hebrew, normalize_ipa


def split_sentence(text, ipa):
    """
    Split sentence into words and align each word separately.
    
    Args:
        text: Hebrew sentence (may contain diacritics)
        ipa: IPA phonemes for entire sentence (space-separated words)
    
    Returns:
        Tuple of (aligned_text, aligned_ipa) with space-separated per-character phonemes
    """
    # Normalize text (removes diacritics and pipe)
    text_normalized = normalize_hebrew(text)
    ipa_normalized = normalize_ipa(ipa)
    
    # Split text into words (Hebrew letters only)
    # Split IPA into words (separated by spaces in the original)
    text_words = []
    current_word = []
    
    for char in text_normalized:
        if char.strip() and not char in ' ,.!?;:()[]{}""\'\'':
            current_word.append(char)
        else:
            if current_word:
                text_words.append(''.join(current_word))
                current_word = []
    if current_word:
        text_words.append(''.join(current_word))
    
    # Split IPA by spaces to get word-level phonemes
    ipa_words = ipa_normalized.split()
    
    # If counts don't match, try to align the whole thing as fallback
    if len(text_words) != len(ipa_words):
        print(f"  Warning: Word count mismatch (text: {len(text_words)}, ipa: {len(ipa_words)})")
        print(f"    Text words: {text_words[:5]}...")
        print(f"    IPA words: {ipa_words[:5]}...")
        return None, None
    
    # Align each word separately
    aligned_chars = []
    aligned_phonemes = []
    
    for text_word, ipa_word in zip(text_words, ipa_words):
        try:
            # Align this word
            word_text, word_ipa = align_word(text_word, ipa_word)
            
            # Add the aligned characters and phonemes
            for char, phoneme in zip(word_text, word_ipa.split(' ')):
                aligned_chars.append(char)
                aligned_phonemes.append(phoneme)
            
            # Add space between words
            aligned_chars.append(' ')
            aligned_phonemes.append('')  # Empty phoneme for space
            
        except Exception as e:
            print(f"  Error aligning word '{text_word}' with '{ipa_word}': {e}")
            return None, None
    
    # Remove trailing space
    if aligned_chars and aligned_chars[-1] == ' ':
        aligned_chars.pop()
        aligned_phonemes.pop()
    
    # Join into strings
    aligned_text = ''.join(aligned_chars)
    aligned_ipa = ' '.join(aligned_phonemes)
    
    return aligned_text, aligned_ipa


def prepare_dataset(input_file: str, output_file: str):
    """
    Prepare dataset by aligning sentences to character-level IPA.
    
    Args:
        input_file: Path to input TSV file (text<TAB>ipa sentence-level)
        output_file: Path to output TSV file (text<TAB>ipa character-level aligned)
    """
    print(f"\n{'='*70}")
    print("Dataset Preparation: Aligning Sentences")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Loaded {len(lines)} lines from input file")
    
    # Process each line
    processed = []
    skipped = 0
    
    for line_num, line in enumerate(tqdm(lines, desc="Processing sentences"), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"Warning: Skipping line {line_num}, expected 2 columns, got {len(parts)}")
            skipped += 1
            continue
        
        text, ipa = parts
        
        # Align sentence
        aligned_text, aligned_ipa = split_sentence(text, ipa)
        
        if aligned_text is None or aligned_ipa is None:
            print(f"Warning: Skipping line {line_num}, alignment failed")
            skipped += 1
            continue
        
        processed.append(f"{aligned_text}\t{aligned_ipa}")
    
    # Write output file
    print(f"\nWriting {len(processed)} aligned samples to output file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed))
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Processed: {len(processed)} samples")
    print(f"   Skipped:   {skipped} samples")
    print(f"{'='*70}\n")
    
    # Show some examples
    if processed:
        print("Example aligned samples (first 3):")
        for i, sample in enumerate(processed[:3], 1):
            text, ipa = sample.split('\t')
            print(f"\n{i}. Text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"   IPA:  {ipa[:80]}{'...' if len(ipa) > 80 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset: align sentences to character-level IPA')
    parser.add_argument('--input', type=str, required=True,
                        help='Input TSV file (text<TAB>ipa sentence-level)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output TSV file (text<TAB>ipa character-level aligned)')
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataset
    prepare_dataset(args.input, args.output)


if __name__ == '__main__':
    main()

