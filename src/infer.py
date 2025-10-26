"""
Inference script for phoneme prediction

uv run python -m src.infer --model ./output/final
"""
import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.model import BertForPhonemeClassification
from src.preprocess import normalize_hebrew


def main():
    parser = argparse.ArgumentParser(description='Predict IPA phonemes for Hebrew text')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--input', type=str,
                        help='Input text file (one sentence per line). If not provided, reads from stdin.')
    parser.add_argument('--output', type=str,
                        help='Output file for predictions. If not provided, writes to stdout.')
    parser.add_argument('--format', type=str, choices=['ipa', 'both'], default='both',
                        help='Output format: "ipa" (only IPA) or "both" (text + IPA)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Inference batch size')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model}...", file=sys.stderr)
    model = BertForPhonemeClassification.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Using device: {device}", file=sys.stderr)
    
    # Read input
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        # Default to example Hebrew text
        sentences = ["בוא תרד לאכול, יש בורקס עם תרד!"]
        print("Using default text: שלום עולם", file=sys.stderr)
    
    if not sentences:
        print("No input provided.", file=sys.stderr)
        return
    
    print(f"Processing {len(sentences)} sentences...", file=sys.stderr)
    
    # Normalize sentences
    normalized_sentences = [normalize_hebrew(s) for s in sentences]
    
    # Process in batches
    all_ipas = []
    for i in range(0, len(normalized_sentences), args.batch_size):
        batch = normalized_sentences[i:i + args.batch_size]
        ipas = model.predict(batch, tokenizer)
        all_ipas.extend(ipas)
    
    # Write output
    output_file = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    
    try:
        for sentence, ipa in zip(sentences, all_ipas):
            if args.format == 'ipa':
                print(ipa, file=output_file)
            else:  # both
                print(f"{sentence}\t{ipa}", file=output_file)
    finally:
        if args.output:
            output_file.close()
    
    if args.output:
        print(f"Results written to {args.output}", file=sys.stderr)
    
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()

