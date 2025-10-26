"""
Training script for phoneme prediction model
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/hedc4-phonemes_v1.txt.7z
sudo apt install p7zip-full -y
7z x hedc4-phonemes_v1.txt.7z
head -n 200 hedc4-phonemes.txt > dataset.txt
uv run python -m src.train --data dataset.txt

uv run python -m src.train --data dataset_aligned.txt --epochs 1 --batch-size 2
"""
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import jiwer
from src.model import BertForPhonemeClassification
from src.dataset import prepare_dataset, create_data_collator
from src.tokenize import CONSONANT_CLASSES, VOWEL_CLASSES, decode, Prediction, reconstruct_sentence
import numpy as np


def compute_metrics(eval_pred):
    """
    Compute accuracy metrics for each prediction head + WER/CER.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dict of metrics
    """
    predictions, labels = eval_pred
    
    # predictions is a tuple of logits from each head
    consonant_logits, vowel_logits, stress_logits, flip_vowel_logits = predictions
    
    # Get argmax predictions
    consonant_preds = np.argmax(consonant_logits, axis=-1)
    vowel_preds = np.argmax(vowel_logits, axis=-1)
    stress_preds = np.argmax(stress_logits, axis=-1)
    flip_vowel_preds = np.argmax(flip_vowel_logits, axis=-1)
    
    # labels is a dict with keys: consonant, vowel, stress, flip_vowel
    consonant_labels = labels['consonant']
    vowel_labels = labels['vowel']
    stress_labels = labels['stress']
    flip_vowel_labels = labels['flip_vowel']
    
    # Calculate accuracy (ignoring -100 labels)
    def accuracy_ignore_pad(preds, labels):
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).mean()
    
    metrics = {
        'consonant_accuracy': accuracy_ignore_pad(consonant_preds, consonant_labels),
        'vowel_accuracy': accuracy_ignore_pad(vowel_preds, vowel_labels),
        'stress_accuracy': accuracy_ignore_pad(stress_preds, stress_labels),
        'flip_vowel_accuracy': accuracy_ignore_pad(flip_vowel_preds, flip_vowel_labels),
    }
    
    # Overall accuracy (all heads correct)
    mask = consonant_labels != -100
    if mask.sum() > 0:
        all_correct = (
            (consonant_preds[mask] == consonant_labels[mask]) &
            (vowel_preds[mask] == vowel_labels[mask]) &
            (stress_preds[mask] == stress_labels[mask]) &
            (flip_vowel_preds[mask] == flip_vowel_labels[mask])
        )
        metrics['overall_accuracy'] = all_correct.mean()
    else:
        metrics['overall_accuracy'] = 0.0
    
    # Compute WER and CER on reconstructed sentences
    try:
        # Get the text data from the evaluation dataset
        # Note: This requires access to the original text, which we'll need to pass through
        # For now, we'll compute WER/CER on the phoneme sequences directly
        
        # Reconstruct predicted and reference phoneme sequences
        pred_sequences = []
        ref_sequences = []
        
        # Process each example in the batch
        batch_size = consonant_preds.shape[0]
        seq_len = consonant_preds.shape[1]
        
        for i in range(batch_size):
            # Get predictions for this example
            example_mask = consonant_labels[i] != -100
            
            if example_mask.sum() == 0:
                continue
            
            # Get predicted phoneme sequence
            pred_consonants = [CONSONANT_CLASSES[c] if 0 <= c < len(CONSONANT_CLASSES) else 'Ø' 
                              for c in consonant_preds[i][example_mask]]
            pred_vowels = [VOWEL_CLASSES[v] if 0 <= v < len(VOWEL_CLASSES) else 'Ø' 
                          for v in vowel_preds[i][example_mask]]
            pred_stress = stress_preds[i][example_mask]
            pred_flip = flip_vowel_preds[i][example_mask]
            
            # Build predicted phoneme string
            pred_phonemes = []
            for c, v, s, f in zip(pred_consonants, pred_vowels, pred_stress, pred_flip):
                if c == 'Ø' and v == 'Ø':
                    pred_phonemes.append('Ø')
                else:
                    phoneme = []
                    if f == 1:  # vowel first
                        if s == 1:
                            phoneme.append('ˈ')
                        if v != 'Ø':
                            phoneme.append(v)
                        if c != 'Ø':
                            phoneme.append(c)
                    else:  # consonant first
                        if c != 'Ø':
                            phoneme.append(c)
                        if s == 1:
                            phoneme.append('ˈ')
                        if v != 'Ø':
                            phoneme.append(v)
                    pred_phonemes.append(''.join(phoneme))
            
            # Get reference phoneme sequence
            ref_consonants = [CONSONANT_CLASSES[c] if 0 <= c < len(CONSONANT_CLASSES) else 'Ø' 
                             for c in consonant_labels[i][example_mask]]
            ref_vowels = [VOWEL_CLASSES[v] if 0 <= v < len(VOWEL_CLASSES) else 'Ø' 
                         for v in vowel_labels[i][example_mask]]
            ref_stress = stress_labels[i][example_mask]
            ref_flip = flip_vowel_labels[i][example_mask]
            
            # Build reference phoneme string
            ref_phonemes = []
            for c, v, s, f in zip(ref_consonants, ref_vowels, ref_stress, ref_flip):
                if c == 'Ø' and v == 'Ø':
                    ref_phonemes.append('Ø')
                else:
                    phoneme = []
                    if f == 1:  # vowel first
                        if s == 1:
                            phoneme.append('ˈ')
                        if v != 'Ø':
                            phoneme.append(v)
                        if c != 'Ø':
                            phoneme.append(c)
                    else:  # consonant first
                        if c != 'Ø':
                            phoneme.append(c)
                        if s == 1:
                            phoneme.append('ˈ')
                        if v != 'Ø':
                            phoneme.append(v)
                    ref_phonemes.append(''.join(phoneme))
            
            pred_sequences.append(' '.join(pred_phonemes))
            ref_sequences.append(' '.join(ref_phonemes))
        
        # Compute WER and CER if we have sequences
        if pred_sequences and ref_sequences:
            # WER: Word Error Rate (treating each phoneme group as a "word")
            wer = jiwer.wer(ref_sequences, pred_sequences)
            metrics['wer'] = wer
            
            # CER: Character Error Rate (on the phoneme string level)
            # Join without spaces for character-level comparison
            pred_chars = [''.join(s.split()) for s in pred_sequences]
            ref_chars = [''.join(s.split()) for s in ref_sequences]
            cer = jiwer.cer(ref_chars, pred_chars)
            metrics['cer'] = cer
    
    except Exception as e:
        # If WER/CER computation fails, log and continue
        print(f"Warning: Could not compute WER/CER: {e}")
        metrics['wer'] = 0.0
        metrics['cer'] = 0.0
    
    return metrics


class PhonemeTrainer(Trainer):
    """Custom Trainer that handles multi-head outputs"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with multi-head labels"""
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for multi-head outputs"""
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Stack logits as tuple for compute_metrics
        logits_tuple = (
            logits.consonant_logits.cpu().numpy(),
            logits.vowel_logits.cpu().numpy(),
            logits.stress_logits.cpu().numpy(),
            logits.flip_vowel_logits.cpu().numpy(),
        )
        
        # Convert labels dict to dict of numpy arrays
        labels_dict = {
            'consonant': labels['consonant'].cpu().numpy(),
            'vowel': labels['vowel'].cpu().numpy(),
            'stress': labels['stress'].cpu().numpy(),
            'flip_vowel': labels['flip_vowel'].cpu().numpy(),
        }
        
        return (loss, logits_tuple, labels_dict)


def main():
    parser = argparse.ArgumentParser(description='Train phoneme prediction model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data file (text<TAB>ipa format)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for model checkpoints')
    parser.add_argument('--base-model', type=str, default='dicta-il/dictabert-large-char-menaked',
                        help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--freeze-bert', action='store_true',
                        help='Freeze BERT base layers (only train head)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phoneme Prediction Model Training")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Freeze BERT: {args.freeze_bert}")
    print(f"Validation split: {args.val_split}")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load and prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset = prepare_dataset(
        args.data,
        split_ratio=1.0 - args.val_split,
        pre_aligned=True  # Always expect pre-aligned data
    )
    
    # Load model config and initialize model
    print("\nInitializing model...")
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Add custom config for phoneme heads
    config.num_consonants = len(CONSONANT_CLASSES)
    config.num_vowels = len(VOWEL_CLASSES)
    
    # Initialize model with pretrained BERT
    model = BertForPhonemeClassification.from_pretrained(
        args.base_model,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True  # We're replacing the head
    )
    
    # Optionally freeze BERT layers
    if args.freeze_bert:
        print("Freezing BERT base layers...")
        for param in model.bert.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Training arguments
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 TensorBoard logging enabled at: {output_dir / 'logs'}")
    print(f"   Run: tensorboard --logdir {output_dir / 'logs'}")
    print()
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=10,  # Log every 10 steps (was 100)
        logging_first_step=True,  # Log the first step
        eval_strategy='no',  # Skip evaluation for now
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=False,  # Can't load best without eval
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to='tensorboard',  # Enable TensorBoard logging
        remove_unused_columns=False,  # Keep our custom columns
        disable_tqdm=False,  # Keep progress bar enabled
        logging_nan_inf_filter=False,  # Show all values
    )
    
    # Initialize trainer
    trainer = PhonemeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # Skip eval for now
        data_collator=data_collator,
        compute_metrics=None,  # Skip metrics computation
        callbacks=None,  # Skip callbacks
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {train_result.training_loss:.4f}")
    
    print(f"\nModel saved to: {output_dir / 'final'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

