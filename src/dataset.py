"""
Dataset preparation for phoneme prediction training
"""
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from src.alignment import align_word
from src.tokenize import encode
from src.preprocess import normalize_hebrew, normalize_ipa


class PhonemeDataset(Dataset):
    """PyTorch Dataset for phoneme prediction"""
    
    def __init__(self, texts: List[str], ipas: List[str], pre_aligned: bool = True):
        """
        Args:
            texts: List of Hebrew text strings
            ipas: List of IPA strings
            pre_aligned: If True, IPA is already aligned (space-separated per char).
                        If False, will normalize and align using align_word.
        """
        self.samples = []
        
        for text, ipa in zip(texts, ipas):
            if pre_aligned:
                # IPA is already aligned (space-separated per character)
                # But still need to normalize characters (r→ʁ, g→ɡ, x→χ)
                text_normalized = normalize_hebrew(text)
                aligned_ipa = normalize_ipa(ipa)
                aligned_text = text_normalized
            else:
                # Need to align word to get character-level phonemes
                text_normalized = normalize_hebrew(text)
                ipa_normalized = normalize_ipa(ipa)
                aligned_text, aligned_ipa = align_word(text_normalized, ipa_normalized)
            
            # Encode to training labels
            encoded = encode(aligned_text, aligned_ipa)
            
            self.samples.append({
                'text': aligned_text,
                'consonant': encoded.consonant,
                'vowel': encoded.vowel,
                'stress': encoded.stress,
                'flip_vowel': encoded.flip_vowel,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_dataset(input_file: str, split_ratio: float = 0.9, pre_aligned: bool = True) -> tuple[HFDataset, HFDataset]:
    """
    Prepare training and validation datasets from tab-separated file.
    
    Args:
        input_file: Path to file with format: text<TAB>ipa
        split_ratio: Ratio of training data (default 0.9)
        pre_aligned: If True, IPA is already aligned (space-separated per char).
                    If False, will normalize and align using align_word.
    
    Returns:
        (train_dataset, val_dataset) as HuggingFace Datasets
    """
    texts = []
    ipas = []
    
    # Read file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Skipping line {line_num}, expected 2 columns, got {len(parts)}")
                continue
            
            text, ipa = parts
            texts.append(text)
            ipas.append(ipa)
    
    print(f"Loaded {len(texts)} samples from {input_file}")
    
    # Create PyTorch dataset
    dataset = PhonemeDataset(texts, ipas, pre_aligned=pre_aligned)
    
    # Convert to HuggingFace Dataset format
    all_samples = [dataset[i] for i in range(len(dataset))]
    
    # Split into train/val
    split_idx = int(len(all_samples) * split_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Create HF Datasets
    train_dataset = HFDataset.from_list(train_samples)
    val_dataset = HFDataset.from_list(val_samples) if val_samples else None
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_data_collator(tokenizer):
    """
    Create a data collator for batching samples with DictaBERT tokenizer.
    
    Args:
        tokenizer: DictaBERT tokenizer instance
    
    Returns:
        Collator function
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of dicts with keys: text, consonant, vowel, stress, flip_vowel
        
        Returns:
            Dict with tokenized inputs and label tensors
        """
        texts = [sample['text'] for sample in batch]
        
        # Tokenize texts
        encoded_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        offset_mapping = encoded_inputs.pop('offset_mapping')
        
        # Prepare labels - align character-level labels to token-level
        max_len = encoded_inputs['input_ids'].shape[1]
        batch_size = len(batch)
        
        # Initialize label tensors with -100 (ignore index)
        consonant_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        vowel_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        stress_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        flip_vowel_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # Map character labels to token positions
        for batch_idx, (sample, offsets) in enumerate(zip(batch, offset_mapping)):
            text = sample['text']
            
            for token_idx, (start, end) in enumerate(offsets):
                # Skip special tokens (offset is (0, 0))
                if start == end:
                    continue
                
                # DictaBERT is character-level, so typically one char per token
                # Use the first character in the token span
                char_idx = start
                
                if char_idx < len(text):
                    # Copy labels from character position
                    if char_idx < len(sample['consonant']):
                        consonant_labels[batch_idx, token_idx] = sample['consonant'][char_idx]
                        vowel_labels[batch_idx, token_idx] = sample['vowel'][char_idx]
                        stress_labels[batch_idx, token_idx] = sample['stress'][char_idx]
                        flip_vowel_labels[batch_idx, token_idx] = sample['flip_vowel'][char_idx]
        
        # Combine all labels into a single dict
        encoded_inputs['labels'] = {
            'consonant': consonant_labels,
            'vowel': vowel_labels,
            'stress': stress_labels,
            'flip_vowel': flip_vowel_labels,
        }
        
        return encoded_inputs
    
    return collate_fn

