"""
Demo script to test dataset loading with progress bars
"""
import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import prepare_dataset

# Create a small sample dataset
def create_sample_dataset(num_samples=50):
    """Create a temporary TSV file with sample data"""
    sample_data = []
    
    # Simple Hebrew words with aligned IPA
    words = [
        ('שלום', 'ʃa lˈo Ø m'),
        ('עולם', 'ʔo lˈa Ø m'),
        ('ישראל', 'j i sʁa ʔˈe l'),
        ('תודה', 'to dˈa Ø'),
        ('בוקר', 'bˈo Ø Ø ke ʁ'),
    ]
    
    for i in range(num_samples):
        word, ipa = words[i % len(words)]
        sample_data.append(f"{word}\t{ipa}")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as f:
        f.write('\n'.join(sample_data))
        return f.name


def test_dataset_loading():
    """Test dataset loading with progress bars"""
    print("\n" + "="*70)
    print("DEMO: Dataset Loading with Progress Bars")
    print("="*70)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    temp_file = create_sample_dataset(50)
    print(f"   Created: {temp_file}")
    
    # Load dataset with progress bars
    print("\n2. Loading dataset (watch for progress bars)...")
    train_dataset, val_dataset = prepare_dataset(
        temp_file,
        split_ratio=0.8,
        pre_aligned=True
    )
    
    print("\n3. Dataset statistics:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"   Sample from training set:")
    sample = train_dataset[0]
    print(f"      Text: {sample['text']}")
    print(f"      Consonant labels: {sample['consonant'][:5]}... (showing first 5)")
    print(f"      Vowel labels: {sample['vowel'][:5]}... (showing first 5)")
    
    # Clean up
    os.unlink(temp_file)
    print(f"\n✅ Demo complete! Temporary file cleaned up.")
    print("="*70)


if __name__ == '__main__':
    test_dataset_loading()

