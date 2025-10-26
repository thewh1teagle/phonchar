"""
Test compute_metrics function from train.py
"""
import pytest
import numpy as np
from src.train import compute_metrics
from src.tokenize import CONSONANT_CLASSES, VOWEL_CLASSES


class TestComputeMetrics:
    """Test the compute_metrics function used in training"""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        batch_size = 2
        seq_len = 5
        
        # Create perfect predictions (predictions == labels)
        consonant_labels = np.array([
            [21, 9, 24, 10, -100],  # ʃ l Ø m <pad>
            [18, 10, -100, -100, -100]  # ʔ m <pad> <pad> <pad>
        ])
        vowel_labels = np.array([
            [0, 3, 5, 5, -100],  # a o Ø Ø <pad>
            [1, 5, -100, -100, -100]  # e Ø <pad> <pad> <pad>
        ])
        stress_labels = np.array([
            [0, 1, 0, 0, -100],
            [1, 0, -100, -100, -100]
        ])
        flip_vowel_labels = np.array([
            [0, 0, 0, 0, -100],
            [0, 0, -100, -100, -100]
        ])
        
        # Create logits that will argmax to the labels
        def create_perfect_logits(labels, num_classes):
            logits = np.random.randn(batch_size, seq_len, num_classes) * 0.1
            for i in range(batch_size):
                for j in range(seq_len):
                    if labels[i, j] != -100:
                        logits[i, j, labels[i, j]] = 10.0  # High logit for correct class
            return logits
        
        consonant_logits = create_perfect_logits(consonant_labels, len(CONSONANT_CLASSES))
        vowel_logits = create_perfect_logits(vowel_labels, len(VOWEL_CLASSES))
        stress_logits = create_perfect_logits(stress_labels, 2)
        flip_vowel_logits = create_perfect_logits(flip_vowel_labels, 2)
        
        predictions = (consonant_logits, vowel_logits, stress_logits, flip_vowel_logits)
        labels = {
            'consonant': consonant_labels,
            'vowel': vowel_labels,
            'stress': stress_labels,
            'flip_vowel': flip_vowel_labels
        }
        
        # Compute metrics
        metrics = compute_metrics((predictions, labels))
        
        # Assert perfect accuracy
        assert metrics['consonant_accuracy'] == 1.0, "Perfect consonant predictions should have 1.0 accuracy"
        assert metrics['vowel_accuracy'] == 1.0, "Perfect vowel predictions should have 1.0 accuracy"
        assert metrics['stress_accuracy'] == 1.0, "Perfect stress predictions should have 1.0 accuracy"
        assert metrics['flip_vowel_accuracy'] == 1.0, "Perfect flip_vowel predictions should have 1.0 accuracy"
        assert metrics['overall_accuracy'] == 1.0, "Perfect predictions should have 1.0 overall accuracy"
        
        # WER and CER should be 0.0 (perfect)
        assert metrics['wer'] == 0.0, "Perfect predictions should have 0.0 WER"
        assert metrics['cer'] == 0.0, "Perfect predictions should have 0.0 CER"
        
        print("\n✅ Perfect predictions test passed!")
        print(f"   Metrics: {metrics}")
    
    def test_partial_errors(self):
        """Test metrics with some errors"""
        batch_size = 1
        seq_len = 4
        
        # Labels: ʃa lˈo Ø m
        consonant_labels = np.array([[21, 9, 24, 10]])  # ʃ l Ø m
        vowel_labels = np.array([[0, 3, 5, 5]])  # a o Ø Ø
        stress_labels = np.array([[0, 1, 0, 0]])  # no, yes, no, no
        flip_vowel_labels = np.array([[0, 0, 0, 0]])
        
        # Predictions with one error: missing stress on position 1
        # Create logits
        consonant_logits = np.random.randn(batch_size, seq_len, len(CONSONANT_CLASSES)) * 0.1
        vowel_logits = np.random.randn(batch_size, seq_len, len(VOWEL_CLASSES)) * 0.1
        stress_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
        flip_vowel_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
        
        # Set correct predictions except stress at position 1
        for j in range(seq_len):
            consonant_logits[0, j, consonant_labels[0, j]] = 10.0
            vowel_logits[0, j, vowel_labels[0, j]] = 10.0
            flip_vowel_logits[0, j, flip_vowel_labels[0, j]] = 10.0
            
            if j == 1:
                # Wrong stress prediction at position 1
                stress_logits[0, j, 0] = 10.0  # Predict 0 instead of 1
            else:
                stress_logits[0, j, stress_labels[0, j]] = 10.0
        
        predictions = (consonant_logits, vowel_logits, stress_logits, flip_vowel_logits)
        labels = {
            'consonant': consonant_labels,
            'vowel': vowel_labels,
            'stress': stress_labels,
            'flip_vowel': flip_vowel_labels
        }
        
        # Compute metrics
        metrics = compute_metrics((predictions, labels))
        
        # Consonant, vowel, flip should be perfect
        assert metrics['consonant_accuracy'] == 1.0
        assert metrics['vowel_accuracy'] == 1.0
        assert metrics['flip_vowel_accuracy'] == 1.0
        
        # Stress should have 75% accuracy (3/4 correct)
        assert metrics['stress_accuracy'] == 0.75, f"Expected 0.75, got {metrics['stress_accuracy']}"
        
        # Overall should be 75% (one character has error)
        assert metrics['overall_accuracy'] == 0.75, f"Expected 0.75, got {metrics['overall_accuracy']}"
        
        # WER and CER should be > 0 (not perfect)
        assert metrics['wer'] > 0.0, "Should have some WER with errors"
        assert metrics['cer'] > 0.0, "Should have some CER with errors"
        
        print("\n✅ Partial errors test passed!")
        print(f"   Metrics: {metrics}")
    
    def test_ignores_padding(self):
        """Test that -100 labels are properly ignored"""
        batch_size = 1
        seq_len = 6
        
        # Labels with padding (-100)
        consonant_labels = np.array([[21, 9, -100, -100, 24, 10]])
        vowel_labels = np.array([[0, 3, -100, -100, 5, 5]])
        stress_labels = np.array([[0, 1, -100, -100, 0, 0]])
        flip_vowel_labels = np.array([[0, 0, -100, -100, 0, 0]])
        
        # Create perfect predictions for non-padded positions
        consonant_logits = np.random.randn(batch_size, seq_len, len(CONSONANT_CLASSES)) * 0.1
        vowel_logits = np.random.randn(batch_size, seq_len, len(VOWEL_CLASSES)) * 0.1
        stress_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
        flip_vowel_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
        
        for j in range(seq_len):
            if consonant_labels[0, j] != -100:
                consonant_logits[0, j, consonant_labels[0, j]] = 10.0
                vowel_logits[0, j, vowel_labels[0, j]] = 10.0
                stress_logits[0, j, stress_labels[0, j]] = 10.0
                flip_vowel_logits[0, j, flip_vowel_labels[0, j]] = 10.0
        
        predictions = (consonant_logits, vowel_logits, stress_logits, flip_vowel_logits)
        labels = {
            'consonant': consonant_labels,
            'vowel': vowel_labels,
            'stress': stress_labels,
            'flip_vowel': flip_vowel_labels
        }
        
        # Compute metrics
        metrics = compute_metrics((predictions, labels))
        
        # Should be perfect on non-padded positions
        assert metrics['consonant_accuracy'] == 1.0
        assert metrics['vowel_accuracy'] == 1.0
        assert metrics['stress_accuracy'] == 1.0
        assert metrics['flip_vowel_accuracy'] == 1.0
        assert metrics['overall_accuracy'] == 1.0
        
        print("\n✅ Padding ignore test passed!")
        print(f"   Metrics: {metrics}")
    
    def test_wer_cer_computation(self):
        """Test WER and CER are computed correctly"""
        batch_size = 2
        seq_len = 4
        
        # Example 1: Perfect match
        # Labels: ʃa lˈo Ø m
        consonant_labels_1 = np.array([21, 9, 24, 10])  # ʃ l Ø m
        vowel_labels_1 = np.array([0, 3, 5, 5])  # a o Ø Ø
        stress_labels_1 = np.array([0, 1, 0, 0])
        flip_labels_1 = np.array([0, 0, 0, 0])
        
        # Example 2: One stress error
        # Labels: ʔˈe (should be) vs ʔe (predicted)
        consonant_labels_2 = np.array([18, -100, -100, -100])  # ʔ <pad> <pad> <pad>
        vowel_labels_2 = np.array([1, -100, -100, -100])  # e <pad> <pad> <pad>
        stress_labels_2 = np.array([1, -100, -100, -100])  # yes <pad> <pad> <pad>
        flip_labels_2 = np.array([0, -100, -100, -100])
        
        consonant_labels = np.vstack([consonant_labels_1, consonant_labels_2])
        vowel_labels = np.vstack([vowel_labels_1, vowel_labels_2])
        stress_labels = np.vstack([stress_labels_1, stress_labels_2])
        flip_vowel_labels = np.vstack([flip_labels_1, flip_labels_2])
        
        # Perfect predictions for example 1, missing stress for example 2
        consonant_logits = np.random.randn(batch_size, seq_len, len(CONSONANT_CLASSES)) * 0.1
        vowel_logits = np.random.randn(batch_size, seq_len, len(VOWEL_CLASSES)) * 0.1
        stress_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
        flip_vowel_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
        
        for i in range(batch_size):
            for j in range(seq_len):
                if consonant_labels[i, j] != -100:
                    consonant_logits[i, j, consonant_labels[i, j]] = 10.0
                    vowel_logits[i, j, vowel_labels[i, j]] = 10.0
                    flip_vowel_logits[i, j, flip_vowel_labels[i, j]] = 10.0
                    
                    # Make example 2 predict wrong stress
                    if i == 1 and j == 0:
                        stress_logits[i, j, 0] = 10.0  # Predict no stress
                    elif stress_labels[i, j] != -100:
                        stress_logits[i, j, stress_labels[i, j]] = 10.0
        
        predictions = (consonant_logits, vowel_logits, stress_logits, flip_vowel_logits)
        labels = {
            'consonant': consonant_labels,
            'vowel': vowel_labels,
            'stress': stress_labels,
            'flip_vowel': flip_vowel_labels
        }
        
        # Compute metrics
        metrics = compute_metrics((predictions, labels))
        
        # WER should reflect 1 error out of 5 total phoneme groups
        # (4 from example 1 + 1 from example 2)
        assert 0.0 < metrics['wer'] < 0.5, f"WER should be between 0 and 0.5, got {metrics['wer']}"
        
        # CER should be > 0 due to missing stress character
        assert metrics['cer'] > 0.0, f"CER should be > 0, got {metrics['cer']}"
        
        print("\n✅ WER/CER computation test passed!")
        print(f"   WER: {metrics['wer']:.4f}")
        print(f"   CER: {metrics['cer']:.4f}")
        print(f"   Full metrics: {metrics}")
    
    def test_all_padding(self):
        """Test behavior when all labels are padding"""
        batch_size = 1
        seq_len = 3
        
        # All labels are -100 (padding)
        consonant_labels = np.full((batch_size, seq_len), -100)
        vowel_labels = np.full((batch_size, seq_len), -100)
        stress_labels = np.full((batch_size, seq_len), -100)
        flip_vowel_labels = np.full((batch_size, seq_len), -100)
        
        consonant_logits = np.random.randn(batch_size, seq_len, len(CONSONANT_CLASSES))
        vowel_logits = np.random.randn(batch_size, seq_len, len(VOWEL_CLASSES))
        stress_logits = np.random.randn(batch_size, seq_len, 2)
        flip_vowel_logits = np.random.randn(batch_size, seq_len, 2)
        
        predictions = (consonant_logits, vowel_logits, stress_logits, flip_vowel_logits)
        labels = {
            'consonant': consonant_labels,
            'vowel': vowel_labels,
            'stress': stress_labels,
            'flip_vowel': flip_vowel_labels
        }
        
        # Compute metrics - should not crash
        metrics = compute_metrics((predictions, labels))
        
        # All accuracies should be 0.0 (no valid tokens to evaluate)
        assert metrics['consonant_accuracy'] == 0.0
        assert metrics['vowel_accuracy'] == 0.0
        assert metrics['stress_accuracy'] == 0.0
        assert metrics['flip_vowel_accuracy'] == 0.0
        assert metrics['overall_accuracy'] == 0.0
        
        print("\n✅ All padding test passed!")
        print(f"   Metrics: {metrics}")


def test_metrics_integration():
    """Integration test showing typical training scenario"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Typical Training Metrics")
    print("="*70)
    
    batch_size = 4
    seq_len = 8
    
    # Simulate realistic predictions with ~90% accuracy
    np.random.seed(42)
    
    consonant_labels = np.random.randint(0, len(CONSONANT_CLASSES), (batch_size, seq_len))
    vowel_labels = np.random.randint(0, len(VOWEL_CLASSES), (batch_size, seq_len))
    stress_labels = np.random.randint(0, 2, (batch_size, seq_len))
    flip_vowel_labels = np.random.randint(0, 2, (batch_size, seq_len))
    
    # Add some padding
    consonant_labels[:, -2:] = -100
    vowel_labels[:, -2:] = -100
    stress_labels[:, -2:] = -100
    flip_vowel_labels[:, -2:] = -100
    
    # Create mostly correct predictions
    consonant_logits = np.random.randn(batch_size, seq_len, len(CONSONANT_CLASSES)) * 0.1
    vowel_logits = np.random.randn(batch_size, seq_len, len(VOWEL_CLASSES)) * 0.1
    stress_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
    flip_vowel_logits = np.random.randn(batch_size, seq_len, 2) * 0.1
    
    # Make 90% of predictions correct
    for i in range(batch_size):
        for j in range(seq_len):
            if consonant_labels[i, j] != -100:
                if np.random.rand() < 0.9:  # 90% correct
                    consonant_logits[i, j, consonant_labels[i, j]] = 10.0
                    vowel_logits[i, j, vowel_labels[i, j]] = 10.0
                    stress_logits[i, j, stress_labels[i, j]] = 10.0
                    flip_vowel_logits[i, j, flip_vowel_labels[i, j]] = 10.0
    
    predictions = (consonant_logits, vowel_logits, stress_logits, flip_vowel_logits)
    labels = {
        'consonant': consonant_labels,
        'vowel': vowel_labels,
        'stress': stress_labels,
        'flip_vowel': flip_vowel_labels
    }
    
    # Compute metrics
    metrics = compute_metrics((predictions, labels))
    
    print("\n📊 Simulated Training Metrics (~90% accuracy):")
    print(f"   Consonant Accuracy: {metrics['consonant_accuracy']:.4f}")
    print(f"   Vowel Accuracy:     {metrics['vowel_accuracy']:.4f}")
    print(f"   Stress Accuracy:    {metrics['stress_accuracy']:.4f}")
    print(f"   Flip Accuracy:      {metrics['flip_vowel_accuracy']:.4f}")
    print(f"   Overall Accuracy:   {metrics['overall_accuracy']:.4f}")
    print(f"   WER:                {metrics['wer']:.4f}")
    print(f"   CER:                {metrics['cer']:.4f}")
    
    # Sanity checks
    assert 0.0 <= metrics['consonant_accuracy'] <= 1.0
    assert 0.0 <= metrics['wer'] <= 1.0
    assert 0.0 <= metrics['cer'] <= 1.0
    
    print("\n✅ Integration test passed!")
    print("="*70)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

