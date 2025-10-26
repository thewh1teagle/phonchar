"""
Full Pipeline Demo for Hebrew Phoneme Prediction Model

This demo shows the complete end-to-end pipeline for fine-tuning DictaBERT
to predict IPA phonemes for Hebrew text using a multi-head classification approach.
"""
import pytest
from src.tokenize import (
    encode, 
    decode, 
    Prediction, 
    reconstruct_sentence,
    CONSONANT_CLASSES,
    VOWEL_CLASSES
)
from src.preprocess import normalize_ipa, normalize_hebrew


class TestFullPipelineDemo:
    """
    Demonstrates the complete pipeline from raw text to phoneme predictions.
    
    Architecture Overview:
    - Base Model: DictaBERT (character-level BERT pretrained on Hebrew)
    - Task: Multi-head classification (4 heads per character)
    - Heads: consonant (24 classes), vowel (6 classes), stress (binary), flip_vowel (binary)
    - Training: Character-level labels, sentence-level context via BERT attention
    """
    
    def test_pipeline_step_by_step(self):
        """
        Complete pipeline walkthrough: text → encoding → training labels → 
        predictions → decoded IPA → final sentence
        """
        print("\n" + "="*70)
        print("FULL PIPELINE DEMO: Hebrew Phoneme Prediction")
        print("="*70)
        
        # Input: Hebrew sentence
        text = 'שלום עולם'
        print(f"\n📝 Input Text: {text}")
        print(f"   Characters: {[c for c in text]}")
        
        # Step 1: Prepare training data (aligned IPA per character)
        ipa_aligned = normalize_ipa('ʃa lˈo Ø m Ø ʔo lˈa Ø m')
        print(f"\n🎯 Training Target (aligned IPA per char):")
        print(f"   {ipa_aligned}")
        print(f"   Split: {ipa_aligned.split(' ')}")
        
        # Step 2: Encode to multi-head labels
        encoded = encode(text, ipa_aligned)
        print(f"\n🔢 Encoded Labels (multi-head):")
        print(f"   Consonants: {encoded.consonant}")
        print(f"   Vowels:     {encoded.vowel}")
        print(f"   Stress:     {encoded.stress}")
        print(f"   Flip:       {encoded.flip_vowel}")
        print(f"\n   Class mappings:")
        print(f"   - Consonants: {CONSONANT_CLASSES}")
        print(f"   - Vowels: {VOWEL_CLASSES}")
        
        # Step 3: Simulate predictions (in real model, these come from forward pass)
        predictions = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        print(f"\n🤖 Model Predictions (simulated as perfect here):")
        print(f"   Shape: {len(predictions.consonant)} characters")
        
        # Step 4: Decode predictions back to per-character IPA
        ipa_decoded = decode(text, predictions, preserve_unknown=True)
        print(f"\n📤 Decoded IPA (per-character):")
        print(f"   {ipa_decoded}")
        
        # Step 5: Reconstruct final sentence
        final_sentence = reconstruct_sentence(text, ipa_decoded)
        print(f"\n✨ Final Output (natural sentence):")
        print(f"   {final_sentence}")
        print(f"\n   Explanation: Phonemes joined per word, spaces preserved")
        
        print("\n" + "="*70)
        
        # Verify correctness
        # Note: decoded has double space for space character
        assert ipa_decoded == 'ʃa lˈo Ø m   ʔo lˈa Ø m'
        assert final_sentence == 'ʃalˈom ʔolˈam'
    
    def test_mixed_content_pipeline(self):
        """
        Demonstrate handling of mixed Hebrew-English content.
        Shows how non-Hebrew characters are handled differently.
        """
        print("\n" + "="*70)
        print("MIXED CONTENT DEMO: Hebrew + English")
        print("="*70)
        
        text = 'hello שלום world'
        print(f"\n📝 Input: {text}")
        
        # Training format: non-Hebrew gets Ø labels
        ipa_training = normalize_ipa('Ø Ø Ø Ø Ø Ø ʃa lˈo Ø m Ø Ø Ø Ø Ø Ø')
        print(f"\n🎯 Training IPA (Ø for non-Hebrew): {ipa_training}")
        
        # Encode
        encoded = encode(text, ipa_training)
        print(f"\n🔢 Encoded Labels:")
        print(f"   Consonants: {encoded.consonant}")
        print(f"   Note: -100 values are ignored in loss (non-Hebrew chars)")
        print(f"   -100 positions: {[i for i, c in enumerate(encoded.consonant) if c == -100]}")
        
        # Decode with preserve_unknown=True
        predictions = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        ipa_decoded = decode(text, predictions, preserve_unknown=True)
        print(f"\n📤 Decoded (preserve_unknown=True):")
        print(f"   {ipa_decoded}")
        print(f"   English chars preserved as-is")
        
        # Reconstruct
        final = reconstruct_sentence(text, ipa_decoded)
        print(f"\n✨ Final: {final}")
        print(f"   Hebrew: ʃalˈom (phonemes joined)")
        print(f"   English: hello, world (preserved)")
        
        print("\n" + "="*70)
        
        assert final == 'hello ʃalˈom world'
    
    def test_context_aware_learning(self):
        """
        Demonstrate why we use BERT (context-aware) instead of character-only model.
        Same letter can have different phonemes based on context.
        """
        print("\n" + "="*70)
        print("CONTEXT-AWARE LEARNING DEMO")
        print("="*70)
        
        # Example: ו can be 'v' or 'o' depending on context
        test_cases = [
            ('ו', 'v', "consonant usage"),
            ('שלום', 'ʃa lˈo Ø m', "ו as vowel 'o' in middle"),
        ]
        
        print("\n💡 Why BERT Context Matters:")
        print("   Same Hebrew letter can have different pronunciations")
        print("   depending on surrounding characters.\n")
        
        for text, ipa_target, description in test_cases:
            ipa_norm = normalize_ipa(ipa_target)
            print(f"   Text: {text:10s} → IPA: {ipa_norm:20s} ({description})")
        
        print("\n   BERT's attention mechanism provides context from:")
        print("   - Previous characters")
        print("   - Following characters")
        print("   - Word boundaries")
        print("   - This enables accurate phoneme prediction!")
        
        print("\n" + "="*70)
    
    def test_training_vs_inference_modes(self):
        """
        Show the difference between training format and inference output.
        """
        print("\n" + "="*70)
        print("TRAINING vs INFERENCE MODES")
        print("="*70)
        
        text = 'App שלום 2024'
        ipa_training = normalize_ipa('Ø Ø Ø Ø ʃa lˈo Ø m Ø Ø Ø Ø Ø')
        
        print(f"\n📝 Input: {text}")
        
        # Training mode: preserve_unknown=False
        encoded = encode(text, ipa_training)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        
        training_output = decode(text, preds, preserve_unknown=False)
        print(f"\n📚 Training Format (preserve_unknown=False):")
        print(f"   {training_output}")
        print(f"   All non-Hebrew → Ø (consistent labels)")
        
        # Inference mode: preserve_unknown=True
        inference_output = decode(text, preds, preserve_unknown=True)
        print(f"\n🚀 Inference Format (preserve_unknown=True):")
        print(f"   {inference_output}")
        print(f"   Non-Hebrew preserved as-is")
        
        # Final reconstruction
        final = reconstruct_sentence(text, inference_output)
        print(f"\n✨ User-Facing Output:")
        print(f"   {final}")
        print(f"   Natural sentence with phonemes")
        
        print("\n" + "="*70)
        
        assert final == 'App ʃalˈom 2024'
    
    def test_model_architecture_summary(self):
        """
        Display the model architecture and training details.
        """
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        
        print("\n🏗️  Architecture:")
        print("   Base: DictaBERT (dicta-il/dictabert-large-char-menaked)")
        print("   - Character-level BERT pretrained on Hebrew")
        print("   - Hidden size: 1024")
        print("   - Max length: 512 characters")
        
        print("\n📊 Custom Phoneme Head (4 classifiers):")
        print(f"   1. Consonant: Linear(1024 → {len(CONSONANT_CLASSES)}) classes")
        print(f"      Classes: {CONSONANT_CLASSES[:5]}... + Ø")
        print(f"   2. Vowel: Linear(1024 → {len(VOWEL_CLASSES)}) classes")
        print(f"      Classes: {VOWEL_CLASSES}")
        print(f"   3. Stress: Linear(1024 → 2) [binary]")
        print(f"      Classes: [no-stress, stress-ˈ]")
        print(f"   4. Flip Vowel: Linear(1024 → 2) [binary]")
        print(f"      Classes: [consonant-first, vowel-first]")
        
        print("\n📉 Loss Function:")
        print("   Combined CrossEntropyLoss:")
        print("   loss = consonant_loss + vowel_loss + stress_loss + flip_loss")
        print("   - Ignores -100 labels (non-Hebrew chars)")
        
        print("\n🎓 Training:")
        print("   - Input: Full sentences (context-aware)")
        print("   - Labels: Per-character (4 labels per char)")
        print("   - Batch processing via HuggingFace Trainer")
        print("   - Option: freeze BERT / full fine-tuning")
        
        print("\n📈 Evaluation Metrics:")
        print("   - Per-head accuracy (consonant, vowel, stress, flip)")
        print("   - Overall accuracy (all 4 heads correct)")
        print("   - Character-level evaluation")
        
        print("\n" + "="*70)
    
    def test_data_format_example(self):
        """
        Show the expected data format for training.
        """
        print("\n" + "="*70)
        print("DATA FORMAT")
        print("="*70)
        
        print("\n📁 Training Data Format (TSV):")
        print("   word<TAB>ipa")
        print("   שלום<TAB>ʃa lˈo Ø m")
        print("   ערב<TAB>ʔˈe re v")
        
        print("\n🔍 IPA Format Requirements:")
        print("   1. Space-separated (one phoneme group per character)")
        print("   2. Normalized (r→ʁ, g→ɡ, x→χ)")
        print("   3. Aligned (text length = IPA parts length)")
        print("   4. Silent chars use 'Ø'")
        
        print("\n💾 Example Processing:")
        text = 'שלום'
        ipa = 'ʃa lˈo Ø m'
        print(f"   Text: {text}")
        print(f"   IPA:  {ipa}")
        print(f"   Chars: {len(text)} = IPA parts: {len(ipa.split(' '))}")
        
        encoded = encode(text, ipa)
        print(f"\n   Encoded shapes:")
        print(f"   - consonant: {len(encoded.consonant)} labels")
        print(f"   - vowel:     {len(encoded.vowel)} labels")
        print(f"   - stress:    {len(encoded.stress)} labels")
        print(f"   - flip:      {len(encoded.flip_vowel)} labels")
        
        print("\n" + "="*70)


def test_quick_demo():
    """Quick one-liner demo for presentations"""
    print("\n" + "="*70)
    print("🚀 QUICK DEMO")
    print("="*70)
    
    from src.tokenize import encode, decode, Prediction, reconstruct_sentence
    
    # Hebrew text
    text = 'שלום עולם'
    
    # Training data (aligned IPA)
    ipa = 'ʃa lˈo Ø m Ø ʔo lˈa Ø m'
    
    # Encode → Predict → Decode → Reconstruct
    encoded = encode(text, ipa)
    preds = Prediction(encoded.consonant, encoded.vowel, encoded.stress, encoded.flip_vowel)
    decoded = decode(text, preds, preserve_unknown=True)
    final = reconstruct_sentence(text, decoded)
    
    print(f"\nInput:  {text}")
    print(f"Output: {final}")
    print("\n✨ Hebrew → IPA Phonemes")
    
    print("="*70 + "\n")
    
    assert final == 'ʃalˈom ʔolˈam'


if __name__ == '__main__':
    # Run all demos
    pytest.main([__file__, '-v', '-s'])

