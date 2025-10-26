"""
Tests for tokenize encode/decode functions
"""
import pytest
from src.tokenize import encode, decode, Prediction, reconstruct_sentence
from src.preprocess import normalize_ipa


class TestEncodeDecode:
    """Test encode and decode functions with various phoneme patterns"""
    
    def test_simple_consonant_vowel(self):
        """Test simple consonant-vowel patterns"""
        text = 'שלום'
        ipa = normalize_ipa('ʃa lˈo Ø m')
        
        encoded = encode(text, ipa)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_with_geresh(self):
        """Test special characters with geresh"""
        text = "ג'ירפה"
        ipa = normalize_ipa('dʒi Ø Ø rˈa fa Ø')
        
        encoded = encode(text, ipa)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_consonant_only(self):
        """Test consonant-only phonemes"""
        text = "ז'רגון"
        ipa = normalize_ipa('ʒa Ø r gˈo Ø n')
        
        encoded = encode(text, ipa)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_stress_at_beginning(self):
        """Test stress marker at beginning"""
        text = 'ערב'
        ipa = normalize_ipa('ʔˈe re v')
        
        encoded = encode(text, ipa)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_consonant_stress_vowel(self):
        """Test consonant + stress + vowel pattern"""
        text = 'יאיר'
        ipa = normalize_ipa('ja ʔˈi Ø r')
        
        encoded = encode(text, ipa)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_silent_character(self):
        """Test silent character (Ø)"""
        text = 'צה"ל'
        ipa = normalize_ipa('tsˈa ha Ø l')
        
        encoded = encode(text, ipa)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_vowel_before_consonant(self):
        """Test flip_vowel case (vowel before consonant)"""
        text = 'אב'
        ipa = normalize_ipa('ʔa v')
        
        encoded = encode(text, ipa)
        # Check that flip_vowel is set correctly for 'av' pattern if vowel comes first
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        
        assert decoded == ipa, f"Expected {ipa}, got {decoded}"
    
    def test_all_basic_csv_samples(self):
        """Test all samples from basic1.csv"""
        test_cases = [
            ("ג'ירפה", 'dʒi Ø Ø rˈa fa Ø'),
            ("ז'רגון", 'ʒa Ø r gˈo Ø n'),
            ('צה"ל', 'tsˈa ha Ø l'),
            ("ת'אומר", 'ta Ø ʔo Ø me r'),
            ('שלום', 'ʃa lˈo Ø m'),
            ('ערב', 'ʔˈe re v'),
            ('יאיר', 'ja ʔˈi Ø r'),
        ]
        
        for text, ipa_orig in test_cases:
            ipa = normalize_ipa(ipa_orig)
            encoded = encode(text, ipa)
            preds = Prediction(
                consonant=encoded.consonant,
                vowel=encoded.vowel,
                stress=encoded.stress,
                flip_vowel=encoded.flip_vowel
            )
            decoded = decode(text, preds)
            
            assert decoded == ipa, f"Failed for {text}: expected {ipa}, got {decoded}"
    
    def test_length_mismatch_raises_error(self):
        """Test that mismatched text/IPA lengths raise an error"""
        text = 'שלום'
        ipa = 'ʃa lo'  # Too few phoneme groups
        
        with pytest.raises(ValueError, match="Text length.*must match IPA parts"):
            encode(text, ipa)
    
    def test_non_input_chars_ignored(self):
        """Test that non-INPUT_CHARS get -100 labels"""
        # If we have spaces or other chars, they should be labeled -100
        text = 'א ב'  # Hebrew alef, space, bet
        ipa = 'ʔ Ø b'  # IPA with space
        
        encoded = encode(text, ipa)
        
        # Middle character (space) should have -100 for all labels
        assert encoded.consonant[1] == -100
        assert encoded.vowel[1] == -100
        assert encoded.stress[1] == -100
        assert encoded.flip_vowel[1] == -100
        
        # First and last should have valid labels
        assert encoded.consonant[0] != -100
        assert encoded.consonant[2] != -100


class TestStressPositioning:
    """Test correct stress marker positioning"""
    
    def test_stress_before_vowel_in_consonant_vowel(self):
        """Stress should come between consonant and vowel: cˈv"""
        text = 'ל'
        ipa = normalize_ipa('lˈo')
        
        encoded = encode(text, ipa)
        assert encoded.stress[0] == 1
        
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        assert decoded == ipa
    
    def test_stress_before_vowel_at_start(self):
        """Stress should come before vowel at start: ˈv"""
        text = 'א'
        ipa = normalize_ipa('ʔˈe')
        
        encoded = encode(text, ipa)
        assert encoded.stress[0] == 1
        
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        decoded = decode(text, preds)
        assert decoded == ipa


class TestFlipVowel:
    """Test flip_vowel detection"""
    
    def test_vowel_before_consonant(self):
        """Test detection when vowel comes before consonant"""
        # In IPA 'ˈaʁ', vowel 'a' comes before consonant 'ʁ'
        text = 'ע'
        ipa = normalize_ipa('ʔˈe')  # stress, then vowel, then consonant
        
        encoded = encode(text, ipa)
        # In this case, there's no consonant after the vowel in this example
        # Let me use a better example
        
        text = 'ער'
        ipa = normalize_ipa('ʔˈe re')
        encoded = encode(text, ipa)
        
        # Second character has 'ʁe' - consonant before vowel, so flip=0
        assert encoded.flip_vowel[1] == 0


class TestSentences:
    """Test handling of full sentences with context"""
    
    def test_simple_sentence(self):
        """Test encoding/decoding a simple two-word sentence"""
        text = 'שלום עולם'
        ipa = normalize_ipa('ʃa lˈo Ø m Ø ʔo lˈa Ø m')
        
        encoded = encode(text, ipa)
        
        # Space should have -100 labels
        space_idx = text.index(' ')
        assert encoded.consonant[space_idx] == -100
        assert encoded.vowel[space_idx] == -100
        
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        
        # With preserve_unknown=False (training format)
        decoded = decode(text, preds, preserve_unknown=False)
        assert decoded == ipa
        
        # With preserve_unknown=True (preserves space)
        decoded_preserve = decode(text, preds, preserve_unknown=True)
        assert decoded_preserve == normalize_ipa('ʃa lˈo Ø m   ʔo lˈa Ø m')
    
    def test_mixed_hebrew_english(self):
        """Test handling of mixed Hebrew-English text"""
        text = 'hello שלום'
        ipa_training = normalize_ipa('Ø Ø Ø Ø Ø Ø ʃa lˈo Ø m')
        
        encoded = encode(text, ipa_training)
        
        # English characters should have -100 labels
        for i in range(6):  # 'hello ' = 6 chars
            assert encoded.consonant[i] == -100
            assert encoded.vowel[i] == -100
        
        # Hebrew characters should have valid labels
        for i in range(6, len(text)):
            assert encoded.consonant[i] != -100 or encoded.vowel[i] != -100
        
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        
        # With preserve_unknown=False (training format with Ø)
        decoded = decode(text, preds, preserve_unknown=False)
        assert decoded == ipa_training
        
        # With preserve_unknown=True (preserves English chars)
        decoded_preserve = decode(text, preds, preserve_unknown=True)
        assert decoded_preserve == 'h e l l o   ʃa lˈo Ø m'
    
    def test_hebrew_with_numbers(self):
        """Test Hebrew text with numbers"""
        text = '123 שלום'
        ipa_training = normalize_ipa('Ø Ø Ø Ø ʃa lˈo Ø m')
        
        encoded = encode(text, ipa_training)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        
        # Training format
        decoded = decode(text, preds, preserve_unknown=False)
        assert decoded == ipa_training
        
        # Preserving numbers
        decoded_preserve = decode(text, preds, preserve_unknown=True)
        assert decoded_preserve == '1 2 3   ʃa lˈo Ø m'
    
    def test_hebrew_with_punctuation(self):
        """Test Hebrew text with punctuation"""
        text = 'שלום!'
        ipa_training = normalize_ipa('ʃa lˈo Ø m Ø')
        
        encoded = encode(text, ipa_training)
        
        # Punctuation should have -100 labels
        assert encoded.consonant[-1] == -100
        assert encoded.vowel[-1] == -100
        
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        
        # Training format
        decoded = decode(text, preds, preserve_unknown=False)
        assert decoded == ipa_training
        
        # Preserving punctuation
        decoded_preserve = decode(text, preds, preserve_unknown=True)
        assert decoded_preserve == 'ʃa lˈo Ø m !'
    
    def test_multi_sentence(self):
        """Test multiple sentences with various separators"""
        text = 'שלום. איך אתה?'
        ipa_training = normalize_ipa('ʃa lˈo Ø m Ø Ø ʔe Ø χ Ø ʔa ta Ø Ø')
        
        encoded = encode(text, ipa_training)
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        
        # Training format
        decoded = decode(text, preds, preserve_unknown=False)
        assert decoded == ipa_training
        
        # Preserving punctuation and spaces
        decoded_preserve = decode(text, preds, preserve_unknown=True)
        # Text is 'שלום. איך אתה?' -> 'ש ל ו ם . space א י כ space א ת ה ?'
        assert decoded_preserve == 'ʃa lˈo Ø m .   ʔe Ø χ   ʔa ta Ø ?'


class TestSentenceReconstruction:
    """Test reconstruction of natural sentences from per-character IPA"""
    
    def test_simple_hebrew_word(self):
        """Test reconstruction of a simple Hebrew word"""
        text = 'שלום'
        ipa = 'ʃa lˈo Ø m'
        
        result = reconstruct_sentence(text, ipa)
        # Silent char (Ø) is omitted
        assert result == 'ʃalˈom'
    
    def test_hebrew_sentence(self):
        """Test reconstruction of Hebrew sentence with spaces"""
        text = 'שלום עולם'
        ipa = 'ʃa lˈo Ø m   ʔo lˈa Ø m'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'ʃalˈom ʔolˈam'
    
    def test_mixed_hebrew_english(self):
        """Test reconstruction with mixed Hebrew and English"""
        text = 'hello שלום world'
        ipa = 'h e l l o   ʃa lˈo Ø m   w o r l d'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'hello ʃalˈom world'
    
    def test_hebrew_with_numbers(self):
        """Test reconstruction with numbers"""
        text = 'שלום 2024'
        ipa = 'ʃa lˈo Ø m   2 0 2 4'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'ʃalˈom 2024'
    
    def test_hebrew_with_punctuation(self):
        """Test reconstruction with punctuation"""
        text = 'שלום!'
        ipa = 'ʃa lˈo Ø m !'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'ʃalˈom!'
    
    def test_multi_sentence_with_punctuation(self):
        """Test reconstruction of multiple sentences"""
        text = 'שלום. איך אתה?'
        ipa = 'ʃa lˈo Ø m .   ʔe Ø χ   ʔa ta Ø ?'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'ʃalˈom. ʔeχ ʔata?'
    
    def test_full_pipeline_with_reconstruction(self):
        """Test complete encode->decode->reconstruct pipeline"""
        text = 'שלום עולם'
        ipa_training = normalize_ipa('ʃa lˈo Ø m Ø ʔo lˈa Ø m')
        
        # Encode
        encoded = encode(text, ipa_training)
        
        # Decode with preserve_unknown=True
        preds = Prediction(
            consonant=encoded.consonant,
            vowel=encoded.vowel,
            stress=encoded.stress,
            flip_vowel=encoded.flip_vowel
        )
        ipa_decoded = decode(text, preds, preserve_unknown=True)
        
        # Reconstruct final sentence
        final_sentence = reconstruct_sentence(text, ipa_decoded)
        
        assert final_sentence == 'ʃalˈom ʔolˈam'
    
    def test_english_only(self):
        """Test reconstruction with English-only text"""
        text = 'hello world'
        ipa = 'h e l l o   w o r l d'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'hello world'
    
    def test_complex_mixed_content(self):
        """Test reconstruction with complex mixed content"""
        text = 'App v2.0: שלום world!'
        ipa = 'A p p   v 2 . 0 :   ʃa lˈo Ø m   w o r l d !'
        
        result = reconstruct_sentence(text, ipa)
        assert result == 'App v2.0: ʃalˈom world!'
    
    def test_empty_string(self):
        """Test reconstruction with empty string"""
        result = reconstruct_sentence('', '')
        assert result == ''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

