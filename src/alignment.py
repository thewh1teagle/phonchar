"""
Align word and IPA to pairs of character and phoneme
"""
from src import config
from src.preprocess import normalize_hebrew

def align_word(word: str, ipa: str) -> tuple[str, str]:
    """
    Align Hebrew word with its IPA transcription.
    
    Args:
        word: Hebrew word (will be normalized)
        ipa: Clean IPA string (no spaces, no Ø symbols)
    
    Returns:
        Tuple of (aligned_word, aligned_ipa) with spaces between character groups
    """
    word = normalize_hebrew(word)
    
    ipa_parts = []
    
    word_idx = 0
    ipa_idx = 0
    
    # Vowels and stress markers that can follow consonants
    VOWELS = set('aeiou')
    STRESS = 'ˈ'
    
    while word_idx < len(word):
        # Check for special character combinations (geresh/gershayim) FIRST
        is_special_combo = False
        num_chars = 1
        
        if word_idx + 1 < len(word):
            two_char = word[word_idx:word_idx + 2]
            if two_char in config.CHAR_TO_PHONEME:
                # Special two-character combination
                char_key = two_char
                is_special_combo = True
                num_chars = 2
            else:
                # Single character
                char_key = word[word_idx]
        else:
            # Single character at end
            char_key = word[word_idx]
        
        # Standalone geresh/gershayim are always silent (when not part of special combo)
        if not is_special_combo and (char_key == config.GERESH or char_key == config.GERSHAYIM):
            ipa_parts.append(config.NONE)
            word_idx += 1
            continue
        
        # Get possible phonemes for this character/combination
        possible_phonemes = config.CHAR_TO_PHONEME.get(char_key, [])
        
        # Try to match a consonant from the IPA
        matched = False
        phoneme_group = []
        
        if ipa_idx < len(ipa):
            # Try to match each possible phoneme (consonant) first
            for phoneme in possible_phonemes:
                # Check if we're at a vowel position and the phoneme can come after it
                start_idx = ipa_idx
                if ipa_idx < len(ipa) and ipa[ipa_idx] in VOWELS:
                    # Tentatively check if consonant follows the vowel
                    temp_idx = ipa_idx + 1
                    if temp_idx + len(phoneme) <= len(ipa) and ipa[temp_idx:temp_idx + len(phoneme)] == phoneme:
                        # Vowel + consonant pattern: consume both
                        phoneme_group.append(ipa[ipa_idx])
                        ipa_idx += 1
                        phoneme_group.append(phoneme)
                        ipa_idx += len(phoneme)
                        matched = True
                        break
                
                # Try to match consonant at current position
                if ipa_idx + len(phoneme) <= len(ipa) and ipa[ipa_idx:ipa_idx + len(phoneme)] == phoneme:
                    # Matched the consonant
                    phoneme_group.append(phoneme)
                    ipa_idx += len(phoneme)
                    matched = True
                    
                    # Consume stress markers and ONE vowel (not multiple)
                    while ipa_idx < len(ipa):
                        if ipa[ipa_idx] == STRESS:
                            phoneme_group.append(STRESS)
                            ipa_idx += 1
                        elif ipa[ipa_idx] in VOWELS:
                            # Consume only ONE vowel, then stop
                            phoneme_group.append(ipa[ipa_idx])
                            ipa_idx += 1
                            break  # Stop after consuming one vowel
                        else:
                            break
                    break
        
        # If special combination, assign phoneme to first char, Ø to second
        if is_special_combo:
            if matched:
                ipa_parts.append(''.join(phoneme_group))
            else:
                ipa_parts.append(config.NONE)
            ipa_parts.append(config.NONE)  # Second character gets Ø
            word_idx += 2
        else:
            # Single character
            if matched:
                ipa_parts.append(''.join(phoneme_group))
            else:
                ipa_parts.append(config.NONE)
            word_idx += 1
    
    # Join IPA parts with spaces, but return word unchanged
    aligned_word = word
    aligned_ipa = ' '.join(ipa_parts)
    
    return aligned_word, aligned_ipa