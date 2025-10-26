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
        
        # SPECIAL CASE: For matres lectionis (ה,ו,י,ע,א) at second-to-last position
        # OR for standalone gershayim/geresh before last char
        # Check if the LAST character can match the remaining IPA instead
        is_second_to_last = (word_idx + num_chars == len(word) - 1)
        should_check_last_char = (
            (is_second_to_last and char_key in ['ה', 'ו', 'י', 'ע', 'א']) or
            (is_second_to_last and char_key == config.GERSHAYIM) or
            (is_second_to_last and char_key == config.GERESH)
        )
        
        if should_check_last_char and ipa_idx < len(ipa):
            # Special heuristic: If last two characters are the SAME matres lectionis,
            # the second-to-last should match (not be silent).
            # Otherwise, check if second-to-last should be silent.
            last_char = word[-1]
            
            # If both characters are the same, don't apply the look-ahead logic
            # (let normal matching proceed)
            if char_key == last_char:
                pass  # Skip the look-ahead, let it match normally
            else:
                # Different characters: check if last char can match and make current silent
                last_char_phonemes = config.CHAR_TO_PHONEME.get(last_char, [])
                
                # Check if last character can match what's remaining in IPA
                last_char_can_match = False
                for last_phoneme in last_char_phonemes:
                    temp_idx = ipa_idx
                    
                    # Try various patterns for the last character
                    # Pattern 1: Just a standalone vowel
                    if temp_idx < len(ipa) and ipa[temp_idx] in VOWELS:
                        last_char_can_match = True
                        break
                    
                    # Pattern 2: [stress?] vowel consonant
                    if temp_idx < len(ipa) and ipa[temp_idx] == STRESS:
                        temp_idx += 1
                    if temp_idx < len(ipa) and ipa[temp_idx] in VOWELS:
                        if temp_idx + 1 + len(last_phoneme) <= len(ipa) and ipa[temp_idx + 1:temp_idx + 1 + len(last_phoneme)] == last_phoneme:
                            last_char_can_match = True
                            break
                    
                    # Pattern 3: [stress?] consonant [vowel?]
                    temp_idx = ipa_idx
                    if temp_idx < len(ipa) and ipa[temp_idx] == STRESS:
                        temp_idx += 1
                    if temp_idx + len(last_phoneme) <= len(ipa) and ipa[temp_idx:temp_idx + len(last_phoneme)] == last_phoneme:
                        last_char_can_match = True
                        break
                
                # If last character can match, make current character silent
                if last_char_can_match:
                    matched = True
                    phoneme_group.append(config.NONE)
        
        if not matched and ipa_idx < len(ipa):
            # PRIORITY 1: Try to match VOWEL + CONSONANT pattern first
            # This is critical for cases like "ha" where we want to consume both together
            for phoneme in possible_phonemes:
                # Check for pattern: [stress?] + vowel + consonant
                temp_idx = ipa_idx
                
                # Optional leading stress
                has_stress = False
                if temp_idx < len(ipa) and ipa[temp_idx] == STRESS:
                    has_stress = True
                    temp_idx += 1
                
                # Check for vowel + consonant pattern
                if temp_idx < len(ipa) and ipa[temp_idx] in VOWELS:
                    vowel_idx = temp_idx
                    consonant_idx = temp_idx + 1
                    
                    # Check if consonant follows the vowel
                    if consonant_idx + len(phoneme) <= len(ipa) and ipa[consonant_idx:consonant_idx + len(phoneme)] == phoneme:
                        # Match! Consume [stress?] + vowel + consonant
                        if has_stress:
                            phoneme_group.append(STRESS)
                            ipa_idx += 1
                        phoneme_group.append(ipa[vowel_idx])
                        ipa_idx += 1
                        phoneme_group.append(phoneme)
                        ipa_idx += len(phoneme)
                        matched = True
                        break
            
            # PRIORITY 2: Try to match CONSONANT at current position (with optional trailing stress/vowel)
            if not matched:
                for phoneme in possible_phonemes:
                    # Check for optional leading stress
                    temp_idx = ipa_idx
                    has_stress = False
                    if temp_idx < len(ipa) and ipa[temp_idx] == STRESS:
                        has_stress = True
                        temp_idx += 1
                    
                    # Try to match consonant at current position
                    if temp_idx + len(phoneme) <= len(ipa) and ipa[temp_idx:temp_idx + len(phoneme)] == phoneme:
                        # Matched the consonant
                        if has_stress:
                            phoneme_group.append(STRESS)
                            ipa_idx += 1
                        phoneme_group.append(phoneme)
                        ipa_idx += len(phoneme)
                        matched = True
                        
                        # Consume trailing stress markers and ONE vowel (not multiple)
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
            
            # If no consonant matched but there's a standalone vowel/stress, consume it
            # Only do this if we're at the last character or if there's no consonant following
            if not matched and ipa_idx < len(ipa):
                # Check if there's a consonant later in the IPA that could match next character
                has_upcoming_consonant = False
                for i in range(ipa_idx, len(ipa)):
                    if ipa[i] not in VOWELS and ipa[i] != STRESS:
                        has_upcoming_consonant = True
                        break
                
                # Only consume standalone stress/vowel if there's no upcoming consonant
                # or if we're at the last character
                if not has_upcoming_consonant or word_idx >= len(word) - num_chars:
                    if ipa[ipa_idx] == STRESS:
                        phoneme_group.append(STRESS)
                        ipa_idx += 1
                        matched = True
                    if ipa_idx < len(ipa) and ipa[ipa_idx] in VOWELS:
                        phoneme_group.append(ipa[ipa_idx])
                        ipa_idx += 1
                        matched = True
        
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