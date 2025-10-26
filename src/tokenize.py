"""
Tokenize pair of word and IPA

There are class and binary flags for each character in the word and IPA.

Class flags:
- consonant
- vowel

Binary flags:
- stress
- flip_vowel

None on both consonant and vowel means silent.
"""
from typing import List, NamedTuple
from src import config


class EncodedSample(NamedTuple):
    char_ids: List[int]
    consonant: List[int]
    vowel: List[int]
    stress: List[int]
    flip_vowel: List[int]

class Prediction(NamedTuple):
    consonant: List[int]
    vowel: List[int]
    stress: List[int]
    flip_vowel: List[int]


# Create class mappings
CONSONANT_CLASSES = config.CONSONANTS + [config.NONE]
VOWEL_CLASSES = config.VOWELS + [config.NONE]
CONSONANT_TO_IDX = {c: i for i, c in enumerate(CONSONANT_CLASSES)}
VOWEL_TO_IDX = {v: i for i, v in enumerate(VOWEL_CLASSES)}
INPUT_CHARS_SET = set(config.INPUT_CHARS)


def parse_phoneme_group(phoneme_group: str) -> tuple[int, int, int, int]:
    """
    Parse a phoneme group (e.g., 'dʒi', 'Ø', 'rˈa', 'ˈvχ') into components.
    
    Returns:
        (consonant_idx, vowel_idx, stress, flip_vowel)
    """
    if phoneme_group == config.NONE:
        # Silent character
        return (
            CONSONANT_TO_IDX[config.NONE],
            VOWEL_TO_IDX[config.NONE],
            0,
            0
        )
    
    # Parse the phoneme group
    has_stress = config.STRESS in phoneme_group
    stress = 1 if has_stress else 0
    
    # Find stress position before removing it
    stress_pos = phoneme_group.find(config.STRESS) if has_stress else -1
    
    # Remove stress marker for parsing
    phoneme_clean = phoneme_group.replace(config.STRESS, '')
    
    # Find consonant and vowel
    consonant = None
    vowel = None
    consonant_pos = -1
    vowel_pos = -1
    
    # Try to find consonants (longest match first)
    for cons in sorted(config.CONSONANTS, key=len, reverse=True):
        if cons in phoneme_clean:
            pos = phoneme_clean.find(cons)
            if consonant is None or pos < consonant_pos:
                consonant = cons
                consonant_pos = pos
                break
    
    # Find vowels
    for v in config.VOWELS:
        if v in phoneme_clean:
            pos = phoneme_clean.find(v)
            vowel = v
            vowel_pos = pos
            break
    
    # Determine flip_vowel: 1 if vowel comes before consonant IN THE ORIGINAL
    # Need to account for stress position
    flip_vowel = 0
    if consonant and vowel:
        # Adjust positions based on original string with stress
        original_consonant_pos = consonant_pos
        original_vowel_pos = vowel_pos
        
        if stress_pos >= 0:
            # Adjust positions if they come after stress marker
            if consonant_pos >= stress_pos:
                original_consonant_pos += 1
            if vowel_pos >= stress_pos:
                original_vowel_pos += 1
        
        flip_vowel = 1 if original_vowel_pos < original_consonant_pos else 0
    
    # Get indices
    consonant_idx = CONSONANT_TO_IDX.get(consonant, CONSONANT_TO_IDX[config.NONE])
    vowel_idx = VOWEL_TO_IDX.get(vowel, VOWEL_TO_IDX[config.NONE])
    
    return (consonant_idx, vowel_idx, stress, flip_vowel)


def encode(text: str, ipa: str) -> EncodedSample:
    """
    Encode text and aligned IPA into training labels.
    
    Args:
        text: Hebrew text (normalized)
        ipa: Space-separated phoneme groups, one per character
    
    Returns:
        EncodedSample with labels for all characters
    """
    ipa_parts = ipa.split(' ')
    
    if len(text) != len(ipa_parts):
        raise ValueError(f"Text length ({len(text)}) must match IPA parts ({len(ipa_parts)})")
    
    char_ids = []
    consonants = []
    vowels = []
    stresses = []
    flip_vowels = []
    
    for char, phoneme_group in zip(text, ipa_parts):
        # Get character ID
        if char in INPUT_CHARS_SET:
            char_id = config.INPUT_CHARS.index(char)
        else:
            char_id = -1  # Not in vocabulary
        
        char_ids.append(char_id)
        
        # Parse phonemes only for INPUT_CHARS
        if char in INPUT_CHARS_SET:
            cons_idx, vowel_idx, stress, flip_vowel = parse_phoneme_group(phoneme_group)
            consonants.append(cons_idx)
            vowels.append(vowel_idx)
            stresses.append(stress)
            flip_vowels.append(flip_vowel)
        else:
            # Ignore non-INPUT_CHARS in loss
            consonants.append(-100)
            vowels.append(-100)
            stresses.append(-100)
            flip_vowels.append(-100)
    
    return EncodedSample(
        char_ids=char_ids,
        consonant=consonants,
        vowel=vowels,
        stress=stresses,
        flip_vowel=flip_vowels
    )


def decode(text: str, preds: Prediction, preserve_unknown: bool = True) -> str:
    """
    Decode predictions back to IPA string.
    
    Args:
        text: Hebrew text
        preds: Prediction with class indices for each character
        preserve_unknown: If True, preserve original character for non-Hebrew chars.
                         If False, output Ø for non-Hebrew chars.
    
    Returns:
        IPA string (space-separated phonemes)
    """
    if len(text) != len(preds.consonant):
        raise ValueError(f"Text length ({len(text)}) must match predictions ({len(preds.consonant)})")
    
    ipa_parts = []
    
    for i, char in enumerate(text):
        if char not in INPUT_CHARS_SET:
            # For non-INPUT_CHARS, either preserve the character or output Ø
            if preserve_unknown:
                ipa_parts.append(char)
            else:
                ipa_parts.append(config.NONE)
            continue
        
        cons_idx = preds.consonant[i]
        vowel_idx = preds.vowel[i]
        stress = preds.stress[i]
        flip_vowel = preds.flip_vowel[i]
        
        # Get phoneme components
        consonant = CONSONANT_CLASSES[cons_idx] if 0 <= cons_idx < len(CONSONANT_CLASSES) else config.NONE
        vowel = VOWEL_CLASSES[vowel_idx] if 0 <= vowel_idx < len(VOWEL_CLASSES) else config.NONE
        
        # Build phoneme group
        phoneme_group = []
        
        if consonant == config.NONE and vowel == config.NONE:
            # Silent character
            phoneme_group.append(config.NONE)
        else:
            # Build phoneme group with stress before first vowel
            if flip_vowel == 1:
                # Vowel before consonant: [stress?] vowel consonant
                if vowel != config.NONE:
                    if stress == 1:
                        phoneme_group.append(config.STRESS)
                    phoneme_group.append(vowel)
                if consonant != config.NONE:
                    phoneme_group.append(consonant)
            else:
                # Consonant before vowel (or just consonant/vowel)
                if consonant != config.NONE:
                    phoneme_group.append(consonant)
                if vowel != config.NONE:
                    if stress == 1:
                        phoneme_group.append(config.STRESS)
                    phoneme_group.append(vowel)
                elif stress == 1:
                    # No vowel but has stress - put stress at beginning
                    phoneme_group.insert(0, config.STRESS)
        
        ipa_parts.append(''.join(phoneme_group))
    
    return ' '.join(ipa_parts)


def reconstruct_sentence(text: str, ipa: str) -> str:
    """
    Reconstruct a natural sentence from per-character IPA.
    
    Joins phonemes for Hebrew characters while preserving spaces and other separators.
    Silent characters (Ø) are omitted from the output.
    
    Args:
        text: Original text
        ipa: Space-separated IPA (one phoneme group per character)
    
    Returns:
        Sentence with phonemes joined into words
        
    Example:
        text='hello שלום'
        ipa='h e l l o   ʃa lˈo Ø m'  # note: double space for the space char
        returns='hello ʃalˈom'
    """
    if not ipa:
        return ''
    
    ipa_parts = ipa.split(' ')
    
    # Map text positions to IPA parts
    # Two consecutive empty strings represent one space character
    text_to_ipa = []
    ipa_idx = 0
    while len(text_to_ipa) < len(text) and ipa_idx < len(ipa_parts):
        if ipa_parts[ipa_idx] == '' and ipa_idx + 1 < len(ipa_parts) and ipa_parts[ipa_idx + 1] == '':
            # Two empty strings = one space character
            text_to_ipa.append(' ')  # Use space as marker
            ipa_idx += 2  # Skip both empty strings
        else:
            text_to_ipa.append(ipa_parts[ipa_idx])
            ipa_idx += 1
    
    if len(text_to_ipa) != len(text):
        raise ValueError(f"Text length ({len(text)}) must match IPA parts ({len(text_to_ipa)}). Got IPA: {repr(ipa)}")
    
    # Build result
    result = []
    current_word = []
    
    for char, phoneme in zip(text, text_to_ipa):
        if char in INPUT_CHARS_SET:
            # Hebrew character - add phoneme to current word (skip Ø)
            if phoneme != config.NONE:
                current_word.append(phoneme)
        else:
            # Non-Hebrew character (space, punctuation, etc.)
            # Flush current word if any
            if current_word:
                result.append(''.join(current_word))
                current_word = []
            
            # Add the separator
            if char == ' ':
                # Preserve space
                result.append(' ')
            elif phoneme != ' ' and phoneme != config.NONE:
                # Non-space character (punctuation, etc.)
                result.append(phoneme)
    
    # Flush remaining word
    if current_word:
        result.append(''.join(current_word))
    
    return ''.join(result)