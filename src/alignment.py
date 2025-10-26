"""
Align word and IPA to pairs of character and phoneme
"""
from src import config
from src.preprocess import normalize

def align_word(word: str, ipa: str) -> list[list[str], list[str]]:
    word = normalize(word)
    
    word_parts = list(word)
    ipa_parts = []
    
    index = 0
    while True:
        cur_char = word[index]
        next_char = word[index + 1] if index + 1 < len(word) else None
        cur_phoneme = ipa[index]
        next_phoneme = ipa[index + 1] if index + 1 < len(ipa) else None

        if next_char == config.GERESH or next_char == config.GERSHAYIM:
            cur_phoneme = config.CHAR_TO_PHONEME[f"{cur_char}{next_char}"]
            index += 2
        else:
            cur_phoneme = config.CHAR_TO_PHONEME[cur_char]
            index += 1

    return [word_parts, ipa_parts]