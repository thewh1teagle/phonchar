"""
Align word and IPA to pairs of character and phoneme
"""
from src.config import VOWELS, CONSONANTS, NONE, STRESS
from src.preprocess import normalize

def align_word(word: str, ipa: str) -> list[tuple[str, str]]:
    word = normalize(word)
    
    pairs = []
    
    word_index = 0
    ipa_index = 0
    


    return pairs