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


def encode(text: str, ipa: str) -> EncodedSample:
    ...

def decode(text: str, preds: Prediction) -> str:
    """
    preds: Prediction
    returns: IPA string (space-separated phonemes)
    """
    ...