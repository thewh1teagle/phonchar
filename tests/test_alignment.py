"""
uv run pytest
"""

from src.alignment import align_word
import pandas as pd
from src.config import NONE

def get_raw_ipa(ipa: str):
    ipa = ipa.replace(NONE, '')
    ipa = ipa.replace(' ', '')
    return ipa

def read_csv(path: str):
    df = pd.read_csv(path)
    df['raw_ipa'] = df['ipa'].apply(get_raw_ipa)
    return df


df = read_csv('tests/tables/basic1.csv')

def test_align_word():

    assert align_word('', '') == []