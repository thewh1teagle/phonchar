"""
uv run pytest
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.alignment import align_word
import pandas as pd
from src.config import NONE
from pathlib import Path
from src.preprocess import normalize

TABLES_DIR = Path('tests/tables')
tables = TABLES_DIR.glob('*.csv')
tables = [TABLES_DIR / 'basic1.csv']

def clean_ipa(ipa: str):
    """
    ʃa lˈo Ø m -> ʃalˈom
    """
    ipa = ipa.replace(NONE, '')
    ipa = ipa.replace(' ', '')
    return ipa

def clean_word(word: str):
    word = normalize(word)
    return word

def read_csv(path: str):
    df = pd.read_csv(path)
    df['clean_ipa'] = df['ipa'].apply(clean_ipa)
    df['word'] = df['word'].apply(clean_word)
    return df

def test_align_word():
    for table in tables:
        df = read_csv(table)
        for _, row in df.iterrows():
            word = row['word']
            clean_ipa = row['clean_ipa']
            expected_ipa = row['ipa']

            aligned_word, aligned_ipa = align_word(word, clean_ipa)
            
            assert aligned_word == word, (
                f"Mismatch in {table} at row {row.name + 2}:\n"
                f"word='{word}' ({len(word)} chars)\n"
                f"aligned_word='{aligned_word}' ({len(aligned_word)} chars)"
            )
            assert aligned_ipa == expected_ipa, (
                f"Mismatch in {table} at row {row.name + 2}:\n"
                f"expected_ipa='{expected_ipa}' ({len(expected_ipa)} chars)\n"
                f"aligned_ipa='{aligned_ipa}' ({len(aligned_ipa)} chars)"
            )
