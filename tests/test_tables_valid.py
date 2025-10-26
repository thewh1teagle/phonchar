"""
uv run pytest
"""

from src.alignment import align_word
import pandas as pd
from pathlib import Path


TABLES_DIR = Path('tests/tables')
tables = TABLES_DIR.glob('*.csv')


def read_csv(path: str):
    df = pd.read_csv(path)
    return df


def test_tables_valid():
    for table in tables:
        df = read_csv(table)
        for _, row in df.iterrows():
            word = row['word']
            ipa = row['ipa']
            ipa_parts = ipa.split(' ')
            assert len(word) == len(ipa_parts), (
                f"Mismatch in {table} at row {row.name + 2}:\n"
                f"word='{word}' ({len(word)} chars)\n"
                f"ipa='{ipa}' ({len(ipa_parts)} parts)"
            )
