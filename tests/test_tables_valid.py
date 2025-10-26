"""
uv run pytest
"""

from src.alignment import align_word
import pandas as pd
from pathlib import Path


TABLES_DIR = Path('tests/tables')
tables = TABLES_DIR.glob('*.csv')
tables = ['tests/tables/basic2.csv']


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


# def test_tables_valid1():
#     for table in tables:
#         df = read_csv(table)

#         total = len(df)
#         bad = 0
#         errors = []

#         for i, row in df.iterrows():
#             word = row['word']
#             ipa = row['ipa']
#             ipa_parts = ipa.split(' ')
#             if len(word) != len(ipa_parts):
#                 bad += 1
#                 errors.append(
#                     f"Row {i + 2}: word='{word}' ({len(word)}), "
#                     f"ipa='{ipa}' ({len(ipa_parts)})"
#                 )

#         good = total - bad
#         ratio = (good / total) * 100 if total > 0 else 0

#         # ✅ Print analysis for debug
#         print(f"\n📊 File: {table}")
#         print(f"✅ Good: {good}/{total} ({ratio:.2f}%)")
#         print(f"❌ Bad:  {bad}/{total}")

#         # ✅ If errors exist, show them and FAIL test
#         if errors:
#             msg = "\n".join(errors[:10])  # limit to first 10 to avoid spam
#             raise AssertionError(f"{bad} mismatches found in {table}:\n{msg}")