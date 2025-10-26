
"""
See https://en.wikipedia.org/wiki/Unicode_and_HTML_for_the_Hebrew_alphabet#Compact_table
"""

import unicodedata
import re


def normalize_ipa(ipa: str) -> str:
    ipa = ipa.replace('g', 'ɡ')
    ipa = ipa.replace('r', 'ʁ')
    ipa = ipa.replace('x', 'χ')
    return ipa

def normalize_hebrew(text: str) -> str:
    """
    Normalize Hebrew text to Unicode NFD form
    Normalize Hebrew Geresh and Gershayim to apostrophe and double quote
    Remove diacritics
    """
    text = unicodedata.normalize('NFD', text)
    text = text.replace('\u05f3', "'") # Geresh -> apostrophe
    text = text.replace('\u05f4', '"') # Gershayim -> double quote
    text = re.sub(r'[\u0590-\u05c7]', '', text) # Remove diacritics
    return text
