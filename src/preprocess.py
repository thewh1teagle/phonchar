
"""
See https://en.wikipedia.org/wiki/Unicode_and_HTML_for_the_Hebrew_alphabet#Compact_table
"""

import unicodedata

def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = text.replace('\u05f3', "'") # Geresh -> apostrophe
    text = text.replace('\u05f4', '"') # Gershayim -> double quote
    return text
