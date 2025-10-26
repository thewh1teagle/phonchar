
"""
See https://en.wikipedia.org/wiki/Unicode_and_HTML_for_the_Hebrew_alphabet#Compact_table
"""

import unicodedata

def normalize(text: str) -> str:
    """
    Normalize Hebrew text to Unicode NFD form
    Normalize Hebrew Geresh and Gershayim to apostrophe and double quote
    """
    text = unicodedata.normalize('NFD', text)
    text = text.replace('\u05f3', "'") # Geresh -> apostrophe
    text = text.replace('\u05f4', '"') # Gershayim -> double quote
    return text
