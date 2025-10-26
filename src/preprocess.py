import unicodedata

def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = text.replace('\u05f3', "'") # Geresh -> apostrophe
    text = text.replace('\u05f4', '"') # Gershayim -> double quote
    return text
