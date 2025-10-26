
"""
See https://github.com/thewh1teagle/phonikud
See https://en.wikipedia.org/wiki/Help:IPA/Hebrew
"""


# Input
INPUT_CHARS = [
    *'אבגדהוזחטיכךלמםנןסעפףצץקרשת',
    r"\"'",
]

# Output
STRESS = 'ˈ'
NONE = 'Ø'
VOWELS = list('aeiou')
CONSONANTS = [
    'b', 
    'v', 
    'd', 
    'h', 
    'z', 
    'χ', 
    't', 
    'j', 
    'k', 
    'l', 
    'm', 
    'n', 
    's', 
    'f', 
    'p', 
    'ts', 
    'tʃ', 
    'w',
    'ʔ',
    'ɡ',
    'ʁ',
    'ʃ',
    'ʒ',
    'dʒ',
]

GERESH = "'"
GERSHAYIM = '"'

CHAR_TO_PHONEME = {
    'א': ['ʔ'],
    'ב': ['b', 'v'],
    'ג': ['ɡ'],
    'ד': ['d'],
    'ה': ['h'],
    'ו': ['v', 'w'],
    'ז': ['z'],
    'ח': ['χ'],
    'ט': ['t'],
    'י': ['j'],
    'כ': ['k', 'χ'],
    'ך': ['k', 'χ'],
    'ל': ['l'],
    'מ': ['m'],
    'ם': ['m'],
    'נ': ['n'],
    'ן': ['n'],
    'ס': ['s'],
    'ע': ['ʔ'],
    'פ': ['p', 'f'],
    'ף': ['p', 'f'],
    "צ'": ['ts', 'tʃ'],
    'ץ': ['ts', 'tʃ'],
    'ק': ['k'],
    'ר': ['r'],
    'ש': ['ʃ', 's'],
    'ת': ['t'],

    # Special
    f"ג{GERESH}": ["dʒ"],
    f"ז{GERESH}": ["ʒ"],
    f"צ{GERESH}": ['tʃ'],
    f"ץ{GERESH}": ['tʃ'],
    f"ת{GERESH}": ['t'],
}