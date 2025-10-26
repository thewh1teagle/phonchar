
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