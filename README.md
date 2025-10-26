# phonchar

Fast per character phoneme classification

inputs: `אבגדהוזחטיכךלמםנןסעפףצץקרשת`

outputs:
- consonants: `b v d h z χ t j k l m n s f p ts tʃ w ʔ ɡ ʁ ʃ ʒ dʒ NONE`
- vowels: `a e i o u NONE`
- stress: `1/0`
- silent: `1/0`
- flip_vowel: `1/0`


heads: 
- consonants (class)
- vowels (class)
- stress (binary)
- silent (binary)
- flip_vowel (binary)


ש ʃ a 0 0 0
ל l o 1 0 0


Datasets: 

- https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data
