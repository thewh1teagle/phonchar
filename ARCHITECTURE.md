# Architecture: Hebrew Phoneme Prediction Model

## Overview

This system fine-tunes DictaBERT (a character-level Hebrew BERT model) to predict IPA phonemes for Hebrew text using a multi-head classification approach. The model processes full sentences to leverage contextual information via BERT's attention mechanism.

## Table of Contents
- [High-Level Architecture](#high-level-architecture)
- [Model Architecture](#model-architecture)
- [Data Flow](#data-flow)
- [Multi-Head Design](#multi-head-design)
- [Character vs Sentence-Level Processing](#character-vs-sentence-level-processing)
- [Handling Mixed Content](#handling-mixed-content)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Design Decisions](#design-decisions)

---

## High-Level Architecture

```
Input Text (Hebrew + mixed content)
         ↓
    Tokenizer (DictaBERT character-level)
         ↓
    BERT Encoder (pretrained on Hebrew)
         ↓
  Custom Phoneme Head (4 classifiers)
         ↓
  Multi-head predictions (per character)
         ↓
   Decode to IPA phonemes
         ↓
Output: Natural sentence with phonemes
```

### Key Components

1. **Base Model**: `dicta-il/dictabert-large-char-menaked`
   - Character-level BERT pretrained on Hebrew
   - 1024 hidden dimensions
   - 512 max sequence length

2. **Custom Head**: Multi-head classifier
   - 4 separate prediction heads per character
   - Shared BERT representations

3. **Tokenization**: Character-to-phoneme alignment
   - Each character gets 4 labels
   - Non-Hebrew characters ignored in training

---

## Model Architecture

### Base: DictaBERT
```python
BertModel(
    vocab_size=~1000,      # Character-level vocabulary
    hidden_size=1024,      # Large hidden dimension
    num_layers=24,         # Deep transformer
    num_heads=16,          # Multi-head attention
    max_position=512       # Character sequence length
)
```

### Custom Phoneme Head
```python
class PhonemeHead(nn.Module):
    def __init__(self, config):
        # 4 independent classifiers
        self.consonant_cls = nn.Linear(1024, 25)     # 24 consonants + Ø
        self.vowel_cls = nn.Linear(1024, 6)          # 5 vowels + Ø
        self.stress_cls = nn.Linear(1024, 2)         # binary: has stress ˈ
        self.flip_vowel_cls = nn.Linear(1024, 2)    # binary: vowel-before-consonant
```

**Why 4 heads instead of 1 combined classifier?**
- Phonemes have compositional structure: consonant + vowel + stress + order
- Separate heads allow the model to learn each aspect independently
- Reduces output space: 25×6×2×2 = 600 vs 600 combined classes
- Better generalization: shared patterns across heads

### Loss Function
```python
loss = CrossEntropyLoss(consonant) + 
       CrossEntropyLoss(vowel) + 
       CrossEntropyLoss(stress) + 
       CrossEntropyLoss(flip_vowel)
```
- Additive loss encourages balanced learning across all heads
- Each head contributes equally to gradient updates
- -100 labels (non-Hebrew chars) are automatically ignored

---

## Data Flow

### Training Data Format
```
Text: ש ל ו ם
IPA:  ʃa lˈo Ø m  (space-separated, one group per character)
```

### Encoding Process
```python
text = 'שלום'
ipa = 'ʃa lˈo Ø m'

# Step 1: Parse each phoneme group
'ʃa'  → consonant=ʃ, vowel=a, stress=0, flip=0
'lˈo' → consonant=l, vowel=o, stress=1, flip=0  # stress before vowel
'Ø'   → consonant=Ø, vowel=Ø, stress=0, flip=0  # silent
'm'   → consonant=m, vowel=Ø, stress=0, flip=0  # consonant-only

# Step 2: Convert to label indices
consonants: [21, 9, 24, 10]  # indices in CONSONANT_CLASSES
vowels:     [0, 3, 5, 5]      # indices in VOWEL_CLASSES
stress:     [0, 1, 0, 0]      # binary flags
flip_vowel: [0, 0, 0, 0]      # binary flags
```

### Decoding Process
```python
# Model outputs logits for each head
logits = {
    'consonant': [batch, seq_len, 25],
    'vowel': [batch, seq_len, 6],
    'stress': [batch, seq_len, 2],
    'flip_vowel': [batch, seq_len, 2]
}

# Get predictions via argmax
predictions = {head: logits[head].argmax(dim=-1) for head in logits}

# Reconstruct phoneme groups
for each character:
    consonant = CONSONANT_CLASSES[pred_consonant[i]]
    vowel = VOWEL_CLASSES[pred_vowel[i]]
    stress_mark = 'ˈ' if pred_stress[i] == 1 else ''
    
    # Build phoneme group respecting order
    if flip_vowel[i] == 1:
        phoneme = stress_mark + vowel + consonant
    else:
        phoneme = consonant + stress_mark + vowel
```

---

## Multi-Head Design

### Why Multi-Head?

**Problem**: Hebrew phonemes have compositional structure
- Same consonant can appear with different vowels: `ba`, `be`, `bi`
- Same vowel can appear with different consonants: `ba`, `ma`, `ta`
- Stress is independent of consonant-vowel choice
- Vowel-consonant order varies (e.g., `av` vs `va`)

**Solution**: Decompose into orthogonal predictions

### Head Definitions

#### 1. Consonant Head (25 classes)
```python
CONSONANTS = ['b', 'v', 'd', 'h', 'z', 'χ', 't', 'j', 'k', 'l', 'm', 
              'n', 's', 'f', 'p', 'ts', 'tʃ', 'w', 'ʔ', 'ɡ', 'ʁ', 'ʃ', 'ʒ', 'dʒ', 'Ø']
```
- Predicts which consonant sound
- `Ø` = silent/no consonant

#### 2. Vowel Head (6 classes)
```python
VOWELS = ['a', 'e', 'i', 'o', 'u', 'Ø']
```
- Predicts which vowel sound
- `Ø` = silent/no vowel

#### 3. Stress Head (2 classes)
```python
STRESS = [0, 1]  # 0 = no stress, 1 = stress marker ˈ
```
- Binary: does this character have stress?
- Stress marker (ˈ) placed before first vowel

#### 4. Flip Vowel Head (2 classes)
```python
FLIP_VOWEL = [0, 1]  # 0 = consonant-first, 1 = vowel-first
```
- Binary: is vowel before consonant?
- Examples: `ba` (flip=0) vs `av` (flip=1)

### Phoneme Reconstruction Rules

```python
def build_phoneme(consonant, vowel, stress, flip_vowel):
    if consonant == 'Ø' and vowel == 'Ø':
        return 'Ø'  # Silent character
    
    if flip_vowel == 1:
        # Vowel before consonant: [ˈ?]vowel consonant
        result = (stress_mark if stress else '') + vowel + consonant
    else:
        # Consonant before vowel: consonant [ˈ?]vowel
        result = consonant + (stress_mark if stress else '') + vowel
    
    return result
```

Examples:
- `consonant=ʃ, vowel=a, stress=0, flip=0` → `ʃa`
- `consonant=l, vowel=o, stress=1, flip=0` → `lˈo`
- `consonant=ʔ, vowel=e, stress=1, flip=0` → `ʔˈe`
- `consonant=v, vowel=a, stress=0, flip=1` → `av`

---

## Character vs Sentence-Level Processing

### Why Process Full Sentences?

**Context Matters**: Same character can have different pronunciations

Example: Hebrew letter `ו`
- In `שלום` (shalom): `ו` → `o` (vowel)
- In `ויקי` (wiki): `ו` → `v` (consonant)

**BERT provides context via attention**:
```
Input:  ש  ל  ו  ם
         ↓  ↓  ↓  ↓
Attention mechanism sees surrounding characters
         ↓  ↓  ↓  ↓
Output: ʃa lˈo Ø m

The `ו` sees:
- Previous: `ל` (lamed)
- Following: `ם` (final mem)
- Context suggests vowel usage
```

### Character-Level Labels, Sentence-Level Context

**Training**:
- Input: Full sentence (up to 512 chars)
- Labels: Per character (4 labels each)
- Loss: Sum over all characters in batch

**Advantages**:
1. **Context-aware predictions**: BERT attention spans entire sentence
2. **Word boundaries implicit**: Model learns from spacing patterns
3. **Batch efficiency**: Process multiple sentences simultaneously
4. **Transfer learning**: Leverages pretrained Hebrew knowledge

---

## Handling Mixed Content

### Non-Hebrew Characters

Hebrew text often contains:
- English words: `hello שלום`
- Numbers: `2024`
- Punctuation: `!`, `.`, `?`

### Strategy: Ignore in Training, Preserve in Inference

**Training Mode** (`preserve_unknown=False`):
```python
text = 'hello שלום'
labels:
  h → consonant=-100, vowel=-100, stress=-100, flip=-100  # ignored in loss
  e → consonant=-100, ...
  ...
  ש → consonant=21, vowel=0, stress=0, flip=0  # actual labels
```
- `-100` is special PyTorch value ignored by `CrossEntropyLoss`
- Model doesn't waste capacity learning English phonemes
- Training focuses purely on Hebrew

**Inference Mode** (`preserve_unknown=True`):
```python
text = 'hello שלום'
output: 'hello ʃalˈom'
```
- Non-Hebrew characters pass through unchanged
- User-friendly output format
- Natural sentence structure preserved

### Implementation
```python
def encode(text, ipa):
    for char in text:
        if char in INPUT_CHARS_SET:  # Hebrew letters + geresh/gershayim
            # Get actual labels
            labels = parse_phoneme_group(ipa[i])
        else:
            # Non-Hebrew: use -100
            labels = [-100, -100, -100, -100]
    return labels

def decode(text, predictions, preserve_unknown=True):
    for char in text:
        if char not in INPUT_CHARS_SET:
            if preserve_unknown:
                output.append(char)  # Keep original char
            else:
                output.append('Ø')    # Training format
```

---

## Training Pipeline

### 1. Data Preparation
```python
from src.dataset import prepare_dataset

# Input: TSV file with aligned data
# Format: word<TAB>ipa
# Example: שלום<TAB>ʃa lˈo Ø m

train_dataset, val_dataset = prepare_dataset('data/train.tsv', split_ratio=0.9)
```

### 2. Model Initialization
```python
from src.model import BertForPhonemeClassification

# Load pretrained DictaBERT
config = AutoConfig.from_pretrained('dicta-il/dictabert-large-char-menaked')
config.num_consonants = 25
config.num_vowels = 6

model = BertForPhonemeClassification.from_pretrained(
    'dicta-il/dictabert-large-char-menaked',
    config=config,
    ignore_mismatched_sizes=True  # We're replacing the head
)

# Option 1: Freeze BERT, train only head (faster, less data needed)
for param in model.bert.parameters():
    param.requires_grad = False

# Option 2: Full fine-tuning (better performance, needs more data)
# (all parameters trainable by default)
```

### 3. Training Loop
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='overall_accuracy'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

### 4. Evaluation Metrics
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Per-head accuracy
    consonant_acc = accuracy(pred_consonant, label_consonant, ignore=-100)
    vowel_acc = accuracy(pred_vowel, label_vowel, ignore=-100)
    stress_acc = accuracy(pred_stress, label_stress, ignore=-100)
    flip_acc = accuracy(pred_flip, label_flip, ignore=-100)
    
    # Overall: all 4 heads correct
    overall_acc = accuracy(all_heads_match, ignore=-100)
    
    # WER: Word Error Rate (phoneme-level)
    # Treats each phoneme group as a "word"
    wer = jiwer.wer(reference_phonemes, predicted_phonemes)
    
    # CER: Character Error Rate (character-level)
    # Measures edit distance at character level
    cer = jiwer.cer(reference_chars, predicted_chars)
    
    return {
        'consonant_accuracy': consonant_acc,
        'vowel_accuracy': vowel_acc,
        'stress_accuracy': stress_acc,
        'flip_vowel_accuracy': flip_acc,
        'overall_accuracy': overall_acc,
        'wer': wer,  # Lower is better (0.0 = perfect)
        'cer': cer   # Lower is better (0.0 = perfect)
    }
```

**Metric Interpretation**:
- **Accuracy metrics** (0-1): Higher is better
  - `overall_accuracy`: All 4 heads must be correct
  - Per-head accuracy: Individual component correctness
- **WER** (Word Error Rate): Phoneme-level error rate
  - `WER = (S + D + I) / N` where S=substitutions, D=deletions, I=insertions, N=reference length
  - Example: `lˈo` → `lo` (missing stress) = 1 error
  - Lower is better (0.0 = perfect match)
- **CER** (Character Error Rate): Character-level edit distance
  - Measures fine-grained phoneme accuracy
  - More sensitive than WER to small differences
  - Lower is better (0.0 = perfect match)

---

## Inference Pipeline

### 1. Load Model
```python
from src.model import BertForPhonemeClassification
from transformers import AutoTokenizer

model = BertForPhonemeClassification.from_pretrained('./trained_model')
tokenizer = AutoTokenizer.from_pretrained('./trained_model')
model.eval()
```

### 2. Predict
```python
sentences = ['שלום עולם', 'hello שלום world']
predictions = model.predict(sentences, tokenizer, preserve_unknown=True)

# Output:
# ['ʃa lˈo Ø m   ʔo lˈa Ø m',     # Per-character IPA
#  'h e l l o   ʃa lˈo Ø m   w o r l d']
```

### 3. Reconstruct Natural Sentences
```python
from src.tokenize import reconstruct_sentence

for text, ipa in zip(sentences, predictions):
    final = reconstruct_sentence(text, ipa)
    print(f"{text} → {final}")

# Output:
# שלום עולם → ʃalˈom ʔolˈam
# hello שלום world → hello ʃalˈom world
```

---

## Design Decisions

### 1. Why Character-Level (not word-level)?

**Advantages**:
- No segmentation required (Hebrew has ambiguous word boundaries)
- Handles unknown words naturally
- Works with mixed scripts seamlessly
- Fine-grained control over phoneme alignment

**Challenges Solved**:
- Context from BERT attention spans beyond single character
- Training on full sentences provides word-level context

### 2. Why Multi-Head (not single classifier)?

**Comparison**:

| Approach | Output Space | Generalization | Interpretability |
|----------|--------------|----------------|------------------|
| Single head | 25×6×2×2=600 classes | Sparse training signal | Opaque |
| Multi-head | 25+6+2+2=35 total | Rich training signal | Clear structure |

**Benefits**:
- Smaller output spaces → less data needed
- Compositional structure → better generalization
- Interpretable predictions → easier debugging

### 3. Why BERT (not simpler RNN/CNN)?

**BERT Advantages**:
- Bidirectional context (sees past and future)
- Pretrained on Hebrew (transfer learning)
- Self-attention captures long-range dependencies
- State-of-the-art on Hebrew NLP tasks

**Comparison**:
```
Character:      ש  ל  ו  ם
                ↓  ↓  ↓  ↓
RNN:           →  →  →  →  (left-to-right only)
BERT:          ↔  ↔  ↔  ↔  (bidirectional attention)
```

### 4. Why Preserve Unknown Characters?

**User Experience**:
```
Input:  "App v2.0: שלום!"
Bad:    "Ø Ø Ø Ø Ø Ø Ø Ø Ø ʃalˈom Ø"
Good:   "App v2.0: ʃalˈom!"
```

**Implementation**:
- Training: Use Ø for consistency
- Inference: Preserve original for readability
- Toggle via `preserve_unknown` flag

### 5. Why Space-Separated IPA?

**Alignment Requirement**:
```
Text: ש ל ו ם      (4 characters)
IPA:  ʃa lˈo Ø m   (4 groups)
```

**Benefits**:
- 1:1 correspondence between text and labels
- Simple parsing and encoding
- Easy visualization and debugging
- Compatible with alignment algorithms

### 6. Why Silent Character (Ø)?

**Hebrew Characteristics**:
- Some letters are sometimes silent (matres lectionis)
- Final letters often don't produce sound
- Geresh/gershayim markers are non-phonetic

**Solution**:
```python
CONSONANT_CLASSES = [..., 'Ø']  # Silent/no consonant
VOWEL_CLASSES = [..., 'Ø']       # Silent/no vowel

# Both Ø → character is completely silent
if consonant == 'Ø' and vowel == 'Ø':
    output = 'Ø'  # Silent character
```

---

## Performance Considerations

### Training Efficiency

**Batch Processing**:
- Characters per batch: `batch_size × max_seq_len`
- Typical: 16 × 512 = 8,192 characters per step
- Only Hebrew characters contribute to loss

**Memory Usage**:
- BERT encoder: ~3GB (1024 hidden, 24 layers)
- Phoneme head: negligible (~100KB)
- Gradients: 2× model size if training full model

**Training Time**:
- Freeze BERT: ~1 hour per epoch (10K examples)
- Full fine-tuning: ~4 hours per epoch
- Convergence: typically 3-5 epochs

### Inference Speed

**Per-sentence**:
- Tokenization: ~1ms
- BERT forward: ~10-50ms (depends on length)
- Phoneme head: ~1ms
- Decoding: ~1ms
- Total: ~15-55ms per sentence

**Batch processing**:
- Linear scaling up to GPU memory limit
- Typical throughput: 50-200 sentences/second

---

## Testing

### Unit Tests (28 tests)
```bash
pytest tests/test_tokenize.py -v
```
- Character-level encode/decode
- Stress positioning
- Silent characters
- Sentence processing
- Mixed content handling
- Sentence reconstruction

### Integration Demo (7 tests)
```bash
pytest tests/test_full_pipeline_demo.py -v -s
```
- Full pipeline walkthrough
- Mixed content demo
- Context-aware learning
- Training vs inference modes
- Architecture summary
- Data format examples

---

## Future Improvements

### Potential Enhancements

1. **Conditional Generation**: Add language model head for sequence prediction
2. **Multi-task Learning**: Joint training on diacritization + phonemes
3. **Confidence Scores**: Output probabilities instead of argmax
4. **Beam Search**: Multiple phoneme hypotheses per character
5. **Attention Visualization**: Interpret what context model uses

### Known Limitations

1. **Out-of-vocabulary**: Special characters may not have phoneme mappings
2. **Dialect Variations**: Model learns from training data dialect
3. **Ambiguity**: Some words have multiple valid pronunciations
4. **Context Window**: Limited to 512 characters per sequence

---

## References

- **DictaBERT**: [dicta-il/dictabert-large-char-menaked](https://huggingface.co/dicta-il/dictabert-large-char-menaked)
- **IPA for Hebrew**: [Wikipedia](https://en.wikipedia.org/wiki/Help:IPA/Hebrew)
- **Dataset**: [thewh1teagle/phonikud-phonemes-data](https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data)

