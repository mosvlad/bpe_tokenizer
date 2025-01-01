# Byte-Pair Encoding (BPE) Tokenizer

A Python implementation of a Byte-Pair Encoding tokenizer optimized for Russian text, supporting UTF-8 encoding, vocabulary persistence, and subword tokenization.

## Features

- Byte-pair encoding with configurable vocabulary size
- UTF-8 support for multilingual text
- Subword tokenization with merge rules
- Progress bars for training phases
- Vocabulary persistence (save/load)
- Debug mode for token visualization
- Space-aware tokenization

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- tqdm
- regex
- numpy

## Usage

```python
from bpe_tokenizer import BPETokenizer, TokenizerConfig

# Initialize tokenizer
config = TokenizerConfig(vocab_size=1000, min_freq=2)
tokenizer = BPETokenizer(config)

# Train from file
tokenizer.train_from_file("data.txt")

# Save vocabulary
tokenizer.save_vocab("vocab.json")

# Load existing vocabulary
tokenizer.load_vocab("vocab.json")

# Tokenize text
text = "Это пример токенизации!"
encoded, debug = tokenizer.encode_with_debug(text)
print(f"Text: {text}")
print(f"Tokens: {debug}")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
```

## Configuration

`TokenizerConfig` parameters:
- `vocab_size`: Maximum vocabulary size (default: 50000)
- `min_freq`: Minimum frequency for merge operations (default: 2)
- `batch_size`: Batch size for training (default: 1000)

## Features

### Training
- Character-level initialization
- Merge pair learning with frequency tracking
- Progress visualization with tqdm
- Batched processing for large datasets

### Encoding/Decoding
- Efficient longest-token-first encoding
- Space-aware tokenization
- UTF-8 support
- Debug mode for token visualization

### Vocabulary Management
- Save/load vocabulary to JSON
- Special tokens support (`<unk>`, `<pad>`, `<s>`, `</s>`)
- Merge rules persistence

## Example Output

```python
Text: разговор по душам
Tokens: разговор | | по | | душа | м
Encoded: [942, 63, 132, 63, 953, 27]
Decoded: разговор по душам
```

## License

MIT License