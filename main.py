import regex as re
from typing import List, Dict, Optional, Tuple, Counter, Any, Iterator
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    vocab_size: int = 50000
    min_freq: int = 2
    batch_size: int = 1000


class BPETokenizer:
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.special_tokens = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3}
        self.vocab = self.special_tokens.copy()
        self.merges = {}

    def train_from_file(self, file_path: str, encoding: str = 'utf-8') -> None:
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        self.train([text])

    def train(self, texts: List[str]) -> None:
        print("Phase 1/2: Building initial vocabulary...")
        word_freqs = self._get_word_freqs(texts)

        print("Phase 2/2: Learning merge rules...")
        self._learn_merges(word_freqs)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        word_freqs = defaultdict(int)
        vocab_size = self.config.vocab_size - len(self.special_tokens)

        # Add individual characters to vocab first
        unique_chars = set(''.join(texts))
        for i, char in enumerate(unique_chars):
            if i + len(self.special_tokens) >= self.config.vocab_size:
                break
            self.vocab[char] = len(self.special_tokens) + i

        # Count word frequencies
        for text in texts:
            words = re.findall(r'\w+|\s+|[^\w\s]', text)
            for word in words:
                chars = ' '.join(list(word))
                word_freqs[chars] += 1
        return word_freqs

    def _learn_merges(self, word_freqs: Dict[str, int]) -> None:
        # Initialize vocabulary with characters first
        all_chars = set()
        for word in word_freqs:
            all_chars.update(word.split())

        # Add characters to vocab
        for char in all_chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        vocab_size = self.config.vocab_size - len(self.vocab)
        with tqdm(total=vocab_size) as pbar:
            while len(self.vocab) < self.config.vocab_size:
                pairs = self._get_pair_stats(word_freqs)
                if not pairs:
                    break

                best_pair = max(pairs.items(), key=lambda x: x[1])[0]
                if pairs[best_pair] < self.config.min_freq:
                    break

                # Add merged token to vocabulary
                merged_token = ''.join(best_pair)
                self.vocab[merged_token] = len(self.vocab)
                self.merges[best_pair] = self.vocab[merged_token]

                # Update word frequencies with merged token
                word_freqs = self._merge_pair(best_pair, word_freqs)
                pbar.update(1)

    def _get_pair_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        new_freqs = defaultdict(int)
        bigram = re.escape(' '.join(pair))
        replacement = ''.join(pair)
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word, freq in word_freqs.items():
            new_word = pattern.sub(replacement, word)
            new_freqs[new_word] += freq

        return new_freqs

    def encode(self, text: str) -> List[int]:
        encoded, _ = self.encode_with_debug(text)
        return encoded

    def encode_with_debug(self, text: str) -> Tuple[List[int], str]:
        tokens = re.findall(r'\S+|\s+', text)
        encoded = []
        debug_parts = []

        for token in tokens:
            # Try encoding with longest possible tokens
            remaining = token
            while remaining:
                longest_match = remaining
                longest_id = None

                # Check all prefixes in vocab
                for i in range(len(remaining), 0, -1):
                    prefix = remaining[:i]
                    if prefix in self.vocab:
                        longest_match = prefix
                        longest_id = self.vocab[prefix]
                        break

                if longest_id is not None:
                    encoded.append(longest_id)
                    debug_parts.append(longest_match)
                    remaining = remaining[len(longest_match):]
                else:
                    # Handle single character
                    char = remaining[0]
                    encoded.append(self.vocab.get(char, self.special_tokens['<unk>']))
                    debug_parts.append(char)
                    remaining = remaining[1:]

        return encoded, " | ".join(debug_parts)

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
        return ''.join(tokens)

    def save_vocab(self, path: str) -> None:
        data = {
            'vocab': {(k[0], k[1]) if isinstance(k, tuple) else k: v for k, v in self.vocab.items()},
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = {}
        for k, v in data['vocab'].items():
            if isinstance(k, list):
                self.vocab[tuple(k)] = v
            else:
                self.vocab[k] = v

        self.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


if __name__ == "__main__":
    config = TokenizerConfig(vocab_size=1000, min_freq=2)
    tokenizer = BPETokenizer(config)

    try:
        if Path("vocab.json").exists():
            print("Loading vocabulary...")
            tokenizer.load_vocab("vocab.json")
        else:
            print("Training new tokenizer...")
            tokenizer.train_from_file("onegin.txt")
            tokenizer.save_vocab("vocab.json")

        text = "Привет! Это пример работы токенизации. " \
               "Приставка, душевный разговор - Это примеры популярных слов!"

        encoded, debug = tokenizer.encode_with_debug(text)
        print(f"Text: {text}")
        print(f"Tokens: {debug}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {tokenizer.decode(encoded)}")
    except Exception as e:
        print(f"Error: {e}")