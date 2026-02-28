"""
Character <-> Index mapping for CTC-based text recognition.
Index 0 is reserved for CTC blank token.
"""
import os
import re


class LabelConverter:
    """Converts between text labels and integer sequences for CTC."""

    def __init__(self, vocab_chars=None):
        """
        Args:
            vocab_chars: string of all unique characters in dataset.
                         If None, must call build_vocab() first.
        """
        self.blank_token = '[blank]'
        if vocab_chars is not None:
            self._build(vocab_chars)

    def build_vocab_from_dir(self, annotation_dir):
        """Scan all TXT annotation files and build vocabulary."""
        chars = set()
        for fname in sorted(os.listdir(annotation_dir)):
            if not fname.endswith('.txt'):
                continue
            if re.search(r'\(\d+\)\.txt$', fname):
                continue  # skip duplicates
            fpath = os.path.join(annotation_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',', 8)
                    if len(parts) >= 9:
                        text = parts[8]
                        for ch in text:
                            chars.add(ch)
        vocab = ''.join(sorted(chars))
        self._build(vocab)
        return vocab

    def _build(self, vocab_chars):
        """Build index mappings from vocabulary string."""
        self.vocab_chars = vocab_chars
        # Index 0 = blank (CTC), then 1..N for each character
        self.char_to_idx = {}
        self.idx_to_char = {0: self.blank_token}
        for i, ch in enumerate(vocab_chars, 1):
            self.char_to_idx[ch] = i
            self.idx_to_char[i] = ch
        self.num_classes = len(vocab_chars) + 1  # +1 for blank

    def encode(self, text):
        """Convert text string to list of indices."""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, indices, raw=False):
        """
        Convert index sequence to text string.
        Args:
            indices: list of int indices
            raw: if False, apply CTC decoding (merge repeats, remove blanks)
        """
        if raw:
            return ''.join(self.idx_to_char.get(i, '?') for i in indices)

        # CTC greedy decode: merge consecutive duplicates, then remove blanks
        chars = []
        prev = -1
        for idx in indices:
            if idx != prev:
                if idx != 0:  # not blank
                    chars.append(self.idx_to_char.get(idx, '?'))
            prev = idx
        return ''.join(chars)

    def decode_batch(self, batch_indices):
        """Decode a batch of index sequences."""
        return [self.decode(seq) for seq in batch_indices]
