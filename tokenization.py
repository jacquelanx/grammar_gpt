from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class SimpleTokenizer:
    """
    Minimal tokenizer for controlled synthetic grammars.
    - mode="word": tokenize on whitespace
    - mode="char": characters (including spaces)
    """
    vocab: List[str]
    mode: str = "word"
    unk_token: str = "<unk>"
    eos_token: str = "<eos>"

    def __post_init__(self) -> None:
        assert self.mode in ("word", "char")
        if self.unk_token not in self.vocab:
            self.vocab.append(self.unk_token)
        if self.eos_token not in self.vocab:
            self.vocab.append(self.eos_token)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: t for t, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        toks: Sequence[str]
        if self.mode == "char":
            toks = list(text) # string to list of characters
        else:
            toks = text.strip().split() # ["the", "cat", "near"...]
        ids = [self.stoi.get(t, self.stoi[self.unk_token]) for t in toks]
        if add_eos:
            ids.append(self.stoi[self.eos_token])
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.itos.get(i, self.unk_token) for i in ids]
        return "".join(toks) if self.mode == "char" else " ".join(toks)