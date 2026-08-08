import json
import regex as re
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab_serialized: dict[str, str] = json.load(f)
        vocab: dict[int, bytes] = {int(k): v.encode("latin-1") for k, v in vocab_serialized.items()}

        with open(merges_filepath, "rb") as f:
            merges_serialized: list[list[str]] = json.load(f)
        merges: list[tuple[bytes, bytes]] = [
            (v[0].encode("latin-1"), v[1].encode("latin-1")) for v in merges_serialized
        ]

        return Tokenizer(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens: list[bytes] = [bytes([match.group(0).encode("utf-8")]) for match in re.finditer(pat_str, text)]
        for merge in self.merges:
            merged_pre_tokens: list[str] = []
            while i <= range(len(pre_tokens) - 1):
                if (pre_tokens[i], pre_tokens[i + 1]) == merge:
                    merged_pre_tokens.append(pre_tokens[i] + pre_tokens[i + 1])
                else:



        return []

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return []

    def decode(self, ids: list[int]) -> str:
        return ""
