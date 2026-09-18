import json
from collections.abc import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.bytes_to_token_id: dict[bytes, int] = dict(zip(self.vocab.values(), self.vocab.keys()))
        self.merges = merges
        self.special_tokens = None if not special_tokens else sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
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

        ids: list[int] = []
        chunks: list[str] = [text]
        if self.special_tokens:
            escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
            chunks = re.split("(" + "|".join(escaped_special_tokens) + ")", text)

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                ids.append(self.bytes_to_token_id[chunk.encode("utf-8")])
                continue

            pre_tokens: list[list[bytes]] = [
                [bytes([b]) for b in match.group(0).encode("utf-8")] for match in re.finditer(pat_str, chunk)
            ]
            for merge in self.merges:
                for i in range(len(pre_tokens)):
                    pre_token = pre_tokens[i]
                    new_pre_token = []
                    k = 0
                    while k < len(pre_token):
                        if k + 1 < len(pre_token) and (pre_token[k], pre_token[k + 1]) == merge:
                            new_pre_token.append(pre_token[k] + pre_token[k + 1])
                            k += 2
                        else:
                            new_pre_token.append(pre_token[k])
                            k += 1
                    pre_tokens[i] = new_pre_token

            for pre_token in pre_tokens:
                for b in pre_token:
                    ids.append(self.bytes_to_token_id[b])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            token_ids = self.encode(s)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        tokens: list[bytes] = []
        for id in ids:
            token = self.vocab[id]
            tokens.append(token)

        return b"".join(tokens).decode(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/kliuz/home/cs336-assignment1-basics/outputs/TinyStoriesV2-GPT4-train_vocab.json",
        merges_filepath="/home/kliuz/home/cs336-assignment1-basics/outputs/TinyStoriesV2-GPT4-train_merges.json",
        special_tokens=["<|endoftext|>"],
    )
    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    print("input text:", text)
    ids: list[int] = tokenizer.encode(text)
    print("token ids:", ids)
    new_text = tokenizer.decode(ids)
    print("recovered text:", new_text)
    print("equal?", text == new_text)
