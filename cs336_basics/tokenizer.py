import json
from collections.abc import Iterable, Iterator

import random
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
        self.merge_order: dict[tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(merges)}
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

    def encode_pre_token(self, pre_token: list[bytes]) -> list[int]:
        if len(pre_token) == 1:
            return [self.bytes_to_token_id[pre_token[0]]]

        merged_pre_token: list[bytes] = pre_token
        while True:
            earliest_merge: tuple[int, tuple[bytes, bytes]] | None = None
            for i in range(len(merged_pre_token) - 1):
                pair = (merged_pre_token[i], merged_pre_token[i + 1])
                if pair in self.merge_order and (not earliest_merge or self.merge_order[pair] < earliest_merge[0]):
                    earliest_merge = (self.merge_order[pair], pair)

            if not earliest_merge:
                break

            i = 0
            new_merged_pre_token: list[bytes] = []
            while i < len(merged_pre_token):
                if (
                    i + 1 < len(merged_pre_token)
                    and (merged_pre_token[i], merged_pre_token[i + 1]) == earliest_merge[1]
                ):
                    new_merged_pre_token.append(merged_pre_token[i] + merged_pre_token[i + 1])
                    i += 2
                else:
                    new_merged_pre_token.append(merged_pre_token[i])
                    i += 1
            merged_pre_token = new_merged_pre_token

        return [self.bytes_to_token_id[b] for b in merged_pre_token]

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
            for pre_token in pre_tokens:
                pre_token_ids = self.encode_pre_token(pre_token)
                ids.extend(pre_token_ids)

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


def reservoir_sample(reservoir: list[bytes], size: int, sample: bytes, sample_num: int, random_generator: random.Random) -> None:
    if len(reservoir) < size:
        reservoir.append(sample)
        return

    # If the sample is not selected with probability (size / sample_num), return and update nothing.
    if random_generator.random() >= (size / sample_num):
        return

    # Randomly choose which element of the reservoir to replace.
    index = random_generator.randint(0, size - 1)
    reservoir[index] = sample


def sample_documents(corpus_path: str, num_samples: int, special_token: bytes, seed: int) -> list[bytes]:
    chunk_size: int = 4096  # Read ahead by 4k bytes at a time.
    reservoir: list[bytes] = []
    random_generator = random.Random(seed)

    with open(corpus_path, "rb") as f:
        buffer: bytes = b""
        document_num: int = 0
        while True:
            chunk = f.read(chunk_size)
            if chunk == b"":
                break
            buffer += chunk

            while (found_at := buffer.find(special_token)) != -1:
                document = buffer[:found_at]
                document_num += 1
                reservoir_sample(reservoir, num_samples, document, document_num, random_generator)

                next_index = found_at + len(special_token)
                buffer = buffer[next_index:]

    return reservoir


if __name__ == "__main__":
    # tinystories_documents: list[bytes] = sample_documents(
    #     corpus_path="/home/kliuz/home/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    #     num_samples=10,
    #     special_token=b"<|endoftext|>",
    #     seed=101,
    # )
    # tinystories_tokenizer = Tokenizer.from_files(
    #     vocab_filepath="/home/kliuz/home/cs336-assignment1-basics/outputs/TinyStoriesV2-GPT4-train_vocab.json",
    #     merges_filepath="/home/kliuz/home/cs336-assignment1-basics/outputs/TinyStoriesV2-GPT4-train_merges.json",
    #     special_tokens=["<|endoftext|>"],
    # )
    owt_documents: list[bytes] = sample_documents(
        corpus_path="/home/kliuz/home/cs336-assignment1-basics/data/owt_valid.txt",
        num_samples=10000,
        special_token=b"<|endoftext|>",
        seed=101,
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/kliuz/home/cs336-assignment1-basics/outputs/owt_train_vocab.json",
        merges_filepath="/home/kliuz/home/cs336-assignment1-basics/outputs/owt_train_merges.json",
        special_tokens=["<|endoftext|>"],
    )

    # tinystories_sample_bytes: int = sum(len(doc) for doc in tinystories_documents)
    # print("tinystories total raw bytes:", tinystories_sample_bytes)
    owt_sample_bytes: int = sum(len(doc) for doc in owt_documents)
    print("owt total raw bytes:", owt_sample_bytes)

    # tinystories_tokenized_documents: list[list[int]] = [tinystories_tokenizer.encode(doc.decode("utf-8")) for doc in tinystories_documents]
    # tinystories_tokenized_bytes: int = sum(len(doc) for doc in tinystories_tokenized_documents)
    # print("tinystories total tokenized bytes:", tinystories_tokenized_bytes)
    from datetime import datetime
    start = datetime.now()
    owt_tokenized_documents: list[list[int]] = [owt_tokenizer.encode(doc.decode("utf-8")) for doc in owt_documents]
    end = datetime.now()
    owt_tokenized_bytes: int = sum(len(doc) for doc in owt_tokenized_documents)
    print("owt total tokenized bytes:", owt_tokenized_bytes)

    # print("tinystories compression ratio (bytes / token):", tinystories_sample_bytes / tinystories_tokenized_bytes)
    print("owt compression ratio (bytes / token):", owt_sample_bytes / owt_tokenized_bytes)
    print("owt tokenization throughput (bytes / s):", owt_tokenized_bytes / (end - start).seconds)

