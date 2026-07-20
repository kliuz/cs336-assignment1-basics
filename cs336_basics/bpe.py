import heapq
import json
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries


class Node:
    def __init__(self, value: bytes, freq: int):
        self.value = value
        self.freq = freq
        self.next: Node | None = None
        self.prev: Node | None = None


class DoublyLinkedList:
    def __init__(self):
        self.head: Node | None = None

    def append(self, node: Node) -> None:
        if self.head is None:
            self.head = node
            return

        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = node
        node.prev = curr


class MaxHeapItem:
    def __init__(self, count: int, byte_pair: tuple[bytes, bytes]):
        self.count = count
        self.byte_pair = byte_pair

    def __lt__(self, other: "MaxHeapItem"):
        return (self.count, self.byte_pair) > (other.count, other.byte_pair)

    def __repr__(self):
        return f"({self.count}, [{self.byte_pair[0]}, {self.byte_pair[1]}])"


def pre_tokenize_debug(chunk: str) -> Counter:
    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = [match.group(0) for match in re.finditer(pat_str, chunk)]

    return Counter(matches)


def get_pre_token_counts_debug(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pre_token_counts = Counter()
    escaped_special_tokens = [re.escape(token) for token in special_tokens]

    chunks = re.split("|".join(escaped_special_tokens), text)
    for chunk in chunks:
        counters = pre_tokenize_debug(chunk)
        pre_token_counts += counters

    bytes_counts = defaultdict(int)
    for pre_token, count in pre_token_counts.items():
        key = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
        bytes_counts[key] = count

    return bytes_counts


def pre_tokenize(input_path: str, start: int, end: int, escaped_special_tokens) -> Counter:
    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    subchunks: list[str] = []
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        subchunks = re.split("|".join(escaped_special_tokens), chunk)

    counter: Counter = Counter()
    for subchunk in subchunks:
        for match in re.finditer(pat_str, subchunk):
            counter[match.group(0)] += 1

    return counter


def get_pre_token_counts(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pre_token_counts = Counter()
    boundaries: list[int] = []
    with open(input_path, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Parallelize by sending each start/end pair to a set of processes.
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    with Pool(processes=16) as pool:
        arguments = [
            (input_path, start, end, escaped_special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        counters = pool.starmap(pre_tokenize, arguments)
        merged_counters = sum(counters, start=Counter())
        pre_token_counts += merged_counters

    bytes_counts = defaultdict(int)
    for pre_token, count in pre_token_counts.items():
        key = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
        bytes_counts[key] = count

    return bytes_counts


class OptimizedBPE:
    def __init__(self, pre_token_counts: dict[tuple[bytes, ...], int], vocab_size: int, special_tokens: list[str]):
        self.pre_token_counts = pre_token_counts
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []
        self.byte_pairs_to_nodes: dict[tuple[bytes, bytes], list[Node]] = self._get_byte_pairs_to_nodes(
            self.pre_token_counts
        )
        self.byte_pairs_to_counts = self._get_byte_pairs_to_counts(self.byte_pairs_to_nodes)
        self.max_pq = self._get_byte_pairs_max_heap(self.byte_pairs_to_counts)

    def _get_byte_pairs_to_nodes(
        self, pre_token_counts: dict[tuple[bytes, ...], int]
    ) -> dict[tuple[bytes, bytes], list[Node]]:
        pre_token_nodes: dict[tuple[bytes, bytes], list[Node]] = defaultdict(list)
        for pre_token, count in pre_token_counts.items():
            ll = DoublyLinkedList()
            for b in pre_token:
                ll.append(Node(b, count))

            node = ll.head
            while node is not None and node.next is not None:
                pre_token_nodes[(node.value, node.next.value)].append(node)
                node = node.next

        return pre_token_nodes

    def _get_byte_pairs_to_counts(
        self,
        byte_pairs_to_nodes: dict[tuple[bytes, bytes], list[Node]],
    ) -> dict[tuple[bytes, bytes], int]:
        byte_pairs_to_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pair, nodes in byte_pairs_to_nodes.items():
            for node in nodes:
                byte_pairs_to_counts[pair] += node.freq
        return byte_pairs_to_counts

    def _get_byte_pairs_max_heap(self, byte_pairs_to_counts: dict[tuple[bytes, bytes], int]) -> list[MaxHeapItem]:
        max_pq = [MaxHeapItem(count, byte_pair) for byte_pair, count in byte_pairs_to_counts.items()]
        heapq.heapify(max_pq)
        return max_pq

    def _merge_pre_token_nodes(self, node: Node | None) -> tuple[Node | None, Node | None, int]:
        if node is None or node.next is None:
            return (None, None, 0)

        node.value = node.value + node.next.value
        removed = node.next
        next = removed.next
        if next is not None:
            next.prev = node
        removed.next = None
        node.next = next

        return (node, removed, node.freq)

    def _print_node(self, node: Node) -> None:
        value = node.value
        prev_value = None if not node.prev else node.prev.value
        next_value = None if not node.next else node.next.value

        print("(prev, curr, next):", prev_value, value, next_value)

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        while len(self.vocab) < self.vocab_size and len(self.max_pq) > 0:
            item: MaxHeapItem = heapq.heappop(self.max_pq)
            heap_count = item.count
            byte_pair = item.byte_pair
            true_count = self.byte_pairs_to_counts[byte_pair]

            if true_count == 0:
                continue
            elif heap_count != true_count:
                heapq.heappush(self.max_pq, MaxHeapItem(true_count, byte_pair))
                continue

            self.merges.append(byte_pair)
            merged_pair: bytes = byte_pair[0] + byte_pair[1]
            self.vocab[len(self.vocab)] = merged_pair

            for node in self.byte_pairs_to_nodes[byte_pair]:
                if not node or node.next is None:
                    continue
                if node.value != byte_pair[0] or node.next.value != byte_pair[1]:
                    continue

                merged_node, removed_node, count = self._merge_pre_token_nodes(node)
                if not merged_node or not removed_node:
                    continue

                node_before = merged_node.prev
                node_after = merged_node.next
                if node_before is not None:
                    old_byte_pair = (node_before.value, byte_pair[0])
                    if old_byte_pair in self.byte_pairs_to_counts:
                        self.byte_pairs_to_counts[old_byte_pair] -= count
                    new_byte_pair = (node_before.value, merged_pair)
                    self.byte_pairs_to_counts[new_byte_pair] += count
                    self.byte_pairs_to_nodes[new_byte_pair].append(node_before)
                    heapq.heappush(self.max_pq, MaxHeapItem(self.byte_pairs_to_counts[new_byte_pair], new_byte_pair))
                if node_after is not None:
                    old_byte_pair = (byte_pair[1], node_after.value)
                    if old_byte_pair in self.byte_pairs_to_counts:
                        self.byte_pairs_to_counts[old_byte_pair] -= count
                    new_byte_pair = (merged_pair, node_after.value)
                    self.byte_pairs_to_counts[new_byte_pair] += count
                    self.byte_pairs_to_nodes[new_byte_pair].append(merged_node)
                    heapq.heappush(self.max_pq, MaxHeapItem(self.byte_pairs_to_counts[new_byte_pair], new_byte_pair))
            self.byte_pairs_to_counts.pop(byte_pair)
            self.byte_pairs_to_nodes.pop(byte_pair)

        return self.vocab, self.merges


class UnoptimizedBPE:
    def __init__(self, pre_token_counts: dict[tuple[bytes, ...], int], vocab_size: int, special_tokens: list[str]):
        self.pre_token_counts = pre_token_counts
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []

    def get_pair_stats(self) -> dict[tuple[bytes, bytes], int]:
        counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pre_token, count in self.pre_token_counts.items():
            # Explicitly ignoring one-character words.
            for i in range(len(pre_token) - 1):
                counts[(pre_token[i], pre_token[i + 1])] += count

        return counts

    def train(self):
        while len(self.vocab) < self.vocab_size and len(self.pre_token_counts) > 0:
            pair_stats = self.get_pair_stats()
            if not pair_stats:
                break
            best_pair = max(pair_stats, key=lambda p: (pair_stats.get(p, 0), p))
            self.merges.append(best_pair)
            self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]

            merged_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
            for pre_token, count in self.pre_token_counts.items():
                merged_word: list[bytes] = []
                i = 0
                while i < len(pre_token):
                    if i + 1 < len(pre_token) and (pre_token[i], pre_token[i + 1]) == best_pair:
                        merged_word.append(pre_token[i] + pre_token[i + 1])
                        i += 2
                    else:
                        merged_word.append(pre_token[i])
                        i += 1

                merged_counts[tuple(merged_word)] += count
            self.pre_token_counts = merged_counts

        return self.vocab, self.merges


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    input_path: Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including
        the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: A list of strings to add to the vocabulary. During training, treat them as hard
        boundaries that prevent merges across their spans, but do not include them when computing
        merge statistics.
    """
    pre_token_counts = get_pre_token_counts(input_path, special_tokens=["<|endoftext|>"])
    bpe = OptimizedBPE(pre_token_counts, vocab_size, special_tokens)
    vocab, merges = bpe.train()

    return vocab, merges


def fuzz_implementations(alphabet: str, text_len: int, vocab_size: int, special_tokens: list[str]):
    while True:
        text = "".join(random.choices(alphabet, k=text_len)) + "<|endoftext|>"

        header = "================ text: " + text + " ================"
        print(header)
        pre_token_counts = get_pre_token_counts_debug(text, special_tokens)
        bpe_opt = OptimizedBPE(pre_token_counts, vocab_size, special_tokens)
        bpe_unopt = UnoptimizedBPE(pre_token_counts, vocab_size, special_tokens)

        vocab_opt, merges_opt = bpe_opt.train()
        vocab_unopt, merges_unopt = bpe_unopt.train()

        if merges_opt != merges_unopt:
            print("merges_opt", merges_opt)
            print("merges_unopt", merges_unopt)
            break

        print("=" * len(header))


def debug_specific_example(text: str, vocab_size: int, special_tokens: list[str]):
    header = "================ text: " + text + " ================"
    print(header)
    pre_token_counts = get_pre_token_counts_debug(text, special_tokens)
    bpe_opt = OptimizedBPE(pre_token_counts, vocab_size, special_tokens)
    bpe_unopt = UnoptimizedBPE(pre_token_counts, vocab_size, special_tokens)

    vocab_opt, merges_opt = bpe_opt.train()
    vocab_unopt, merges_unopt = bpe_unopt.train()

    if merges_opt != merges_unopt:
        print("merges_opt", merges_opt)
        print("merges_unopt", merges_unopt)
    print("=" * len(header))


def serialize_vocab_and_merges(bpe: OptimizedBPE, vocab_path: str, merge_path: str):
    vocab: dict[int, str] = {}
    merges: list[tuple[str, str]] = []

    for k, v in bpe.vocab.items():
        vocab[k] = v.decode("latin-1")
    for v in bpe.merges:
        merges.append((v[0].decode("latin-1"), v[1].decode("latin-1")))

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    with open(merge_path, "w") as f:
        json.dump(merges, f)


if __name__ == "__main__":
    # debug_specific_example(text="bababbab", vocab_size=257 + 4, special_tokens=["<|endoftext|>"])
    # fuzz_implementations(alphabet="ab", text_len=8, vocab_size=257 + 4, special_tokens=["<|endoftext|>"])

    print("Starting BPE training!")
    pre_token_counts = get_pre_token_counts(
        input_path="data/TinyStoriesV2-GPT4-train.txt", special_tokens=["<|endoftext|>"]
    )
    print("Finished pretokenization.")
    bpe = OptimizedBPE(pre_token_counts, vocab_size=10000, special_tokens=["<|endoftext|>"])
    print("Finished initializing OptimizedBPE")
    bpe.train()
    print("Completed tokenization training")
    serialize_vocab_and_merges(
        bpe,
        vocab_path="outputs/TinyStoriesV2-GPT4-train_vocab.json",
        merge_path="outputs/TinyStoriesV2-GPT4-train_merges.json",
    )
