import heapq
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex as re
from pretokenization_example import find_chunk_boundaries


class Node:
    def __init__(self, value: bytes, freq: int):
        self.value = value
        self.freq = freq
        self.next: Node | None = None


class LinkedList:
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


def pre_tokenize(chunk: str) -> Counter:
    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = [match.group(0) for match in re.finditer(pat_str, chunk)]

    return Counter(matches)


def get_pre_token_counts(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pre_token_counts = Counter()
    with open(input_path, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Parallelize by sending each start/end pair to a set of processes.
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        with Pool(processes=16) as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                sub_chunks = re.split("|".join(escaped_special_tokens), chunk)

                counters = pool.map(pre_tokenize, sub_chunks)
                merged_counters = sum(counters, start=Counter())
                pre_token_counts += merged_counters

    bytes_counts = defaultdict(int)
    for pre_token, count in pre_token_counts.items():
        key = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
        bytes_counts[key] = count

    return bytes_counts


def get_byte_pairs_to_nodes(pre_token_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], set[Node]]:
    pre_token_nodes: dict[tuple[bytes, bytes], set[Node]] = defaultdict(set)
    for pre_token, count in pre_token_counts.items():
        ll = LinkedList()
        for b in pre_token:
            ll.append(Node(b, count))

        node = ll.head
        while node is not None and node.next is not None:
            pre_token_nodes[(node.value, node.next.value)].add(node)
            node = node.next

    return pre_token_nodes


def get_byte_pairs_to_counts(
    byte_pairs_to_nodes: dict[tuple[bytes, bytes], set[Node]],
) -> dict[tuple[bytes, bytes], int]:
    byte_pairs_to_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pair, nodes in byte_pairs_to_nodes.items():
        for node in nodes:
            byte_pairs_to_counts[pair] += node.freq
    return byte_pairs_to_counts


def get_byte_pairs_max_heap(byte_pairs_to_counts: dict[tuple[bytes, bytes], int]) -> list:
    max_pq = [(-count, byte_pair) for byte_pair, count in byte_pairs_to_counts.items()]
    heapq.heapify(max_pq)
    return max_pq


def merge_pre_token_nodes(node: Node | None, pair: tuple[bytes, bytes]) -> tuple[Node | None, Node | None, int]:
    if node is None or node.next is None:
        return (None, None, 0)

    node.value = pair[0] + pair[1]
    removed = node.next
    next = removed.next
    removed.next = None
    node.next = next

    return (node, removed, node.freq)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    input_path: Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including
        the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: A list of strings to add to the vocabulary. During training, treat them as hard
        boundaries that prevent merges across their spans, but do not include them when computing
        merge statistics.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    pre_token_counts = get_pre_token_counts(input_path, special_tokens)
    byte_pairs_to_nodes = get_byte_pairs_to_nodes(pre_token_counts)
    byte_pairs_to_counts = get_byte_pairs_to_counts(byte_pairs_to_nodes)
    max_pq = get_byte_pairs_max_heap(byte_pairs_to_counts)

    # after calling merge_pre_token_nodes, we need to delete the entries for `node` and `removed`
    # for the current max pair as well as all subsequent pair, and create a new entry for the
    # merged pair + subsequent byte.

    return vocab, merges


if __name__ == "__main__":
    train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=500, special_tokens=["<|endoftext|>"])
