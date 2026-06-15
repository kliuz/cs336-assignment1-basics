import collections


def pre_tokenize(corpus: str) -> list[str]:
    return corpus.split()


def get_word_counts(words: list[str]) -> dict[tuple[bytes, ...], int]:
    counts: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    for word in words:
        key = tuple(bytes([b]) for b in bytes(word, "utf-8"))
        counts[key] += 1

    return counts


def get_pair_stats(word_counts: dict[tuple[bytes, ...], int]):
    counts: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    for word, count in word_counts.items():
        # Explicitly ignoring one-character words.
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += count

    return counts


def merge(word_counts: dict[tuple[bytes, ...], int], pair: tuple[bytes, ...]) -> dict[tuple[bytes, ...], int]:
    merged_counts: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    for word, count in word_counts.items():
        merged_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                merged_word.append(word[i] + word[i + 1])
                i += 2
            else:
                merged_word.append(word[i])
                i += 1
        
        merged_counts[tuple(merged_word)] += count
    
    return merged_counts



def main():
    corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    vocab[len(vocab) + 1] = bytes("<|endoftext|>", "utf-8")

    words = pre_tokenize(corpus)
    word_counts = get_word_counts(words)

    merges = 6
    for i in range(merges):
        pair_stats = get_pair_stats(word_counts)
        best = max(pair_stats, key=lambda p: (pair_stats.get(p, (0)), p))
        vocab[len(vocab) + 1] = best[0] + best[1]

        word_counts = merge(word_counts, best)
        print(word_counts)
    
    print(vocab)


if __name__ == "__main__":
    main()
