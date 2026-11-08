"""Run the current Tiny LLM lesson from the terminal."""

import argparse

from tiny_llm.bigram import count_bigrams, next_token_probabilities
from tiny_llm.tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize names, count pairs, and calculate next-token probabilities.",
    )
    parser.parse_args()

    names = ("anna", "bob", "ava")
    tokenizer = CharacterTokenizer(names)
    text = ".anna."
    token_ids = tokenizer.encode(text)

    print("Vocabulary:", dict(enumerate(tokenizer.vocabulary)))
    print("Text:", text)
    print("Token IDs:", token_ids)
    print("Decoded:", tokenizer.decode(token_ids))

    sequences = [tokenizer.encode(f".{name}.") for name in names]
    counts = count_bigrams(sequences)

    print("\nBigram counts (current -> next):")
    for (current_id, next_id), count in sorted(counts.items()):
        current = tokenizer.vocabulary[current_id]
        following = tokenizer.vocabulary[next_id]
        print(f"{current} -> {following}: {count}")
    print("Total pairs:", counts.total())

    current_id = tokenizer.encode("a")[0]
    probabilities = next_token_probabilities(counts, current_id)
    print("\nNext-character probabilities after 'a':")
    for next_id, probability in sorted(probabilities.items()):
        following = tokenizer.vocabulary[next_id]
        print(f"a -> {following}: {probability:.0%}")


if __name__ == "__main__":
    main()
