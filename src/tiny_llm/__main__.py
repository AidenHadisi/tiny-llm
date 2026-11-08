"""Run the current Tiny LLM lesson from the terminal."""

import argparse

from tiny_llm.tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode and decode a name with a character tokenizer.",
    )
    parser.parse_args()

    tokenizer = CharacterTokenizer(("anna", "bob", "ava"))
    text = ".anna."
    token_ids = tokenizer.encode(text)

    print("Vocabulary:", dict(enumerate(tokenizer.vocabulary)))
    print("Text:", text)
    print("Token IDs:", token_ids)
    print("Decoded:", tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
