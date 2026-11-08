"""Run the current Tiny LLM lesson from the terminal."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn how language models work, one small lesson at a time.",
    )
    parser.parse_args()
    print("Tiny LLM workspace is ready. Next: build the character tokenizer.")


if __name__ == "__main__":
    main()
