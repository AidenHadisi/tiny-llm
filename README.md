# Tiny LLM

A small character-level language model built one lesson at a time to understand
how GPT-style models work. Start with a tokenizer and a counting model, then
learn neural predictions and causal attention. PyTorch will provide tensor math
and automatic gradients; we will write the model, training, and generation.

## Current state

The first lesson is a character tokenizer. It builds a fixed vocabulary from
`anna`, `bob`, and `ava`, then translates text into integer IDs and back. There
is no model or training code yet.

Running the package demonstrates the mapping:

```text
Vocabulary: {0: '.', 1: 'a', 2: 'b', 3: 'n', 4: 'o', 5: 'v'}
Text: .anna.
Token IDs: [0, 1, 3, 3, 1, 0]
Decoded: .anna.
```

The boundary marker `.` always has ID 0; the remaining characters are sorted.
The tokenizer preserves the supplied text exactly: it does not insert or remove
boundaries. The demo explicitly supplies `.anna.`. Unknown characters and
invalid token IDs raise `ValueError`; encoding never grows the vocabulary.

Read `src/tiny_llm/tokenizer.py` in order: vocabulary construction, encoding,
then decoding. The tests in `tests/test_tokenizer.py` show the expected behavior
with small examples.

## Development

Use Python 3.14.4 and uv 0.13.7 or newer. uv will obtain the pinned Python version
if it is not available locally. From the repository root:

```sh
uv sync --locked
uv run --locked python -m tiny_llm
uv run --locked python -m tiny_llm --help
```

Dependencies are locked in `uv.lock`. Ruff formats and lints the code, Pyrefly
checks types, and pytest runs the lesson tests. Runtime code currently uses
only the standard library; PyTorch will be added for the neural lessons.

Point your editor at `.venv` and use the project-installed Pyrefly for type
checking so editor diagnostics match the command-line checks.

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyrefly check
uv run --locked pytest
uv build
```

To apply formatting, run `uv run --locked ruff format .`.

Source code lives in `src/tiny_llm/`, and behavioral tests live in `tests/`.
Add small text datasets under `data/` when needed; generated results belong in
ignored `runs/`. Run locally on CPU to begin.

See [AGENTS.md](AGENTS.md) for the engineering conventions. `CLAUDE.md` links to
the same instructions.
