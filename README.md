# Tiny LLM

A small character-level language model built one lesson at a time to understand
how GPT-style models work. Start with a tokenizer and a counting model, then
learn neural predictions and causal attention. PyTorch will provide tensor math
and automatic gradients; we will write the model, training, and generation.

## Current state

The Python package and development tools are ready. Running the package prints
a workspace readiness message. The first lesson, the character tokenizer, is
next; no model or training code is implemented yet.

## Development

Use Python 3.14.4 and uv 0.13.7 or newer. uv will obtain the pinned Python version
if it is not available locally. From the repository root:

```sh
uv sync --locked
uv run --locked python -m tiny_llm
uv run --locked python -m tiny_llm --help
```

Dependencies are locked in `uv.lock`. Ruff formats and lints the code, Pyrefly
checks types, and pytest will run the lesson tests. Runtime code currently uses
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

There are no automated tests yet: pytest currently reports no tests collected
and exits with code 5. Tests will arrive with the tokenizer lesson. To apply
formatting, run `uv run --locked ruff format .`.

Source code lives in `src/tiny_llm/`, and `tests/` is ready for the first lesson.
Add small text datasets under `data/` when needed; generated results belong in
ignored `runs/`. Run locally on CPU to begin.

See [AGENTS.md](AGENTS.md) for the engineering conventions. `CLAUDE.md` links to
the same instructions.
