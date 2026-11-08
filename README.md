# Tiny LLM

A small character-level language model built one lesson at a time to understand
how GPT-style models work. Start with a tokenizer and a counting model, then
learn neural predictions and causal attention. PyTorch will provide tensor math
and automatic gradients; we will write the model, training, and generation.

## Current state

This directory contains foundation docs only. The Python package, configuration,
lockfile, dependencies, and tests have not been scaffolded. The first feature
will encode characters as integer IDs and decode them back to the original text.

## Planned development setup

Use Python 3.14 and uv. The planned baseline is uv 0.13.7, uv_build 0.13.7,
Ruff 0.17.4, Pyrefly 1.4.1, and pytest 9.2.0. Python 3.14.4 is already available
locally. The initial code will use only the standard library; add PyTorch when
the neural-network lessons begin.

After scaffolding has created `pyproject.toml`, the package, and `uv.lock`,
these commands will be available from this directory:

```sh
uv sync --locked
uv run --locked python -m tiny_llm
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyrefly check
uv run --locked pytest
uv build
```

Tests become available with the first implemented feature. Source code will
live in `src/tiny_llm/`, tests in `tests/`, small text datasets in `data/`, and
generated results in ignored `runs/`. Run locally on CPU to begin.

See [AGENTS.md](AGENTS.md) for the engineering conventions. `CLAUDE.md` links to
the same instructions.
