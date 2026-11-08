# Tiny LLM engineering contract

## Status and purpose

The package, character tokenizer, and tokenizer tests are implemented. The CLI
demonstrates vocabulary construction, encoding, and decoding, and supports
`--help`. Models, external datasets, and training remain future lessons. Do not
report checks as passing before they exist.

Build a tiny character-level language model to learn how GPT-style models work.
Work through one understandable lesson at a time: tokenization, character
counts, learned predictions, then a small causal Transformer. The first feature
is a character tokenizer with a reversible text-to-ID mapping.

Run locally from the terminal on Python 3.14, initially on CPU. Use PyTorch for
tensor operations and automatic differentiation when the neural lessons begin;
write the model, loss calculation, training loop, and generation ourselves.
Weights start randomly. Keep the relevant mathematics visible in ordinary
Python; do not introduce pretrained models or high-level training frameworks.

## Stack and dependencies

- Use one packaged application named `tiny-llm`, imported as `tiny_llm`.
- Use Python `>=3.14,<3.15`; pin the selected installed patch in
  `.python-version`. Python 3.14.4 is the available initial interpreter.
- Use uv for environment management, dependency changes, execution, and builds.
  Require uv 0.13.7 or newer and use `uv_build>=0.13.7,<0.14` as the build
  backend.
- Start with no runtime dependencies. Use the standard library for tokenization,
  file handling, argument parsing, and the counting model.
- Put Ruff 0.17.4, Pyrefly 1.4.1, and pytest 9.2.0 in the `dev` dependency group,
  initially pinned with `==`. These are the sole formatter/linter, type checker,
  and test runner respectively.
- Add `torch` when the neural lesson starts; select a stable release compatible
  with Python 3.14 and macOS ARM64, and record it in the lockfile. The choice to
  use PyTorch is already agreed. Ask before adding any other dependency.
- Keep project metadata and all tool configuration in `pyproject.toml`. Commit
  `uv.lock`; update it through uv, never by hand. Do not add a parallel
  `requirements.txt` or environment manager.
- Configure the editor to use `.venv` and the project's Pyrefly version. Use
  ordinary Python types initially; experimental tensor-shape typing is not part
  of this baseline.

## Layout and ownership

The layout below distinguishes implemented files from future lesson modules.
Create the planned files only when a lesson needs them.

```text
tiny-llm/
  AGENTS.md
  CLAUDE.md                 # Relative symlink to AGENTS.md
  README.md
  pyproject.toml            # Project and tool configuration
  uv.lock                   # Resolved dependencies
  .python-version           # Interpreter pin
  .gitignore                # Exclusions
  src/tiny_llm/
    __init__.py              # Package marker; no work at import time
    __main__.py              # CLI entry point
    tokenizer.py             # Vocabulary, encode, decode
    bigram.py                # Later: counting and next-character probabilities
    model.py                 # Later: neural model and causal attention
    train.py                 # Later: batches, loss, and parameter updates
    generate.py              # Later: autoregressive sampling
  tests/
    test_tokenizer.py        # Tokenizer behavioral tests
  data/                      # Planned small, explicitly selected text corpora
  runs/                      # Future ignored generated output and checkpoints
```

Create a module when its lesson needs it. Keep related functions together; do
not introduce generic `utils`, registries, service layers, or interfaces for a
single implementation. Keep earlier working models available as later lessons
arrive, sharing the tokenizer rather than copying it.

`__main__.py` owns arguments and user-facing output. It calls the relevant
lesson functions. Training and generation may use models and tokenization;
models operate on numeric inputs and do not import the CLI, training, or
generation modules. Tokenization has no dependency on other project modules.
Never import tests from source or introduce import cycles.

## Python conventions

- Use `snake_case` for modules, functions, and variables; `PascalCase` for
  classes; and `UPPER_SNAKE_CASE` for constants. Name dimensions descriptively
  at boundaries, such as `batch_size`, `context_length`, and `embedding_dim`.
- Annotate function parameters and returns, including tests. Use built-in
  generics and `T | None`; use `collections.abc` interfaces for read-only inputs.
  Use explicit `None` for absence. Do not use `Any` or casts to evade a mismatch.
- Prefer functions. Use a class for owned state, such as a fitted tokenizer,
  and `torch.nn.Module` for a trainable neural model. Extract a helper only for
  current reuse or a separate substantial responsibility.
- Keep module imports free of file reads, downloads, training, or random draws.
  Use absolute imports from `tiny_llm`; never modify `sys.path` or `PYTHONPATH`.
- Let Ruff format code with its default 88-character line length. Configure
  `target-version = "py314"` and lint rules `E4`, `E7`, `E9`, `F`, `I`, `UP`,
  `B`, and `ANN`. `ANN` enforces annotated function boundaries; Pyrefly checks
  their consistency.
- Configure Pyrefly in `pyproject.toml` with `project-includes = ["src", "tests"]`,
  `python-version = "3.14"`, `preset = "strict"`, and
  `check-unannotated-defs = true`. Keep these checks active for tests too.
- Comments explain mathematics, dimensions, or a non-obvious decision. Keep
  explanations of the lesson in the README alongside runnable examples once
  those examples exist.

## Inputs, errors, and local data

- Use `pathlib.Path` and UTF-8 text files. Read data explicitly at the CLI or
  training boundary and pass it to computation functions. Resolve user-supplied
  relative paths against the current working directory.
- Validate external input at its owning boundary. Raise `ValueError` for invalid
  values and preserve `OSError` for filesystem failures. Include the invalid
  value or path when useful; never silently substitute data or drop bad input.
- The tokenizer owns its vocabulary and validates unknown characters and token
  IDs. Its mapping must be deterministic. Use `.` as the name boundary token
  with ID 0, followed by sorted observed characters; reject literal `.` in
  training names. Preserve other characters exactly rather than normalizing
  case or whitespace implicitly. Encoding and decoding preserve the text
  exactly, including any explicit boundary tokens: `encode("anna")` adds no
  boundaries, and decoding ID 0 produces `.`. The caller wraps a training name
  as `.anna.`. An empty training corpus or empty name is
  invalid; encoding an empty text and decoding an empty ID list return empty
  results. These behaviors are enforced by tokenizer tests.
- Keep internal programming errors visible. At the CLI, use `argparse` for usage
  errors; catch expected input or filesystem errors narrowly, print a concise
  error to stderr, and return a nonzero status. Do not catch `Exception` or add
  logging inside numerical functions.
- Keep small approved corpora under `data/`, with source and license information
  beside downloaded datasets. No automatic downloads on import or test runs.
  Use `tmp_path` for filesystem tests. No credentials or external services are
  required for the initial project.
- Ignore `.venv/`, caches, `dist/`, and `runs/`. Write generated results under
  `runs/`; keep model weights and generated corpora out of source control.

## Neural lessons

Use PyTorch primitives and ordinary layers such as `nn.Linear` and
`nn.Embedding`; implement causal attention explicitly rather than substituting
an entire Transformer implementation. Keep the forward pass, loss, backward
pass, and optimizer step readable in the owning modules.

Document tensor shapes at function boundaries and assert required shape and
dtype invariants in tests. Ordinary `Tensor` annotations do not establish
dimensions or numerical correctness. Use explicit seeds for repeatable local
experiments, without promising bit-identical results across devices. Split
training and evaluation examples before fitting the model, keep targets hidden
from causal predictions, and use evaluation mode with gradients disabled for
generation. Tests enforce each behavior when its lesson is implemented.

## Verification

Run `uv sync --locked` from the repository root to create or synchronize the
environment. Run the following commands from the same directory:

```sh
uv run --locked python -m tiny_llm
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyrefly check
uv run --locked pytest
uv build
```

Configure pytest with `testpaths = ["tests"]` and
`addopts = ["--import-mode=importlib"]`. Run checks through the installed package;
do not add path hacks to make imports pass. Use `uv run --locked ruff format .`
to apply formatting, then rerun the format check.

Write tests for observable behavior and mathematical invariants, using small
examples whose expected answers can be checked by hand. The first feature lives
in `tokenizer.py`, with `tests/test_tokenizer.py` covering round trips,
deterministic IDs, boundaries, empty inputs, and invalid characters/IDs. Later,
use `torch.testing.assert_close` for expected numeric results, and test that
causal predictions cannot depend on future tokens. Keep tests CPU-only, seeded
when random, and independent of the network or large training jobs.

Do not create placeholder tests to obtain a green result. State which checks ran
and their actual outcome. Review module ownership and
dependency direction during code review; tooling does not enforce those rules.

## Consistency contract

- One way per responsibility: no competing library, parallel pattern, or second
  approach without first changing the convention.
- Fix problems where they belong: no bypasses, silent fallbacks, duplicate
  models, pass-through wrappers, or compatibility shims. Add abstractions only
  for a current need. Distinct lesson models are intentional; copies that work
  around defects are not.
- Never disable checks, weaken types, suppress errors, or skip tests to make
  code pass.
- When a requirement does not fit, explain the conflict and propose a convention
  change. Once approved, update the convention and affected code together.
