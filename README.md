# Tiny LLM

A small character-level language model built one lesson at a time to understand
how GPT-style models work. Start with a tokenizer and a counting model, then
learn neural predictions and causal attention. PyTorch will provide tensor math
and automatic gradients; we will write the model, training, and generation.

## Current state

The character tokenizer builds a fixed vocabulary from `anna`, `bob`, and
`ava`, then translates text into integer IDs and back. The bigram counter now
counts adjacent token pairs and converts those counts into next-token
probabilities. Generation and neural training remain future lessons.

Running the package first demonstrates the mapping:

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

## Bigram counts

The CLI wraps each name with boundaries and encodes it separately. For example,
`.anna.` becomes `[0, 1, 3, 3, 1, 0]`, with overlapping pairs
`(0, 1), (1, 3), (3, 3), (3, 1), (1, 0)`.

`count_bigrams` in `src/tiny_llm/bigram.py` takes these separate sequences of
IDs and returns a standard-library `Counter`. Each pair is a key; its value is
the number of observations. Missing pairs have count zero. The function adds no
boundaries and makes no pairs between names. Empty or single-token sequences
contribute no pairs.

After the tokenizer output, the demo prints:

```text
Bigram counts (current -> next):
. -> a: 2
. -> b: 1
a -> .: 2
a -> n: 1
a -> v: 1
b -> .: 1
b -> o: 1
n -> a: 1
n -> n: 1
o -> b: 1
v -> a: 1
Total pairs: 13
```

For example, `a -> .` appears twice because both `anna` and `ava` end in
`a`. Counts are observations, not probabilities yet. Tests in
`tests/test_bigram.py` check the exact counts, overlapping pairs, separate
sequences, repeated observations, and empty inputs.

## Next-token probabilities

`next_token_probabilities(counts, current_id)` in `src/tiny_llm/bigram.py`
selects pairs starting with the current token, sums their counts, and divides
each count by that sum. It returns a dictionary of next-token IDs to
probabilities without changing the counts. Missing next tokens have probability
zero; no smoothing is applied. A token with no outgoing observations raises
`ValueError`, as do negative outgoing counts.

After `a`, we observed `.` twice, `n` once, and `v` once: four observations.
The demo prints:

```text
Next-character probabilities after 'a':
a -> .: 50%
a -> n: 25%
a -> v: 25%
```

The denominator is the total after `a` (4), not the total across all pairs (13).
The tests check all six current tokens against hand-calculated probabilities,
verify each distribution sums to one, and cover missing or invalid observations.

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
