from collections import Counter

import pytest

from tiny_llm.bigram import count_bigrams, next_token_probabilities
from tiny_llm.tokenizer import CharacterTokenizer


def test_counts_for_training_names() -> None:
    names = ("anna", "bob", "ava")
    tokenizer = CharacterTokenizer(names)
    sequences = [tokenizer.encode(f".{name}.") for name in names]

    counts = count_bigrams(sequences)

    # IDs are ordered as ".", "a", "b", "n", "o", "v".
    assert dict(counts) == {
        (0, 1): 2,
        (0, 2): 1,
        (1, 0): 2,
        (1, 3): 1,
        (1, 5): 1,
        (2, 0): 1,
        (2, 4): 1,
        (3, 1): 1,
        (3, 3): 1,
        (4, 2): 1,
        (5, 1): 1,
    }
    assert counts.total() == 13
    assert counts[0, 0] == 0


def test_overlapping_pairs_are_counted() -> None:
    assert dict(count_bigrams([[1, 1, 1, 1]])) == {(1, 1): 3}


def test_sequences_stay_separate_and_inputs_are_unchanged() -> None:
    sequences = [[0, 1], [2, 0]]

    assert dict(count_bigrams(sequences)) == {(0, 1): 1, (2, 0): 1}
    assert sequences == [[0, 1], [2, 0]]


def test_empty_and_single_token_sequences_have_no_pairs() -> None:
    assert dict(count_bigrams([])) == {}
    assert dict(count_bigrams([[], [0], []])) == {}


def test_repeated_sequences_add_counts() -> None:
    sequences = ([0, 1, 0] for _ in range(2))

    assert dict(count_bigrams(sequences)) == {(0, 1): 2, (1, 0): 2}


@pytest.mark.parametrize(
    ("current_id", "expected"),
    [
        (0, {1: 2 / 3, 2: 1 / 3}),
        (1, {0: 0.5, 3: 0.25, 5: 0.25}),
        (2, {0: 0.5, 4: 0.5}),
        (3, {1: 0.5, 3: 0.5}),
        (4, {2: 1.0}),
        (5, {1: 1.0}),
    ],
)
def test_probabilities_for_training_names(
    current_id: int, expected: dict[int, float]
) -> None:
    names = ("anna", "bob", "ava")
    tokenizer = CharacterTokenizer(names)
    counts = count_bigrams(tokenizer.encode(f".{name}.") for name in names)
    original_counts = counts.copy()

    probabilities = next_token_probabilities(counts, current_id)

    assert probabilities == pytest.approx(expected)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert counts == original_counts


@pytest.mark.parametrize("counts", [{}, {(2, 0): 3}, {(1, 0): 0}])
def test_probabilities_require_outgoing_observations(
    counts: dict[tuple[int, int], int],
) -> None:
    with pytest.raises(ValueError, match="No outgoing observations for token ID: 1"):
        next_token_probabilities(Counter(counts), 1)


def test_probabilities_reject_negative_outgoing_counts() -> None:
    counts = Counter({(1, 0): -1, (1, 3): 2})

    with pytest.raises(ValueError, match="Negative outgoing count for token ID: 1"):
        next_token_probabilities(counts, 1)
