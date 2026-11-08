"""Count adjacent token pairs and calculate next-token probabilities."""

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise


def count_bigrams(sequences: Iterable[Sequence[int]]) -> Counter[tuple[int, int]]:
    """Count overlapping pairs of token IDs without crossing sequence boundaries.

    Callers supply tokenized sequences with any desired boundary tokens.
    Empty sequences and single tokens contribute no pairs.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for token_ids in sequences:
        for current_id, next_id in pairwise(token_ids):
            counts[current_id, next_id] += 1
    return counts


def next_token_probabilities(
    counts: Counter[tuple[int, int]], current_id: int
) -> dict[int, float]:
    """Normalize outgoing counts for one token; omitted next tokens have probability zero."""
    next_counts = {
        next_id: count
        for (previous_id, next_id), count in counts.items()
        if previous_id == current_id
    }
    if any(count < 0 for count in next_counts.values()):
        raise ValueError(f"Negative outgoing count for token ID: {current_id}")

    total = sum(next_counts.values())
    if total == 0:
        raise ValueError(f"No outgoing observations for token ID: {current_id}")

    return {next_id: count / total for next_id, count in next_counts.items()}
