"""A reversible character vocabulary with an explicit name boundary token."""

from collections.abc import Sequence


class CharacterTokenizer:
    """Build a fixed character-to-ID mapping from training names."""

    def __init__(self, names: Sequence[str]) -> None:
        if not names:
            raise ValueError("Training corpus must not be empty")

        characters: set[str] = set()
        for name in names:
            if not name:
                raise ValueError("Training names must not be empty")
            if "." in name:
                raise ValueError(f"Training name contains boundary token '.': {name!r}")
            characters.update(name)

        self._vocabulary: tuple[str, ...] = (".", *sorted(characters))
        self._character_to_id: dict[str, int] = {
            character: token_id for token_id, character in enumerate(self._vocabulary)
        }

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return characters in token-ID order; the boundary '.' always has ID 0."""
        return self._vocabulary

    def encode(self, text: str) -> list[int]:
        """Map each character to an ID without adding boundary tokens."""
        token_ids: list[int] = []
        for character in text:
            if character not in self._character_to_id:
                raise ValueError(f"Unknown character: {character!r}")
            token_ids.append(self._character_to_id[character])
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Map each ID to a character, preserving any boundary tokens."""
        characters: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self._vocabulary):
                raise ValueError(f"Invalid token ID: {token_id}")
            characters.append(self._vocabulary[token_id])
        return "".join(characters)
