from collections.abc import Sequence

import pytest

from tiny_llm.tokenizer import CharacterTokenizer


def test_vocabulary_and_explicit_boundary_ids() -> None:
    tokenizer = CharacterTokenizer(["anna", "bob", "ava"])

    assert tokenizer.vocabulary == (".", "a", "b", "n", "o", "v")
    assert tokenizer.encode(".anna.") == [0, 1, 3, 3, 1, 0]
    assert tokenizer.decode([0, 1, 3, 3, 1, 0]) == ".anna."
    assert tokenizer.encode(".") == [0]
    assert tokenizer.decode([0]) == "."


def test_plain_name_has_no_implicit_boundaries() -> None:
    tokenizer = CharacterTokenizer(["anna", "bob", "ava"])

    assert tokenizer.encode("anna") == [1, 3, 3, 1]
    assert tokenizer.decode([1, 3, 3, 1]) == "anna"


@pytest.mark.parametrize("text", ["aA", " a\t\n", "Zoë李🙂", "e\u0301é"])
def test_round_trip_preserves_characters(text: str) -> None:
    tokenizer = CharacterTokenizer([text])

    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.encode(text)) == len(text)


def test_mapping_is_independent_of_name_order_and_duplicates() -> None:
    first = CharacterTokenizer(["anna", "bob", "ava"])
    second = CharacterTokenizer(["ava", "anna", "bob", "anna"])

    assert first.vocabulary == second.vocabulary
    assert first.encode(".ava.bob.anna.") == second.encode(".ava.bob.anna.")


def test_empty_text_and_ids() -> None:
    tokenizer = CharacterTokenizer(["anna"])

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([], "empty"),
        ([""], "empty"),
        (["anna", ""], "empty"),
        (["."], "boundary"),
        (["an.na"], "boundary"),
    ],
)
def test_invalid_training_names(names: Sequence[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CharacterTokenizer(names)


def test_unknown_character_does_not_change_vocabulary() -> None:
    tokenizer = CharacterTokenizer(["anna"])
    original_vocabulary = tokenizer.vocabulary

    with pytest.raises(ValueError, match="Unknown character: 'z'"):
        tokenizer.encode("anz")

    assert tokenizer.vocabulary == original_vocabulary
    with pytest.raises(ValueError, match="Unknown character: 'z'"):
        tokenizer.encode("z")
    assert tokenizer.encode(".anna.") == [0, 1, 2, 2, 1, 0]


@pytest.mark.parametrize("token_id", [-1, -3, 3, 100])
def test_invalid_token_ids(token_id: int) -> None:
    tokenizer = CharacterTokenizer(["anna"])

    with pytest.raises(ValueError, match=f"Invalid token ID: {token_id}"):
        tokenizer.decode([token_id])
