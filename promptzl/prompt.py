"""Promptzl, 2024.

MIT LICENSE
"""

from typing import List


class Text:
    """Docstring TODO."""

    def __init__(self, text):
        """Initialize TODO."""
        self.text = text


class Key:
    """Docstring TODO."""

    def __init__(self, key):
        """Initialize TODO."""
        self.key = key


# class Mask:
#     """Docstring TODO."""

#     pass


class Verbalizer:
    """Docstring TODO."""

    def __init__(self, verbalizer: List[List[str]]):
        """Initialize TODO."""
        self.verbalizer: List[List[str]] = verbalizer


class Prompt:
    """Docstring TODO."""

    def __init__(self, *args, sep=" "):
        """Initialize TODO."""
        self.prompt = args if len(args) > 1 else [args[0]]
        self.sep = sep
        self.tokenizer = None

    def subinit(self, tokenizer, generate: bool):
        """Docstring TODO."""
        self.tokenizer = tokenizer
        self.generate = generate
        verb_filtered: List[Verbalizer] = [
            e for e in self.prompt if isinstance(e, Verbalizer)
        ]
        assert len(verb_filtered) == 1, "One Verbalizer must be provided in prompt."
        assert True in [
            isinstance(e, Verbalizer) for e in self.prompt
        ], "The prompt must contain the verbalizer. E.g. `Prompt(Key('text'), Text('It was '), Verbalizer([['good'], ['bad']]))`."
        if generate:
            assert isinstance(
                self.prompt[-1], Verbalizer
            ), "When using `CausalModel4Classification`, the last token must be of type `Verbalizer`."
        self.verbalizer: Verbalizer = verb_filtered[0]

    def decide(self, e, data) -> str:
        """Docstring TODO."""
        assert (
            self.tokenizer is not None
        ), "You must call `subinit` before calling `get_text`"
        if isinstance(e, Text):
            return e.text
        elif isinstance(e, Key):
            return data[e.key]
        elif isinstance(e, Verbalizer):
            return (
                self.tokenizer.mask_token if not self.generate else ""
            )  # TODO check of eos?
        else:
            raise NotImplementedError(f"Type '{type(e)}' not considered.")

    def get_text(self, data):
        """Docstring TODO."""
        return self.sep.join([self.decide(e, data) for e in self.prompt])

    def prepare_dataset(self, dataset, padding="do_not_pad"):
        """Docstring TODO."""
        return dataset.map(
            lambda e: self.tokenizer(
                self.get_text(e), padding=padding, truncation=True
            ),
            remove_columns=dataset.column_names,
        )
