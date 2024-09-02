"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, List, Union

from datasets import Dataset


class Text:
    """Class for Text within the Prompt Object."""

    def __init__(self, text: str):
        """Initialize Text Object.

        Args:
            text (str): Text used in the prompt. E.g.: ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```
        """
        self.text = text


class Key:
    """Class for a Dataset-Key within the Prompt Object."""

    def __init__(self, key: str = "text"):
        """Initialize Key Object.

        Args:
            key (str): Key used to access the text in the dataset. E.g.: ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```
        """
        self.key = key


class Verbalizer:
    """Class for Verbalizer within the Prompt Object."""

    def __init__(self, verbalizer: List[List[str]]):
        """Initialize Text Object.

        Args:
            verbalizer (List[List[str]]): Verbalizer used in the prompt. E.g.: ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```.
                It is important to select appropriate label words.
        """
        self.verbalizer: List[List[str]] = verbalizer


class Prompt:
    """Prompt Class."""

    def __init__(self, *args, sep: str = " "):
        """Initialize Prompt.

        Args:
            *args: Key, Text and Verbalizer Objects passed as arguments for this class.
                Only one verbalizer per prompt and arbitrary many texts and keys possible. In case of a causal model,
                the Verbalizer must be at the end of the arguments.
            sep (str): Seperator for combining text and data,
                i.e. `Prompt(Key('text'), Text('It was ...') ..., sep='#)` -> `'lorem ipsum...#It was ...'`
        """
        self.prompt = args if len(args) > 1 else [args[0]]
        self.sep = sep
        self.tokenizer = None

    def subinit(self, tokenizer: Any, generate: bool):
        """Subinitialization for Main Class.

        Second initializatio that happens hidden in the main class.

        Args:
            tokenizer (Any): Tokenizer for preparing the dataset and using the mask token.
            generate (bool): Flag for using causal language models, set to False when using MLM-based models.
        """
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

    def decide(self, elem: Union[Key, Text, Verbalizer], data: Dataset) -> str:
        """Decide String for Prompt.

        Decide the appropriate string for the position within the prompt.

        Args:
            elem (Union[Key, Text, Verbalizer]): Element from prompt (i.e. what
                to do at position idx for ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```)
            data (Dataset): Data for which the key(s) will be used to index the column(s).

        Raises:
            AssertionError: If tokenizer is not set and `subinit` has not been called yet.
            NotImplementedEror: If a wrong `elem` has been passed that is not `Union[Key, Text, Verbalizer]`.

        Returns:
            str: Formated string.
        """
        assert (
            self.tokenizer is not None
        ), "You must call `subinit` before calling `get_text`"
        if isinstance(elem, Text):
            return elem.text
        elif isinstance(elem, Key):
            return data[elem.key]
        elif isinstance(elem, Verbalizer):
            return (
                self.tokenizer.mask_token if not self.generate else ""
            )  # TODO check of eos?
        else:
            raise NotImplementedError(f"Type '{type(elem)}' not considered.")

    def get_text(self, data: Dataset) -> str:
        """Join Prompt to String.

        Join prompt to string with `sep` seperator.

        Args:
            data (Dataset): Data for which the key(s) will be used to index the column(s).

        Returns:
            str: Prompt in natural language.
        """
        return self.sep.join([self.decide(e, data) for e in self.prompt])

    def prepare_dataset(self, dataset: Dataset, padding="do_not_pad"):
        """Prepare Dataset.

        Args:
            dataset (Dataset): Data for which the key(s) will be used to index the column(s).
            padding (str): Padding strategy for tokenizer. # TODO: rm?

        Raises:
            AssertionError: If tokenizer is not set and `subinit` has not been called yet.

        Returns:
            Dataset: Tokenized dataset with incorporated prompts.
        """
        assert (
            self.tokenizer is not None
        ), "You must call `subinit` before calling `prepare_dataset`"
        return dataset.map(
            lambda e: self.tokenizer(
                self.get_text(e), padding=padding, truncation=True
            ),
            remove_columns=dataset.column_names,
        )
