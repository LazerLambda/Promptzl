"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, List, Optional, Union

from datasets import Dataset
from torch import tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Tokenizable:
    """Base Class for Tokenizable Objects."""

    def __init__(self) -> None:
        """Initialize Tokenized Object."""
        self.text_tokenized = tensor([], dtype=int)


class Text(Tokenizable):
    """Class for Text within the Prompt Object."""

    def __init__(self, text: str):
        """Initialize Text Object.

        Args:
            text (str): Text used in the prompt. E.g.: ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```
        """
        self.text = text
        super().__init__()

    def set_text_tokenized(self, tokenizer: Any) -> None:
        """Set Tokenized Text.

        Args:
            tokenizer (Any): Tokenizer for preparing the dataset and using the mask token.
        """
        self.text_tokenized = tokenizer(
            self.text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].squeeze()

    def __str__(self) -> str:
        """Return String Representation."""
        return self.text


class Key:
    """Class for a Dataset-Key within the Prompt Object."""

    def __init__(self, key: str = "text", truncate: bool = False):
        """Initialize Key Object.

        Args:
            key (str): Key used to access the text in the dataset. E.g.: ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```
        """
        self.key = key

    def __str__(self) -> str:
        """Return String Representation."""
        return f"<Data-Key: '{self.key}'>"


class Verbalizer(Tokenizable):
    """Class for Verbalizer within the Prompt Object."""

    def __init__(self, verbalizer: List[List[str]]):
        """Initialize Text Object.

        Args:
            verbalizer (List[List[str]]): Verbalizer used in the prompt. E.g.: ``Prompt(Key('text'), Text('It was'), Verbalizer([['good'], ['bad']]))```.
                It is important to select appropriate label words.
        """
        self.verbalizer: List[List[str]] = verbalizer
        super().__init__()

    def set_mask_token(self, mask_token: str) -> None:
        """Set Mask Token.

        Args:
            mask_token (str): Mask Token for the model.
        """
        self.text_tokenized = tensor([mask_token])

    def __str__(self) -> str:
        """Return String Representation."""
        return f"<Verbalizer: [{', '.join([f'[\"{str(elem[0])}\",...]' for elem in self.verbalizer])}]>"


class Prompt:
    """Prompt Class."""

    def __init__(self, *args: Any, truncate_data: bool = True, sep: str = " "):
        """Initialize Prompt.

        Args:
            *args: Key, Text and Verbalizer Objects passed as arguments for this class.
                Only one verbalizer per prompt and arbitrary many texts and keys possible. In case of a causal model,
                the Verbalizer must be at the end of the arguments.
            sep (str): Seperator for combining text and data,
                i.e. `Prompt(Key('text'), Text('It was ...') ..., sep='#)` -> `'lorem ipsum...#It was ...'`
        """
        self.sep: str = sep
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.truncate_data: bool = truncate_data
        self.prompt: List[Union[Key, Text, Verbalizer]] = list(args) if len(args) > 1 else [args[0]]  # type: ignore[arg-type,list-item]
        self.intermediate_token: Optional[str] = None

        assert len(
            [
                e
                for e in self.prompt
                if isinstance(e, Verbalizer)
                or isinstance(e, Key)
                or isinstance(e, Text)
            ]
        ) == len(
            self.prompt
        ), "Only Key, Text and Verbalizer objects are allowed in Prompt."

        verb_filtered: List[Verbalizer] = [
            (i, e) for i, e in enumerate(self.prompt) if isinstance(e, Verbalizer)
        ]

        self.key_list: List[str] = [e.key for e in self.prompt if isinstance(e, Key)]

        assert (
            len(verb_filtered) == 1
        ), "The prompt must contain the verbalizer. E.g. `Prompt(Key('text'), Text('It was '), Verbalizer([['good'], ['bad']]))`."

        self.verbalizer: Verbalizer = verb_filtered[0][1]
        self.idx: int = verb_filtered[0][0]
        self.before_verb: Union[Text, Key] = self.prompt[self.idx - 1]

    def subinit(self, tokenizer: Any, generate: bool, multi_token: bool = False) -> None:
        """Subinitialization for Main Class.

        Second initializatio that happens hidden in the main class.

        Args:
            tokenizer (Any): Tokenizer for preparing the dataset and using the mask token.
            generate (bool): Flag for using causal language models, set to False when using MLM-based models.
        """
        self.tokenizer = tokenizer
        self.generate = generate

        if generate:
            assert isinstance(
                self.prompt[-1], Verbalizer
            ), "When using `CausalLM4Classification`, the last token must be of type `Verbalizer`."

        for e in [e for e in self.prompt if isinstance(e, Text)]:
            e.set_text_tokenized(self.tokenizer)

        self.used_tokens: int = (
            sum(
                [
                    (
                        e.text_tokenized.shape[0]  # type: ignore[attr-defined]
                        if len(e.text_tokenized.size()) != 0  # type: ignore[attr-defined]
                        else 0
                    )
                    for e in self.prompt
                    if isinstance(e, Text)
                ]
            )
            - 1
        )  # -1 for mask/last token
        if not self.generate:
            self.verbalizer.set_mask_token(self.tokenizer.mask_token_id)
        if self.generate:
            if self.idx != 0:
                if self.sep != "":
                    self.intermediate_token = self.sep
                elif isinstance(self.before_verb, Text):
                    self.intermediate_token = self.before_verb.text[-1]
                else:
                    # TODO: Warning!
                    pass

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
        else:
            return self.tokenizer.mask_token if not self.generate else ""

    def get_text(self, data: Dataset) -> str:
        """Join Prompt to String.

        Join prompt to string with `sep` seperator.

        Args:
            data (Dataset): Data for which the key(s) will be used to index the column(s).

        Returns:
            str: Prompt in natural language.
        """
        return self.sep.join([self.decide(e, data) for e in self.prompt])

    def __str__(self) -> str:
        """Return String Representation."""
        return " ".join([str(e) for e in self.prompt])

    def __repr__(self) -> str:
        """Repr Representation."""
        return self.__str__()
