"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, List, Optional, Tuple, Union
from warnings import warn

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
        )["input_ids"][0]

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
        # TODO: Test these cases!
        assert (
            len([e for e in verbalizer if not isinstance(e, list)]) == 0
        ), "Verbalizer must be of type List[List[str]]."
        assert (
            len(
                [e for sublist in verbalizer for e in sublist if not isinstance(e, str)]
            )
            == 0
        ), "Verbalizer must be of type List[List[str]]."
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
        return (
            "<Verbalizer: ["
            + ", ".join(['["%s",...]' % str(elem[0]) for elem in self.verbalizer])
            + "]>"
        )


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

        verb_filtered: List[Tuple[int, Verbalizer]] = [
            (i, e) for i, e in enumerate(self.prompt) if isinstance(e, Verbalizer)
        ]

        self.key_list: List[str] = [e.key for e in self.prompt if isinstance(e, Key)]

        assert (
            len(verb_filtered) == 1
        ), "The prompt must contain the verbalizer. E.g. `Prompt(Key('text'), Text('It was '), Verbalizer([['good'], ['bad']]))`."

        self.verbalizer: Verbalizer = verb_filtered[0][1]
        self.idx: int = verb_filtered[0][0]
        self.before_verb: Union[Text, Key, Verbalizer] = self.prompt[self.idx - 1]

    def subinit(self, tokenizer: Any, generate: bool) -> None:
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
                    # TODO: Test this case!
                    warn(
                        "Data is used directly before the verbalizer. Without a seperator, the verbalizer can not be enhanced automatically."
                    )
            # TODO: Test Prompt(Verbalizer([[...], [...]]), Key('text'), Text('...')) /Prompt(Verbalizer([[...], [...]]))

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
        return "".join([str(e) for e in self.prompt])

    def __repr__(self) -> str:
        """Repr Representation."""
        return self.__str__()


def get_prompt(
    prompt: str, key_list: Union[List[str], str], verbalizer: List[List[str]]
) -> Prompt:
    """Get Prompt Ready for Instantiation.

    Prepare Prompt class by using a template string with %s and %m placeholders.
    %s refers to the dataset key, %m refers to the mask. The respective dataset keys
    are passed in the argument `key_list`.

    Args:
        prompt (str): Template prompt with c-string-style placeholders (%s: dataset key, %m: mask/verbalizer).
        key_list (Union[List[str], str]): List of column-keys for the used dataset.
        verbalizer (List[List[str]]): Verbalizer list.

    Raises:
        TypeError: If `key_list` is not of type Union[List[str], str].
        AssertionError: If the number of keys in `key_list` does not match the number of %s placeholders in `prompt`.

    Returns:
        str: The instantiated prompt with placeholders replaced by the corresponding keys and masks.
    """
    n: int = 0
    if isinstance(key_list, list):
        n = len(key_list)
    elif isinstance(key_list, str):
        n = 1
    else:
        raise TypeError("Argument `key_list` must be of type Union[List[str], str].")
    prompt_splitted: List[str] = prompt.split("%s")
    assert (
        n == len(prompt_splitted) - 1
    ), "Number of keys in `key_list` does not match %s-placeholders"
    collector: List[Union[Tuple[str, Union[Key, Text]], Text, Key, Verbalizer]] = []

    for i in range(len(prompt_splitted)):
        if len(prompt_splitted[i]) > 0:
            collector.append((prompt_splitted[i], Text(prompt_splitted[i])))
        if i < len(key_list):
            if len(key_list[i]) > 0:
                collector.append((key_list[i], Key(key_list[i])))
            else:
                raise ValueError(
                    f"`key_list` must not contain empty elements!\nkey_list: '{key_list}'"
                )
    mask_loc: List[Tuple[int, List[str]]] = [(i, e[0].split("%m")) for i, e in enumerate(collector) if "%m" in e[0]]  # type: ignore[index]
    if len(mask_loc) > 0:
        assert (
            len(mask_loc) == 1 and len(mask_loc[0][1]) == 2
        ), "Only one %m-placeholder must be provided."
        idx: int = mask_loc[0][0]
        mask_splitted: List[str] = mask_loc[0][1]
        collector = (
            collector[0:idx]
            + [Text(mask_splitted[0]), Verbalizer(verbalizer), Text(mask_splitted[1])]
            + collector[(idx + 1) : :]
        )
    else:
        collector.append(Verbalizer(verbalizer))
    collector = [e[1] if isinstance(e, tuple) else e for e in collector]
    return Prompt(*collector, sep="")
