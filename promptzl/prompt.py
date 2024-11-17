"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, Callable, List, Tuple

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Prompt:
    """Base Function for Prompt.

    Provides magic __add__ method to combine prompt parts.
    """

    def __init__(self, collector: list):
        """Initialiize Class.

        Args:
            collector (list): List of prompt parts.
        """
        self.collector = collector

    def __add__(self, other: "Prompt") -> "Prompt":
        """Add prompt parts.

        Args:
            other (Prompt): Another prompt part.

        Returns:
            Prompt: Combined prompt.
        """
        self.collector.append(other)
        return Prompt(self.collector)

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return "".join([str(e) for e in self.collector])

    def __fn_str__(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Return String Representation for Prompt-Building-Function.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer to mask.
        """
        return "".join([e.__fn_str__(tokenizer) for e in self.collector])

    def __repr__(self) -> str:
        """Representation of prompt.

        Returns:
            str: Representation of prompt
        """
        return self.__str__()

    def prompt_fun(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Callable[[Tuple[str]], str]:
        """Return a function that can be used to build the prompt.

        The function Return a string formatting function '%s' that can be
        used to build the prompt. Each '%s' corresponds to a key in the dataset.
        """
        return lambda e: self.__fn_str__(tokenizer) % e


class Txt(Prompt):
    """Object for Text Representation."""

    def __init__(self, text: str = " "):
        """Initialize Class.

        Args:
            text (str, optional): Text. Defaults to " ".
        """
        self.text: str = text
        super().__init__([self])

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return self.text

    def __fn_str__(self, *args: Any) -> str:
        """Return String Representation for Prompt-Building-Function.

        Args:
            *args: Arguments (not required).
        """
        return self.text

    def __repr__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return self.__str__()


class TKy(Prompt):
    """Placeholder (Object) for Key Representation."""

    def __init__(self, key: str = "text"):
        """Initialize Class.

        Args:
            key (str, optional): Key. Defaults to "text".
        """
        self.key: str = key
        super().__init__([self])

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return f"<{self.key}>"

    def __fn_str__(self, *args: Any) -> str:
        """Return String Representation for Prompt-Building-Function.

        Args:
            *args: Arguments (not required).
        """
        return "%s"

    def __repr__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return self.__str__()


class IKy(Prompt):
    """Placeholder (Object) for Image Key Representation."""

    def __init__(self, key: str):
        """Initialize Class.

        Args:
            key (str): Key.
        """
        self.key: str = key
        super().__init__([self])

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return f"[{self.key}]"

    def __fn_str__(self, *args: Any) -> str:
        """Return String Representation for Prompt-Building-Function.

        Args:
            *args: Arguments (not required).
        """
        return "[IMAGE]"

    def __repr__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return self.__str__()


class Vbz(Prompt):
    """Object for Verbalizer Representation."""

    def __init__(self, verbalizer: List[List[str]]):
        """Initialize Class.

        Args:
            verbalizer (List[List[str]]): List of verbalizers.
        """
        self.verbalizer: List[List[str]] = verbalizer
        super().__init__([self])

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return (
            "<Vbz: ["
            + ", ".join(['["%s",...]' % str(elem[0]) for elem in self.verbalizer])
            + "]>"
        )

    def __fn_str__(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Return String Representation for Prompt-Building-Function.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer to mask.
        """
        return f"{tokenizer.mask_token}"

    def __repr__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return self.__str__()
