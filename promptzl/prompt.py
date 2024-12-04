from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Prompt:
    """Base Function for Prompt.

    Provides magic __add__ method to combine prompt parts, __fn_str__ method
    for the functional representation in the also included method _prompt_fun.
    _prompt_fun returns a function that can be used to build the final prompt
    before the tokenization.
    """

    def __init__(self, collector: list):
        """Base Class for User-Facing Prompt.

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
        if isinstance(other, FVP):
            raise ValueError("FVP cannot be added to a prompt.")
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

    def _prompt_fun(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Union[Callable[[Tuple[str]], str], Callable[[Dict[str, str]], str]]:
        """Return a function that can be used to build the prompt.

        The function Return a string formatting function '%s' that can be
        used to build the prompt. Each '%s' corresponds to a key in the dataset.
        """
        return lambda e: self.__fn_str__(tokenizer) % e

    def _get_verbalizer(self) -> "Vbz":
        verb_filter: List[Vbz] = [e for e in self.collector if isinstance(e, Vbz)]
        if len(verb_filter) != 1:
            raise ValueError(
                f"Prompt must entail one verbalizer!\n\t'-> {str(self.__str__())}"
            )
        else:
            return verb_filter[0]

    def _check_valid_keys(self) -> None:
        if len([e for e in self.collector if isinstance(e, (Key, Img))]) < 1:
            raise ValueError(
                f"No key found in prompt. Please provide a `Key` key!\n\t'-> {str(self.__str__())}"
            )


class Txt(Prompt):
    """Object for Text Representation."""

    def __init__(self, text: str = " "):
        """**Text Representation in Prompt**

        This class allows including custom text in the prompt.
        I.e. :code:`Txt("Hello ") + Vbz([['World'], ['Mars]]) + Txt('!')` will prepend "Hello World!"
        and append "!" to the prompt.

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


class Key(Prompt):
    """Placeholder (Object) for Key Representation."""

    def __init__(self, key: str = "text"):
        """**Data-Placeholder to Corresponding Key**

        This class allows including a key in the prompt that will be replaced by the
        corresponding value in the dataset. I.e. :code:`Key("text") + Txt(" is ") + Vbz([['good', 'bad']])`
        will replace the key "text" with the corresponding value in the dataset and append " is " and the verbalizer.

        If the dataset consists of :code:`{"text": ["Restaurant X", "Restaurant Y"]}`, the text the model will
        see is :code:`"Restaurant X is "` and :code:`"Restaurant Y is "`.

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


class Img(Prompt):
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

    def __init__(
        self,
        verbalizer: Optional[Dict[Union[int, str], List[str]]],
    ):
        """**Verbalizer Representation in Prompt**

        The verbalizer object is the most crucial part of this approach as it defines where the masked token
        is located in the MLM setting and defines the words where the tokens in the vocabulary are extracted.
        Hence a valid prompt must include one verbalizer. For causal models, the verbalizer **must** be at the end
        of the prompt while the verbalizer can be at **any position** in the prompt when using masked models.

        The corresponding words for each class (can be multiple), must be provided in the form of a list of lists
        or a dictionary where the key is the class label (ideally refering to the representation in the dataset)
        and the value is a list of words corresponding to the semantic of the class.

        Valid verbalizers:

        .. code-block:: python

            Key() + Txt("Is this good?") + Vbz([["good", "bad"], ["ugly"]])
            Key() + Txt("Is this good?") + Vbz({0: ["good", "bad"], 1: ["ugly"]})

        The above examples are valid for causal and masked models. The following example is only valid for masked models:

        .. code-block:: python

            Key("headline") + Txt("[Category:]")  + Vbz([["Politics", "Nature", "Technology"], ["ugly"]]) + Txt("]") + Key("body")


        Args:
            verbalizer (Optional[Dict[Union[int, str], List[str]]]): List of verbalizers.
        """
        self.verbalizer_dict: Optional[Dict[Union[int, str], List[str]]] = None
        if isinstance(verbalizer, dict):
            self.verbalizer: List[List[str]] = [val for val in verbalizer.values()]
            self.verbalizer_dict = verbalizer
        elif isinstance(verbalizer, list):
            assert all(isinstance(val, list) for val in verbalizer) and all(
                isinstance(val, str) for labels in verbalizer for val in labels
            ), "Verbalizer must be a list of lists of strings or a dictionary."
            self.verbalizer = verbalizer
        else:
            raise ValueError("Verbalizer must be a list of lists or a dictionary.")
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
        if tokenizer.mask_token is None:
            return ""
        else:
            return f"{tokenizer.mask_token}"

    def __repr__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return self.__str__()


class FVP(Prompt):
    """Function Verbalizer Pair (FVP) Class."""

    def __init__(
        self, _prompt_function: Callable[[Dict[str, str]], str], verbalizer: Vbz
    ):
        """Initialize Class.

        Args:
            _prompt_function (Callable[[Dict[str, str]], str]): Function to build prompt.
            verbalizer (Vbz): Verbalizer.
        """
        self.fvp_fn = _prompt_function
        super().__init__([self, verbalizer])

    def __add__(self, *args: Any) -> Prompt:
        """Add Prompt Parts (Not Supported for FVP).

        Args:
            *args: Arguments (not required).

        Raises:
            ValueError: FVP cannot be added to a prompt.
        """
        raise ValueError("FVP cannot be added to a prompt.")

    def __str__(self) -> str:
        """Represent Object as String."""
        return "<FVP>"

    def __fn_str__(self, *args: Any) -> str:
        """Return String Representation for Prompt-Building-Function.

        Args:
            *args: Arguments (not required).

        Raises:
            NotImplementedError: `__fn_str__` not implemented for FVP.
        """
        raise NotImplementedError("`__fn_str__` not implemented for FVP.")

    def __repr__(self) -> str:
        """Represent Object as String."""
        return self.__str__()

    def _prompt_fun(self, *args: Any) -> Callable[[Dict[str, str]], str]:
        """Return a function that can be used to build the prompt.

        The function returns a prompt generating function in which the arguments
        must be a dict with keys corresponding to the keys in the data.

        Returns:
            Callable[[Dict[str, str]], str]: Prompt generating function.
        """
        return self.fvp_fn

    def _check_valid_keys(self) -> None:
        pass
