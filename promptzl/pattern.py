"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, Dict, List

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Placeholder:
    """Placeholder class.

    Base-class as a placeholder for the prompt pattern. Different placeholder are necessary, i.e.
    data keys, prompts, masks, etc.
    """

    def __init__(self, text: str):
        """Initialize Placeholder.

        :param text: The text of the placeholder.
        """
        self.text: str = text

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> List[int]:
        """Tokenize Function.

        Tokenize the placeholder text using the provided tokenizer. Necessary for prompt and mask.

        :param tokenizer: The tokenizer to be used.
        :return: The tokenized placeholder text.
        """
        return tokenizer(self.text)["input_ids"]


class DataKey(Placeholder):
    """DataKey class.

    Placeholder for the data keys in the prompt pattern.
    """

    def __init__(self, text: str):
        """Initialize DataKey.

        :param text: The text of the data key.
        """
        super().__init__(text)

    def tokenize(self, *args, **kwargs) -> List[int]:
        """Tokenize Function.

        In the case of a data key, the tokenization is not necessary, as the data key is not part of the prompt.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Empty list.
        """
        return []


class Prompt(Placeholder):
    """Prompt class.

    Placeholder for the prompt in the prompt pattern.
    """

    def __init__(self, text: str, autospaces: bool = True):
        """Initialize Prompt.

        :param text: The text of the prompt.
        :param autospaces: Whether to add spaces at the beginning and end of the prompt.
        """
        if autospaces:
            if text[-1] != " ":
                text = text + " "
            if text[0] != " ":
                text = " " + text
        super().__init__(text)


class Mask(Placeholder):
    """Mask Class.

    Placeholder for the mask token in the prompt pattern.
    """

    def __init__(self):
        """Initialize Mask."""
        super().__init__("MASK")

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> List[int]:
        """Tokenize Function.

        Return the token id of the mask token.

        :param tokenizer: The tokenizer to be used.
        :return: The token id of the mask token.
        """
        if tokenizer.mask_token_id is not None:
            return [tokenizer.mask_token_id]
        else:
            return []


class Pattern:
    """Pattern Class.

    Class to define the pattern of the prompt. The pattern is a list of placeholders.
    """

    def __init__(self, pattern: List[Placeholder]):
        """Initialize Pattern.

        :param pattern: The pattern of the prompt.
        """
        assert (
            len(list(filter(lambda x: isinstance(x, Mask), pattern))) <= 1
        ), "Exactly one mask token must be present in the pattern!"
        self.pattern: List[Placeholder] = pattern
        self.prompt_len: int = -1
        self.prompt_tokenized: Any = []

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> List[List[int]]:
        """Tokenize Function.

        :param tokenizer: The tokenizer to be used.
        :return: The tokenized pattern.
        """
        self.prompt_tokenized = [e.tokenize(tokenizer) for e in self.pattern]
        self.prompt_len = sum(map(len, self.prompt_tokenized))
        return self.prompt_tokenized

    def get_prompt_single(self, elem: Dict[str, Any]):
        """Get Prompt for a Single Instance.

        Transform input data into prompt format.

        :param elem: The input data.
        :return: The prompt for the input data.
        """
        return [
            item
            for sublist in list(
                map(
                    lambda e: elem[e[0].text] if isinstance(e[0], DataKey) else e[1],
                    zip(self.pattern, self.prompt_tokenized),
                )
            )
            for item in sublist
        ]

    def get_prompt_batch(self, batch: Any) -> List[List[List]]:
        """Transform Batch into Desired Prompt-Format.

        :param batch: The input batch.
        :return: The prompt for the input batch.
        """
        return [self.get_prompt_single(e) for e in batch]
