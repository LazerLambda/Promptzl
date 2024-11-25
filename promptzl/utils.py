"""Utils for Prompt Generation.

Promptzl, 2024

MIT LICENSE
"""

import operator
from functools import reduce
from typing import Any, Dict, List, Tuple, Union
from warnings import warn

from datasets import Dataset
from torch import tensor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .prompt import Img, Key, Prompt, Txt, Vbz


class SystemPrompt:
    """Class for Internal Prmopt Handling."""

    def __init__(
        self,
        prompt: Prompt,
        tokenizer: PreTrainedTokenizerFast,
        mlm: bool = True,
    ):
        """Initialize Class.

        Initialize and check if prompt is valid.

        Args:
            prompt (Prompt): Prompt to be used.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to be used.
            mlm (bool, optional): Whether to use MLM. Defaults to True.

        Raises:
            ValueError: If prompt does not include a verbalizer.
            ValueError: If prompt does not include a key.
            ValueError: If tokenizer does not have a mask token.
            AssertionError: If prompt is not of type Prompt.
            AssertionError: If tokenizer is not of type PreTrainedTokenizerFast.
            AssertionError: If mlm is not of type bool.
        """
        self.prompt: Prompt = prompt
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.mlm: bool = mlm

        assert isinstance(prompt, Prompt), "`prompt` must be of type Prompt."
        assert isinstance(
            tokenizer, PreTrainedTokenizerFast
        ), "`tokenizer` must be of type PreTrainedTokenizer."
        assert isinstance(mlm, bool), "`mlm` must be of type bool."

        # TODO: Check if prompt includes key
        # TODO: Check if vbz is at the end for causal

        verb_filter: List[Vbz] = [
            e for e in self.prompt.collector if isinstance(e, Vbz)
        ]
        if len(verb_filter) != 1:
            raise ValueError(f"No verbalizer found in prompt:\n\t'-> {str(prompt)}")
        else:
            self.verbalizer: Vbz = verb_filter[0]

        if len([e for e in self.prompt.collector if isinstance(e, (Key, Img))]) < 1:
            raise ValueError(
                f"No key found in prompt. Please provide a `Key` key!\n\t'-> {str(prompt)}"
            )

        self.intermediate_token = None
        if mlm:
            if self.tokenizer.mask_token_id is None or not hasattr(
                self.tokenizer, "mask_token_id"
            ):
                raise ValueError(
                    "Tokenizer does not have a mask token. Please provide a tokenizer with a mask token."
                )

        # TODO: TEST THIS and check if subsequent token is also important.
        for i in range(0, len(self.prompt.collector) - 1):
            if isinstance(self.prompt.collector[i], (Txt)) and isinstance(
                self.prompt.collector[i + 1], Vbz
            ):
                self.intermediate_token = self.prompt.collector[i].text[-1]

        prefix_suffix: Tuple[List[int], List[int]] = self.get_prefix_suffix()
        self.prefix: List[int] = prefix_suffix[0]
        self.suffix: List[int] = prefix_suffix[1]
        n_pre_suffix: int = 0

        if len(self.prefix) > 0:
            n_pre_suffix += 1

        if len(self.suffix) > 0:
            n_pre_suffix += 1

        self.dict_ds: Dict[str, Dataset] = {
            k: None
            for k in [e.key for e in self.prompt.collector if isinstance(e, (Img, Key))]
        }

        self.template_prmpt: List[Union[List[int], Key, Img, Vbz]] = [
            self.tokenizer(e.text, add_special_tokens=False)["input_ids"]
            if isinstance(e, Txt)
            else e
            for e in self.prompt.collector
        ]

        only_keys: int = len(
            [None for e in self.prompt.collector if isinstance(e, (Img, Key))]
        )
        used_tokens: int = int(
            self.tokenizer.model_max_length
            - sum([len(e) for e in self.template_prmpt if isinstance(e, list)])
            - n_pre_suffix
        )  # Account for prefix and suffix tokens
        if mlm:
            self.max_len_keys: int = (
                int((used_tokens - 1) // only_keys) if only_keys != 0 else used_tokens
            )
        else:
            self.max_len_keys = (
                int(used_tokens // only_keys) if only_keys != 0 else used_tokens
            )

        self.prmpt_f = self.prompt.prompt_fun(self.tokenizer)
        self.key_list = [
            e.key for e in self.prompt.collector if isinstance(e, (Img, Key))
        ]

    def get_prefix_suffix(self) -> Tuple[List[int], List[int]]:
        """Get Prefix and Suffix Tokens.

        Get the prefix and suffix tokens for the tokenizer.

        Returns:
            Tuple[List[int], List[int]]: Prefix and Suffix tokens.
        """
        ex = list(
            set(self.tokenizer.vocab.keys()) - set(self.tokenizer.all_special_tokens)
        )[0]
        example = self.tokenizer(ex)["input_ids"]

        prefix = []
        suffix = []

        if example[0] in self.tokenizer.all_special_ids:
            prefix = [example[0]]
        if example[-1] in self.tokenizer.all_special_ids:
            suffix = [example[-1]]

        return prefix, suffix

    def prepare_and_tokenize_dataset(
        self, data: Dict[str, List[Union[str, Any]]]
    ) -> List[List[int]]:
        """Prepare and Tokenize Dataset.

        Prepare and tokenize the dataset for the model. Tokenize first and then concatenate the tokens.
        This technique is less reliable than tokenizing the prepared sequence. Results might vary. However,
        it is more safe as it does not exceed the context length of the model.

        Args:
            data (Dict[str, List[Union[str, Any]]]): Data to be tokenized.

        Returns:
            List[List[int]]: Tokenized Data.
        """
        n: int = len(data[list(data.keys())[0]])

        # Prepare prompts in batch
        tknzd_prmpt: List[Union[List[List[int]], Key, Img, Vbz]] = [
            [e] * n if not isinstance(e, (Vbz, Img, Key)) else e
            for e in self.template_prmpt
        ]

        # Tokenize data by key # TODO: max lengths
        tknzd_prmpt = [
            self.tokenizer(
                data[e.key],
                add_special_tokens=False,
                max_length=self.max_len_keys,
                truncation=True,
            )["input_ids"]
            if isinstance(e, (Key, Img))
            else e
            for e in tknzd_prmpt
        ]

        # Add Mask if necessary TODO: different case for autoregressive decoding
        if self.mlm:
            tknzd_prmpt = [
                [[self.tokenizer.mask_token_id]] * n if isinstance(e, Vbz) else e
                for e in tknzd_prmpt
            ]
        else:
            tknzd_prmpt = [e for e in tknzd_prmpt if not isinstance(e, Vbz)]

        # Add Prefix and Suffix
        tknzd_prmpt = [[self.prefix] * n] + tknzd_prmpt + [[self.suffix] * n]

        # Remove superfluous tokens
        tknzd_prmpt = [e for e in tknzd_prmpt if len(e) > 0]  # type: ignore[arg-type]

        # Add lists together horizontically
        tknzd_prmpt = [reduce(operator.add, e) for e in zip(*tknzd_prmpt)]

        return tknzd_prmpt  # type: ignore[return-value]

    def pad_and_stack(self, data: List[List[int]]) -> Dict[str, tensor]:
        """Pad and Stack Data.

        Pad data and create the tensor for inference.

        Args:
            data (List[List[int]]): Tokenized Data.

        Returns:
            Dict[str, tensor]: Padded and Stacked Data.
        """
        n: int = max([len(e) for e in data])
        if self.mlm:
            input_ids: tensor = tensor(
                [e + [self.tokenizer.pad_token_id] * (n - len(e)) for e in data]
            )
            attention_mask: tensor = tensor(
                [[1] * len(e) + [0] * (n - len(e)) for e in data]
            )
        else:
            input_ids = tensor(
                [[self.tokenizer.pad_token_id] * (n - len(e)) + e for e in data]
            )
            attention_mask = tensor([[0] * (n - len(e)) + [1] * len(e) for e in data])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_tensors(self, data: Dict[str, List[Union[str, Any]]]) -> Dict[str, tensor]:
        """Get Tensors.

        Wrapper for `prepare_and_tokenize_dataset` and `pad_and_stack`.

        Args:
            data (Dict[str, List[Union[str, Any]]]): Data to be tokenized.

        Returns:
            Dict[str, tensor]: Padded and Stacked Data.
        """
        tknzd_data: List[List[int]] = self.prepare_and_tokenize_dataset(data)
        return self.pad_and_stack(tknzd_data)

    def get_tensors_fast(
        self, data: Dict[str, List[Union[str, Any]]]
    ) -> Dict[str, tensor]:
        """Get Tensors Fast.

        The prompts' function for buildling the prompt is used to prepare the data. The data is then tokenized
        and returned as a list of tokenized data. If the tokenized data is longer than the model's maximum length,
        a safe but less precised approach is used (see `prepare_and_tokenize_dataset`).

        Args:
            data (Dict[str, List[Union[str, Any]]]): Data to be tokenized.

        Returns:
            List[List[int]]: Tokenized Data.
        """
        prepared_data: Union[List[str], Dict[str, tensor]] = [
            self.prmpt_f(tuple([elem for elem in e]))
            for e in zip(*[data[val] for val in self.key_list])
        ]
        prepared_data = self.tokenizer(
            prepared_data, padding="longest", return_tensors="pt"
        )
        if prepared_data["input_ids"].shape[1] > self.tokenizer.model_max_length:  # type: ignore[call-overload]
            warn(
                "Data is longer than model's maximum length. Truncating data, this may lead to inaccurate results.",
                category=UserWarning,
            )
            # TODO: Remove truncation option
            # TODO: Tidy up this
            return self.get_tensors(data)
        else:
            return prepared_data  # type: ignore[return-value]

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return str(self.prompt)
