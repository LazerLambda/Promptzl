import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
import torch
from datasets import Dataset
from torch import Tensor, tensor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .prompt import FVP, Img, Key, Prompt, Txt, Vbz


class SystemPrompt:
    """Class for Internal Prompt Handling."""

    def __init__(
        self,
        prompt: Prompt,
        tokenizer: PreTrainedTokenizerFast,
        generate: bool = True,
    ):
        """Initialize Class.

        Initialize and check if prompt is valid.

        Args:
            prompt (Prompt): Prompt to be used.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to be used.
            generate (bool, optional): Whether to use a causal LM setup. Defaults to True.

        Raises:
            ValueError: If prompt does not include a verbalizer.
            ValueError: If prompt does not include a key.
            ValueError: If tokenizer does not have a mask token.
            AssertionError: If prompt is not of type Prompt.
            AssertionError: If tokenizer is not of type PreTrainedTokenizerFast.
            AssertionError: If generate is not of type bool.
        """
        assert isinstance(prompt, Prompt), "`prompt` must be of type Prompt."
        assert isinstance(
            tokenizer, PreTrainedTokenizerFast
        ), "`tokenizer` must be of type PreTrainedTokenizer."
        assert isinstance(generate, bool), "`generate` must be of type bool."

        self.prompt: Prompt = prompt
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.generate: bool = generate

        if self.generate:
            assert isinstance(prompt.collector[-1], Vbz), "No Verbalizer found at the end of the sequence!"

        self.fvp: bool = False
        # Check for FVP because it does not support truncation
        if isinstance(prompt, FVP):
            self.fvp = True

        self.verbalizer: Vbz = self.prompt._get_verbalizer()

        self.prompt._check_valid_keys()

        if not self.generate:
            if self.tokenizer.mask_token_id is None or not hasattr(
                self.tokenizer, "mask_token_id"
            ):
                raise ValueError(
                    "Tokenizer does not have a mask token. Please provide a tokenizer with a mask token."
                )

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
        if generate:
            self.max_len_keys: int = (
                int(used_tokens // only_keys) if only_keys != 0 else used_tokens
            )
        else:
            self.max_len_keys = (
                int((used_tokens - 1) // only_keys)
                if only_keys != 0
                else used_tokens  # Subtract one for mask token for prediction
            )
        
        if self.max_len_keys < 1:
            raise ValueError(
                (f"No tokens left for data. Please reduce the length of the prompt."
                f"Max. length: {self.tokenizer.model_max_length}, beyond max. length: {(-1) * self.max_len_keys}")
            )

        self.prmpt_f: Callable[[Any], str] = self.prompt._prompt_fun(self.tokenizer)
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

        # Tokenize data by key
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

        # Add Mask if necessary
        if self.generate:
            tknzd_prmpt = [e for e in tknzd_prmpt if not isinstance(e, Vbz)]
        else:
            tknzd_prmpt = [
                [[self.tokenizer.mask_token_id]] * n if isinstance(e, Vbz) else e
                for e in tknzd_prmpt
            ]

        # Add Prefix and Suffix
        tknzd_prmpt = [[self.prefix] * n] + tknzd_prmpt + [[self.suffix] * n]

        # Remove superfluous tokens
        tknzd_prmpt = [e for e in tknzd_prmpt if len(e) > 0]  # type: ignore[arg-type]

        # Add lists together horizontically
        tknzd_prmpt = [reduce(operator.add, e) for e in zip(*tknzd_prmpt)]

        return tknzd_prmpt  # type: ignore[return-value]

    def pad_and_stack(self, data: List[List[int]]) -> Dict[str, Tensor]:
        """Pad and Stack Data.

        Pad data and create the tensor for inference.

        Args:
            data (List[List[int]]): Tokenized Data.

        Returns:
            Dict[str, Tensor]: Padded and Stacked Data.
        """
        n: int = max([len(e) for e in data])
        if self.generate:
            input_ids: Tensor = tensor(
                [[self.tokenizer.pad_token_id] * (n - len(e)) + e for e in data]
            )
            attention_mask: Tensor = tensor(
                [[0] * (n - len(e)) + [1] * len(e) for e in data]
            )
        else:
            input_ids = tensor(
                [e + [self.tokenizer.pad_token_id] * (n - len(e)) for e in data]
            )
            attention_mask = tensor([[1] * len(e) + [0] * (n - len(e)) for e in data])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_tensors(self, data: Dict[str, List[Union[str, Any]]]) -> Dict[str, Tensor]:
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
    ) -> Dict[str, Tensor]:
        """Get Tensors Fast.

        The prompts' function for buildling the prompt is used to prepare the data. The data is then tokenized
        and returned as a list of tokenized data. If the tokenized data is longer than the model's maximum length,
        a safe but less precised approach is used (see `prepare_and_tokenize_dataset`).

        Args:
            data (Dict[str, List[Union[str, Any]]]): Data to be tokenized.

        Returns:
            List[List[int]]: Tokenized Data.
        """
        if self.fvp:
            prepared_data: Union[List[str], Dict[str, tensor]] = [
                self.prmpt_f(dict(zip(data.keys(), e)))
                for e in list(zip(*data.values()))
            ]
        else:
            prepared_data = [
                self.prmpt_f(tuple([elem for elem in e]))
                for e in zip(*[data[val] for val in self.key_list])
            ]
        prepared_data = self.tokenizer(
            prepared_data, padding="longest", return_tensors="pt"
        )
        if prepared_data["input_ids"].shape[1] > self.tokenizer.model_max_length:  # type: ignore[call-overload]
            if self.fvp:
                raise ValueError(
                    (
                        "Model input longer than maximum length!\n\t'->FVP does not support truncation."
                        "Please shorten the text beforehand or use `Txt` and `Key` classes instead."
                    )
                )
            warn(
                "Data is longer than model's maximum length. Truncating data, this may lead to inaccurate results.",
                category=UserWarning,
            )
            return self.get_tensors(data)
        else:
            return prepared_data  # type: ignore[return-value]

    def __str__(self) -> str:
        """Represent Object as String.

        Returns:
            str: String representation of prompt.
        """
        return str(self.prompt)


@dataclass
class LLM4ClassificationOutput:
    """Class for Organizing Output of LLM4Classification.

    Attributes:
        predictions (Optional[Any]): Predictions (i.e. predicted label for each instance).
        distribution (Optional[Any]): Distribution of predictions (i.e. probabilities for each label).
    """

    predictions: Optional[
        Union[Tensor, pd.DataFrame, pl.DataFrame, List[Union[int, str]], np.ndarray]
    ] = None
    distribution: Optional[
        Union[Tensor, pd.DataFrame, pl.DataFrame, List[List[float]], np.ndarray]
    ] = None
    logits: Optional[
        Union[Tensor, pd.DataFrame, pl.DataFrame, List[List[float]], np.ndarray]
    ] = None


def calibrate(probs: Tensor) -> Tensor:
    """**Calibrates Probabilities**

    Addresses the calibartion issue (`Zhao et al., 2021 <https://arxiv.org/abs/2102.09690>`_,
    `Hu et al., 2022 <https://aclanthology.org/2022.acl-long.158>`_). In MLM-based models, some tokens
    are more likely to be generated than others. This can lead to biased probabilities. To address this,
    the probabilities are calibrated in the following way.

    The mean of the probabilities for each class is computed and then used as the divisor for the
    probabilities of each class. The probabilities are then normalized by the sum of the probabilities
    to form a valid probability distribution.

    This technique can lead to improved performance in some cases.

    .. note::
        The calibration is always performed on the given probabilities, which works well in the case of many examples.
        For consecutevly calibration, it is may be helpfull to store the mean of the probabilities and use it as the divisor.

    Args:
        probs (tensor): The probabilities to be calibrated.

    Returns:
        tensor: The calibrated probabilities.
    """
    probs = probs / (torch.mean(probs, dim=0) + 1e-50)
    return probs / probs.sum(dim=-1, keepdim=True)
