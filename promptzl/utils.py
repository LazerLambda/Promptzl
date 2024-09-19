"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .prompt import Key, Prompt


class DataCollatorPrompt:
    """Data-Collator for Prompt.

    This class entails all functions to construct the data for each instance by concatenating the prompt and the
    respective texts. To avoid extending the maximum length of the model, the text from the data is truncated and
    it is ensured that the prompt is available in the text. The data is then tokenized and prepared for the model.
    """

    def __init__(
        self,
        prompt: Prompt,
        tokenizer: PreTrainedTokenizerBase,
        padding_side: str,
        padding: bool = True,
    ) -> None:
        """Initialize Class.

        Args:
            prompt (Prompt): Prompt object.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for padding.
            padding_side (str): Padding side.
            padding (bool): Padding
        """
        self.prompt: Prompt = prompt
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.padding_side: str = padding_side
        self.padding: bool = padding
        self.max_len: int = tokenizer.model_max_length

        self.special_ids_tensor: tensor = tensor(
            list(
                set(self.tokenizer.all_special_ids)
                - set([self.tokenizer.mask_token_id])
            )
        )
        self.truncate_data: bool = self.prompt.truncate_data
        self.max_allowed_per_prompt: int = int(
            (self.max_len - self.prompt.used_tokens) // len(self.prompt.key_list)
        )

    def _extract_text(
        self, input_ids: Union[List[int], tensor]
    ) -> Tuple[tensor, tensor, tensor]:
        """Extract Text from Input IDs.

        All text (tokens that are not special tokens besides MASK token) are extracted and returned along
        the prefix and suffix special tokens (which may be bos or eos tokens).

        Args:
            input_ids (Union[List[int], tensor]): Input IDs.

        Returns:
            Tuple[tensor, tensor, tensor]: Text, Prefix Special Tokens, Suffix Special Tokens.
        """
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        text_ids = input_ids[
            torch.isin(input_ids, self.special_ids_tensor, invert=True)
        ]
        special_token_idx = torch.where(
            torch.isin(input_ids, self.special_ids_tensor, invert=False)
        )[0]
        range_tensor = torch.arange(special_token_idx.size(0))
        prefix_special, suffix_special = (
            input_ids[special_token_idx[range_tensor - special_token_idx == 0]],
            input_ids[special_token_idx[range_tensor - special_token_idx != 0]],
        )
        return text_ids, prefix_special, suffix_special

    def _tokenizer_call(
        self, elem: List[List[str]], special_token: bool
    ) -> Dict[str, tensor]:
        return self.tokenizer(
            *elem,
            padding=False,
            truncation=True,
            add_special_tokens=special_token,
            return_token_type_ids=False,
        )

    def _format_tokens(
        self, elem: tensor, key: str, prefix_len: int, suffix_len: int
    ) -> tensor:
        return (
            elem[key][0 : (self.max_allowed_per_prompt - suffix_len - prefix_len)]
            if self.truncate_data
            else elem[key]
        )

    def _combine_and_prepare(self, elem: Dict[str, tensor]) -> Dict[str, tensor]:
        input_ids: tensor = torch.concat(
            [elem["prefix"]]
            + [
                (
                    self._format_tokens(
                        elem, e.key, len(elem["prefix"]), len(elem["suffix"])
                    )
                    if isinstance(e, Key)
                    else e.text_tokenized
                )
                for i, e in enumerate(self.prompt.prompt)
            ]
            + [elem["suffix"]]
        )
        attention_mask: tensor = torch.ones(len(input_ids), dtype=int)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def prepare_data(self, data: List[Dict[str, str]]) -> Dict[str, tensor]:
        """Prepare Data for the Model.

        List of dicts with potentially multiple texts is put together in the order provided by the prompt. The data is
        then tokenized and prepared for the model.

        Args:
            data (List[Dict[str, str]]): Data to be prepared.

        Returns:
            Dict[str, tensor]: Prepared Data.
        """
        data_combined = {
            k: [dic[k] for dic in data]
            for k in [e.key for e in self.prompt.prompt if isinstance(e, Key)]
        }
        data_prepared = [
            (
                self._tokenizer_call([v], True)
                if i == 0
                else self._tokenizer_call([v], False)
            )
            for i, (k, v) in enumerate(data_combined.items())
            if False not in [isinstance(e, str) for e in v]
        ]
        data = [
            dict(
                zip(
                    [self.prompt.key_list[0], "prefix", "suffix"],
                    self._extract_text(e[0]),
                )
            )
            | dict(zip(self.prompt.key_list[1::], [tensor(elem) for elem in e[1::]]))
            for e in zip(*[v["input_ids"] for v in data_prepared])
        ]
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [self._combine_and_prepare(e) for e in data],
            padding="longest",
            max_length=self.tokenizer.max_len_single_sentence - 1,  # TODO generate
            return_tensors="pt",
        )
        return batch

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, tensor]:
        """Call Collator.

        Args:
            batch (List[Dict[str, str]]): Instances for creating a batch.

        Returns:
            Dict[str, tensor]: Batch.
        """
        batch: Dict[str, tensor] = self.prepare_data(batch)  # type: ignore[no-redef]
        return batch  # type: ignore[return-value]


class DataCollatorPromptFast:
    """Data-Collator for Prompt.

    This class is a faster version of `DataCollatorPrompt` that puts the text together with the prompt and tokenizes
    but does not truncate the data. This is useful when the data is already truncated and the prompt is not too long.
    It is important to ensure that the prompt is available in the text.
    """

    def __init__(
        self, prompt: Prompt, tokenizer: PreTrainedTokenizerBase, padding_side: str
    ) -> None:
        """Initialize Class.

        Args:
            prompt (Prompt): Prompt object.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for padding.
            padding_side (str): Padding side.
        """
        self.prompt = prompt
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.padding_side = padding_side
        self.max_len = tokenizer.model_max_length

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, tensor]:
        """Call Collator.

        Args:
            examples (List[Dict[str, str]]): Instances for creating a batch.

        Returns:
            Dict[str, tensor]: Batch.
        """
        batch = self.tokenizer(
            *[[self.prompt.get_text(example) for example in examples]],
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_len,
        )
        return batch


class DataCollatorPromptPad:
    """Data-Collator for Padding.

    Similar to `DataCollatorWithPadding` from the transformers library but works on already tokenized
    datasets.
    """

    def __init__(
        self,
        tokenizer: Any,
        padding: str,
        padding_side: str,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """Initialize Class.

        Args:
            tokenizer (Any): Tokenizer for padding.
            padding (str): Padding strategy.
            padding_side (str): Padding side.
            max_length (int): Max length.
            pad_to_multiple_of (int): Important when using tensor cores (https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481).
        """
        self.tokenizer: Any = tokenizer
        self.padding: str = padding
        self.max_length: Optional[int] = max_length
        self.pad_to_multiple_of: Optional[int] = pad_to_multiple_of

    def __call__(self, elem: Any) -> Dict[str, tensor]:
        """Call Collator.

        Args:
            elem (Any): Instances for creating a batch.

        Returns:
            Dict[str, tensor]: Batch.
        """
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            elem,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return batch
