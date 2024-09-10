"""Promptzl, 2024.

MIT LICENSE
"""

import torch
from typing import Any, Dict, List, Optional, Tuple

from torch import tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from .prompt import Prompt

def extract_text(input_ids: tensor, tokenizer: : PreTrainedTokenizerBase) -> Tuple[tensor, tensor, tensor]:
    text_ids: tensor = input_ids[torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids), invert=True)]
    special_token_idx = torch.where(torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids), invert=False))[0]
    range_tensor: tensor = torch.arange(special_token_idx.size(0))
    prefix_special, suffix_special = (
        input_ids[special_token_idx[range_tensor - special_token_idx == 0]],
        input_ids[special_token_idx[range_tensor - special_token_idx != 0]]
    )
    return text_ids, prefix_special, suffix_special



def combine_text(prompt: Prompt, batch: Dict[str, tensor], tokenizer: PreTrainedTokenizerBase) -> str:
    """Combine prompt and text.

    Args:
        prompt (Prompt): Prompt.
        text (str): Text.

    Returns:
        str: Combined text.
    """
    batch_tokenized = [tokenizer.encode(*[v]) if i == 0 else tokenizer.encode(*[v], add_special_tokens=False)
                       for i, (k,v) in tokenizer.batch.items()]
    text_ids, prefix_special, suffix_special = extract_text(batch["input_ids"], prompt.tokenizer)


class DataCollatorPrompt:

    def __init__(self, prompt, tokenizer, padding_side: str, padding: bool=True):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.padding = padding
        self.max_len = tokenizer.model_max_length

    def __call__(self, examples):
        print(examples)
        batch = self.tokenizer(
            *[[self.prompt.get_text(example) for example in examples]],
            padding=self.padding,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_len # TODO: Check in s-trafo
        )
        print(batch)
        return batch



class DataCollatorPromptFast:

    def __init__(self, prompt, tokenizer, padding_side: str, padding: bool=True):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.padding = padding
        self.max_len = tokenizer.model_max_length

    def __call__(self, examples):
        print(examples)
        batch = self.tokenizer(
            *[[self.prompt.get_text(example) for example in examples]],
            padding=self.padding,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_len # TODO: Check in s-trafo
        )
        print(batch)
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
