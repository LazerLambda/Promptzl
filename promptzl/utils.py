"""Promptzl, 2024.

MIT LICENSE
"""

import torch
from typing import Any, Dict, List, Optional, Tuple

from torch import tensor
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from .prompt import Prompt

def split_tokens_left_right(input_ids: tensor, attention_mask: tensor, generate: bool) -> Tuple[tensor, tensor, List[tensor]]:
    if generate:
        attention_mask = attention_mask[:, :-1]
    rows_indices: tensor = tensor(range(input_ids.size(0)))
    bos_indices: tensor = input_ids.shape[1] - attention_mask.sum(dim=1) 
    bos_tokens: tensor = input_ids[rows_indices, bos_indices]
    eos_tokens: tensor = input_ids[rows_indices, torch.zeros_like(bos_indices, dtype=int) + input_ids.shape[1] - 1]
    text_tokens: tensor = [input_ids[i][e] for i, e in enumerate(list(map(lambda e: list(range(1, e)), bos_indices - 1)))]
    return eos_tokens, bos_tokens, text_tokens

# def split_tokens_right_left(input_ids: tensor, attention_mask: tensor) -> Tuple[tensor, tensor, List[tensor]]:
#     rows_indices: tensor = tensor(range(input_ids.size(0)))
#     eos_indices: tensor = attention_mask.sum(dim=1) - 1
#     eos_tokens: tensor = input_ids[rows_indices, eos_indices]
#     bos_tokens: tensor = input_ids[rows_indices, torch.zeros_like(eos_indices, dtype=int)]
#     text_tokens: tensor = [input_ids[i][e] for i, e in enumerate(list(map(lambda e: list(range(1, e)), eos_indices - 1)))]
#     return eos_tokens, bos_tokens, text_tokens



def combine_text(prompt: Prompt, batch: Dict[str, tensor]) -> str:
    """Combine prompt and text.

    Args:
        prompt (Prompt): Prompt.
        text (str): Text.

    Returns:
        str: Combined text.
    """
    eos_tokens, bos_tokens, text_tokens = extract_last_tokens(batch["input_ids"], batch["attention_mask"])

    


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
