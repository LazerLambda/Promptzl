"""Promptzl, 2024.

MIT LICENSE
"""
from typing import Any, Optional

from transformers.data.data_collator import pad_without_fast_tokenizer_warning


class DataCollatorPromptPad:
    """Docstring TODO."""

    def __init__(
        self,
        tokenizer: Any,
        padding: str,
        padding_side: str,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """Initialize TODO."""
        self.tokenizer: Any = tokenizer
        self.padding: Any = padding
        self.max_length: Optional[int] = max_length
        self.pad_to_multiple_of: Optional[int] = pad_to_multiple_of

    def __call__(self, elem):
        """Call TODO."""
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            elem,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return batch
