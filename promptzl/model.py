"""Promptzl, 2024.

MIT LICENSE
"""

import random
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
import torch
from datasets import Dataset, DatasetDict
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .prompt import Prompt, TKy, IKy, Vbz, Txt
from .utils import SystemPrompt
# from .prompt import Key, Prompt, Verbalizer, get_prompt
# from .utils import DataCollatorPrompt, DataCollatorPromptFast, DataCollatorPromptPad


class LLM4ClassificationBase(torch.nn.Module):
    """Handles the main computations like extracting the logits, calibration and returning new logits."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: Prompt, # TODO: Allow also only verbalizer
        generate: bool,
        device: Optional[str] = None,
        lower_verbalizer: bool = False,
        truncate: bool = True
    ) -> None:
        
        assert isinstance(model, PreTrainedModel), "Model must be of type PreTrainedModel"
        assert isinstance(tokenizer, PreTrainedTokenizerBase), "Tokenizer must be of type PreTrainedTokenizerBase"
        assert isinstance(prompt, Prompt), "Prompt must be of type Prompt"
        assert isinstance(generate, bool), "Generate must be of type bool"
        assert device is None or isinstance(device, str), "Device must be of type str or None"
        assert isinstance(lower_verbalizer, bool), "Lower Verbalizer must be of type bool"
        assert isinstance(truncate, bool), "Truncate must be of type bool"

        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model: PreTrainedModel = model

        self._can_generate: bool = generate

        self.prompt: SystemPrompt = SystemPrompt(prompt, tokenizer, truncate=truncate, mlm=(not generate))
        self.verbalizer_raw: List[List[str]] = self.prompt.verbalizer.verbalizer

        if device is None and torch.cuda.is_available():
            self.device: str = "cuda"
        else:
            self.device: str = self.model.device

        try:
            self.model.to(self.device)
        except Exception as exp:
            self.device: str = self.model.device
            warn(f"Could not move the model to the specified device. The `device` is set to the model's current device.\n\t'->{exp}")

        # TODO Add last token for generation
        if self._can_generate:
            self.verbalizer_indices, self.grouped_indices = self._get_verbalizer(
                self.verbalizer_raw,
                lower=lower_verbalizer,
                last_token=self.prompt.intermediate_token
            )
        else:
            self.verbalizer_indices, self.grouped_indices = self._get_verbalizer(
                self.verbalizer_raw, lower=lower_verbalizer
            )
        self.calibration_probs: Optional[tensor] = None

        if self._can_generate:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id


    def _get_verbalizer(
        self,
        verbalizer_raw: List[List[str]],
        lower: bool = False,
        last_token: Optional[str] = None,
    ) -> Tuple[List[int], List[List[int]]]:
        combine: Callable[
            [List[List[Any]], List[List[Any]]], List[List[Any]]
        ] = lambda a, b: [e[0] + e[1] for e in list(zip(a, b))]

        if lower:
            verbalizer_raw = combine(
                verbalizer_raw, [[elem.lower() for elem in e] for e in verbalizer_raw]
            )
            verbalizer_raw = list(map(lambda e: list(set(e)), verbalizer_raw))

        verbalizer_tokenized_raw: List[List[List[int]]] = [
            [self.tokenizer.encode(e, add_special_tokens=False) for e in label_words]
            for label_words in verbalizer_raw
        ]
        if not self._can_generate:
            if True in [
                True in [len(v) > 1 for v in e] for e in verbalizer_tokenized_raw
            ]:
                warn(
                    "Warning: Some tokens are subwords and only the first subword is used. "
                    + "This may lead to unexpected behavior. Consider using a different word.", category=UserWarning
                )
        verbalizer_tokenized: List[List[int]] = [
            [tok[0] for tok in label_tok] for label_tok in verbalizer_tokenized_raw
        ]

        # TODO: rename to prev_token
        if last_token is not None:
            last_token_ids: List[int] = self.tokenizer.encode(
                last_token, add_special_tokens=False
            )
            last_token_added: List[List[str]] = list(
                map(
                    lambda labels: list(map(lambda e: last_token + e, labels)),
                    verbalizer_raw,
                )
            )
            # Remove if already exists
            last_token_added = [
                [
                    list(
                        filter(
                            lambda token: token not in last_token_ids,
                            self.tokenizer.encode(e, add_special_tokens=False),
                        )
                    )[0]
                    for e in labels
                ]
                for labels in last_token_added
            ]
            verbalizer_tokenized = combine(verbalizer_tokenized, last_token_added)

        # Remove duplicates
        verbalizer: List[List[int]] = list(
            map(lambda e: list(set(e)), verbalizer_tokenized)
        )

        # Check for duplicates in different classes
        verbalizer_indices: List[int] = [
            item for sublist in verbalizer for item in sublist
        ]
        assert len(set(verbalizer_indices)) == len(
            verbalizer_indices
        ), "Equivalent tokens for different classes detected! This also happens if subwords are equal. Tokens must be unique for each class!"

        indices: List[int] = list(range(len(verbalizer_indices)))
        grouped_indices: List[List[int]] = list(  # type: ignore[assignment]
            reduce(
                lambda coll, elem: (  # type: ignore[arg-type,return-value]
                    coll[0] + [indices[coll[1] : (coll[1] + len(elem))]],  # type: ignore[index]
                    coll[1] + len(elem),  # type: ignore[index]
                ),
                verbalizer,
                ([], 0),
            )
        )[0]

        return verbalizer_indices, grouped_indices

    @staticmethod
    def _combine_logits(logits: tensor, grouped_indices: List[List[int]]) -> tensor:
        """Combine Logits.

        Combine the logits for different class labels.

        Args:
            logits (tensor): The logits to be combined.

        Returns:
            tensor: The combined logits.
        """
        return torch.stack(
            [
                torch.stack(
                    [torch.sum(e[idx] / len(idx)) for idx in grouped_indices]
                )
                for e in logits
            ]
        )

    @staticmethod
    def _calibrate(probs: tensor) -> tensor:
        calibration_probs = torch.mean(probs, dim =0)
        shape = probs.shape
        probs = probs / (calibration_probs + 1e-15)
        norm = probs.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
        probs = probs.reshape(shape[0], -1) / norm
        probs = probs.reshape(*shape)
        return probs

    def forward(
        self,
        batch: Dict[str, tensor],
        return_model_output: bool = False,
        combine: bool = True,
        calibrate: bool = False,
        **kwargs: Any,
    ) -> Union[tensor, Tuple[tensor, Any]]: # TODO: Find type
        
        """Forward Function.

        Perform the forward pass of the model.

        Args:
            batch (Dict[str, tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            combine (bool): A flag to determine if the probabilities for each label word should be combined.
            calibrate (bool): Boolean determining whether or not logits will be calibrated.
            kwargs: Additional arguments for the model.

        Returns:
            Union[tensor, Tuple[tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits: Optional[tensor] = None
        if self._can_generate:
            outputs: GenerateDecoderOnlyOutput = self.model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,  # TODO temperature
                # top_k=5,
                do_sample=False,
                **kwargs,
            )
            logits = outputs.scores[0].detach().cpu()
        else:
            mask_index_batch, mask_index_tok = torch.where(
                batch["input_ids"] == self.tokenizer.mask_token_id
            )
            assert (
                mask_index_tok.shape[0] == batch["input_ids"].shape[0]
            ), "Mask token not found in input!"
            outputs = self.model(**batch)
            logits = outputs.logits[mask_index_batch, mask_index_tok].detach().cpu()
        logits = logits[:, self.verbalizer_indices]
        # probs: tensor = logits
        # probs: tensor = self._class_logits(logits, combine=combine, calibrate=calibrate)
        if combine:
            logits = self._combine_logits(logits, self.grouped_indices)
        probs: tensor = torch.nn.functional.softmax(logits, dim=1)
        if calibrate:
            probs = self._calibrate(probs)
        if return_model_output:
            return probs, outputs
        else:
            return probs


    def _smart_forward(self, dataset: Dataset, batch_size: int, return_logits: bool = False, show_progress_bar: bool = True, return_type="torch", calibrate: bool = False, **kwargs) -> List[tensor]:
        length_sorted_idx = np.argsort([-len(e) for e in dataset])
        dataset = dataset.select(length_sorted_idx)
        collector: List[tensor] = []

        for i in trange(0, len(dataset), batch_size, desc="Classify Batches...", disable=not show_progress_bar):
            batch = self.prompt.get_tensors(dataset[i:i+batch_size])
            with torch.no_grad():
                output: tensor = self.forward(batch, calibrate=calibrate, **kwargs)
                output = (
                    torch.nn.functional.softmax(output, dim=-1)
                    if not return_logits
                    else output
                )
                collector.extend(output)

        output = torch.stack([collector[idx] for idx in np.argsort(length_sorted_idx)])
        if return_type == "torch":
            return output
        elif return_type == "numpy":
            return output.numpy()
        elif return_type == "list":
            return output.tolist()
        elif return_type == "polars":
            return pl.DataFrame(
                output.numpy(), schema=[e[0] for e in self.verbalizer_raw]
            )
        else:
            return pd.DataFrame(
                output.numpy(), columns=[e[0] for e in self.verbalizer_raw]
            )

    def classify(
            self,
            data: Union[Dataset, Any], # TODO IMplement DatasetDict
            batch_size: int = 64,
            show_progress_bar: bool = False,
            return_logits: bool = False,
            return_type: str = "torch", # TODO use enum type
            calibrate: bool = False,
            **kwargs: Any,
            ) -> Any:

        assert return_type in [
            "list",
            "torch",
            "numpy",
            "pandas",
            "polars"
        ], "`return_type` must be: 'list', 'numpy', 'torch', 'polars' or 'pandas'"    

        if isinstance(data, Dataset):
            return self._smart_forward(data, batch_size, return_logits, show_progress_bar=show_progress_bar, return_type=return_type, calibrate=calibrate, **kwargs)
        elif isinstance(data, DatasetDict):
            return_dict: Dict[str, List[tensor]] = {}
            for key in data.keys():
                results: tensor = self._smart_forward(data[key], batch_size, return_logits, show_progress_bar=show_progress_bar, return_type=return_type, calibrate=calibrate, **kwargs)
                return_dict[key] = results
            return return_dict

class MaskedLM4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Masked-Language-Modeling-Based Classification.

    This class can be used with all masked-language-based language models from huggingface.co.
    """

    def __init__(
        self,
        model_id: str,
        prompt: Prompt,
        device: Optional[str] = None,
        lower_verbalizer: bool = False,
        truncate: bool = True,
        **kwargs: Any,
    ) -> None:

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True, use_fast=True
        )
        model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
        super().__init__(
            model,
            tokenizer,
            prompt,
            generate=False,
            device=device,
            lower_verbalizer=lower_verbalizer,
            truncate=truncate
        )


class CausalLM4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Causal-Language-Modeling-Based Classification.

    This class can be used with all causal/autoregressive language models from huggingface.co.
    """

    def __init__(
        self,
        model_id: str,
        prompt: Prompt,
        device: Optional[str] = None,
        lower_verbalizer: bool = False,
        truncate: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Class.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt_or_verbalizer (Union[Prompt, Verbalizer, Tuple[str, List[str], Verbalizer]]): An Prompt objectm, a Verbalizer Object or a
            tuple with a c-string like placeholder pattern, List of keys or a key for the data and a Verbalizer. The verbalizer object
            is used, when the data is already pre-processed otherwise
                the pre-processing happens inside the Prompt class. Example:
                    1. Verbalizer:
                        ```Verbalizer([['good'], ['bad']])```
                    2. Prompt:
                        ```Prompt(Text("Classify the following with 'good' or 'bad'"), Text('text'), Verbalizer([['good'], ['bad']]))``
                    3. Tuple[str, List[str], Verbalizer]:
                        ```("Classify the following with 'good' or 'bad': %s", ['text'], Verbalizer([['good'], ['bad']]) )```
                        In case only one key in the template, a single string can also be provided:
                        ```("Classify the following with 'good' or 'bad': %s", 'text', Verbalizer([['good'], ['bad']]) )```
            lower_verbalizer (bool): A flag to determine if the verbalizer should be enhanced with lowercased words.
            **kwargs: Additional arguments for initializing the underlying huggingface-model.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True, use_fast=True # , padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        super().__init__(
            model,
            tokenizer,
            prompt,
            generate=True,
            device=device,
            lower_verbalizer=lower_verbalizer,
            truncate=truncate
        )
