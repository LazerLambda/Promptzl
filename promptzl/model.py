"""Promptzl, 2024.

MIT LICENSE
"""

import random
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .prompt import Prompt, Verbalizer
from .utils import DataCollatorPrompt, DataCollatorPromptFast, DataCollatorPromptPad


class LLM4ClassificationBase(torch.nn.Module):
    """Handles the main computations like extracting the logits, calibration and returning new logits."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt_or_verbalizer: Union[Prompt, Verbalizer],
        generate: bool,
        lower_verbalizer: bool = False,
    ) -> None:
        """Initialize of the Main Class.

        Args:
            model (PreTrainedModel): The model to be used. It is a pretrained model from the Huggingface Transformers library.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used. It is a pretrained tokenizer from the Huggingface
            Transformers library.
            prompt_or_verbalizer (Union[Prompt, Verbalizer]): An Prompt object or a Verbalizer Object. The verbalizer object is used,
            when the data is already pre-processed otherwise
                the pre-processing happens inside the Prompt class. Example:
                    1. Verbalizer:
                        ```Verbalizer([['good'], ['bad']])```
                    2. Prompt:
                        ```Prompt(Text("Classify the following with 'good' or 'bad'"), Text('text'), Verbalizer([['good'], ['bad']]))```
            generate (bool): A flag to determine if the model is autoregressive and can _generate_ or not. If not, the
                model is treated as a masked language model.
                E.g.: `[["good"], ["bad"]]`, `[["good", "positive"], ["bad", "negative"]]`
            lower_verbalizer (bool): A flag to determine if the verbalizer should be enhanced with lowercased words.

        Raise:
            TypeError: In case neither a Prompt of a Verbalizer object is passed for `prompt_or_verbalizer`.
            ValueError: In case the `tokenizer` object does not possess a `mask_token_id` attribute.
        """
        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model: PreTrainedModel = model

        self._can_generate: bool = generate

        self.prompt: Optional[Prompt] = None
        self.verbalizer_raw: List[List[str]] = []

        if not self._can_generate:
            if self.tokenizer.mask_token_id is None or not hasattr(
                self.tokenizer, "mask_token_id"
            ):
                raise ValueError(
                    "The tokenizer does not have a mask token. Please use a model that supports masked language modeling."
                )

        if isinstance(prompt_or_verbalizer, Prompt):
            self.prompt = prompt_or_verbalizer
            self.prompt.subinit(self.tokenizer, self._can_generate)
            self.verbalizer_raw = self.prompt.verbalizer.verbalizer
        elif isinstance(prompt_or_verbalizer, Verbalizer):
            self.verbalizer_raw = prompt_or_verbalizer.verbalizer
            self.prompt = Prompt(prompt_or_verbalizer)  # type: ignore[arg-type]
            self.prompt.subinit(self.tokenizer, self._can_generate)
        else:
            raise TypeError(
                "Argument `prompt_or_verbalizer` must be of either `Prompt` or `Verbalizer`."
            )

        # self.verbalizer_tok, self.i_dict = self._get_verbalizer(self.verbalizer_raw)
        # TODO Add last token for generation
        if self._can_generate:
            self.verbalizer_indices, self.grouped_indices = self._get_verbalizer(
                self.verbalizer_raw,
                lower=lower_verbalizer,
                last_token=self.prompt.intermediate_token,
                generate=self._can_generate,
            )
        else:
            self.verbalizer_indices, self.grouped_indices = self._get_verbalizer(
                self.verbalizer_raw, lower=lower_verbalizer
            )
        self.calibration_probs: Optional[tensor] = None

        if self._can_generate:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def set_contextualized_prior(self, support_set: DataLoader) -> None:
        """Compute Contextualized Prior.

        Compute the contextualized prior form equation (2) based on [Hu et al., 2022](https://aclanthology.org/2022.acl-long.158/).
        A support set is used to obtain the logits of the model for the labels. The logits are then averaged and normalized (softmax) to obtain
        the prior.Function taken from [OpenPrompt])(https://thunlp.github.io/OpenPrompt/_modules/openprompt/utils/calibrate.html#calibrate) and
        adapted for our framework.

        Args:
            support_set (DataLoader): The support set to be used for calibration.
        """
        all_logits: List[tensor] = []
        self.model.eval()
        for batch in support_set:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            logits: tensor = self.forward(
                batch, combine=False
            )  # TODO predict method with smart batching
            all_logits.append(logits)
        all_logits_combined: tensor = torch.cat(all_logits, dim=0)
        all_logits_combined = all_logits_combined.mean(dim=0)
        self.calibration_probs = torch.nn.functional.softmax(
            all_logits_combined, dim=-1
        )

    # def get_masked_verbalizer(self, verbalizer_raw):

    def _get_verbalizer(
        self,
        verbalizer_raw: List[List[str]],
        lower: bool = False,
        last_token: Optional[str] = None,
        generate: bool = False,
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
        if not generate:
            if True in [
                True in [len(v) > 1 for v in e] for e in verbalizer_tokenized_raw
            ]:
                warn(
                    "Warning: Some tokens are subwords and only the first subword is used. "
                    + "This may lead to unexpected behavior. Consider using a different word."
                )
        verbalizer_tokenized: List[List[int]] = [
            [tok[0] for tok in label_tok] for label_tok in verbalizer_tokenized_raw
        ]

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

    # def _get_verbalizer(
    #     self,
    #     verbalizer: List[List[str]],
    # ) -> Tuple[List[List[int]], Dict[str, List[int]]]:
    #     """Prepare verbalizer.

    #     Preprocess the verbalizer to be used in the model. The verbalizer is tokenized and the indexes are stored in a dictionary.
    #     The indices are further necessary to obtain the logits from the models output.

    #     Args:
    #         verbalizer (List[List[str]]): The verbalizer to be used. E.g.: `[["good"], ["bad"]]`, `[["good", "positive"], ["bad", "negative"]]`.

    #     Raises:
    #         AssertionError: In case a verbalizer with the same tokens in multiple classes is passed or in case a label word with
    #         multiple subtokens is used in MLM.

    #     Returns:
    #         Tuple[List[List[int]], Dict[str, List[int]]]: The tokenized verbalizer (first) and the dictionary with the indices (second) as a tuple.
    #     """
    #     tokenized: List[List[List[List[int]]]] = list(
    #         map(
    #             lambda elem: [
    #                 self.tokenizer(e, add_special_tokens=False)["input_ids"]
    #                 for e in elem
    #             ],
    #             [[[elem] for elem in e] for e in verbalizer],
    #         )
    #     )
    #     if not self._can_generate:
    #         check_token_list: List[List[List[int]]] = [
    #             item for one_dim in tokenized for item in one_dim if len(item[0]) != 1
    #         ]
    #         assert check_token_list == [], (
    #             "Multi token word found. When using MLM-models, only one token per word is permitted.",
    #             f"['{self.tokenizer.decode(check_token_list[0][0])}'] -> '{check_token_list[0]}'",
    #         )
    #     verbalizer_tok: List[List[int]] = [
    #         [item[0] for one_dim in two_dim for item in one_dim]
    #         for two_dim in tokenized
    #     ]

    #     counter = 0
    #     i_dict: Dict[str, List[int]] = {}
    #     for e in verbalizer:
    #         i_dict[e[0]] = []
    #         for _ in e:
    #             i_dict[e[0]].append(counter)
    #             counter += 1
    #     verbalizer_tok_seq: List[int] = [
    #         item for innerlist in verbalizer_tok for item in innerlist
    #     ]
    #     assert len(set(verbalizer_tok_seq)) == len(
    #         verbalizer_tok_seq
    #     ), "Equivalent tokens for different classes detected! This also happens if subwords are equal. Tokens must be unique for each class!"
    #     return verbalizer_tok, i_dict

    def _combine_logits(self, logits: tensor) -> tensor:
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
                    [torch.sum(e[idx] / len(idx)) for idx in self.grouped_indices]
                )
                for e in logits
            ]
        )

    def _class_logits(
        self, logits: tensor, combine: bool = True, calibrate: bool = False
    ) -> tensor:  # TODO: maybe add i_dict or in inference method
        """Get the Class Logits.

        Get the class probabilities from the logits. The logits are transformed into probabilities using the softmax function
        based on the indices.

        Args:
            logits (tensor): The models logits from the batch.
            combine (bool): Boolean determining whether or not the logits for different class labels will be averaged for each class.
            calibrate (bool): Boolean determining whether or not logits will be calibrated.

        Returns:
            tensor: The class probabilities.
        """
        out_res = torch.nn.functional.softmax(logits, dim=1)
        if self.calibration_probs is not None and calibrate:
            shape = out_res.shape
            out_res = out_res / (self.calibration_probs + 1e-15)
            norm = out_res.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
            out_res = out_res.reshape(shape[0], -1) / norm
            out_res = out_res.reshape(*shape)
        out_res = torch.log(out_res)
        if combine:
            out_res = self._combine_logits(out_res)
        return out_res

    def forward(
        self,
        batch: Dict[str, tensor],
        return_model_output: bool = False,
        combine: bool = True,
        calibrate: bool = False,
        **kwargs: Any,
    ) -> Union[tensor, Tuple[tensor, Any]]:
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
        logits: Optional[tensor] = None
        if self._can_generate:
            outputs: GenerateDecoderOnlyOutput = self.model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,  # TODO temperature
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
        probs: tensor = self._class_logits(logits, combine=combine, calibrate=calibrate)
        if return_model_output:
            return probs, outputs
        else:
            return probs

    def _text_length(self, elem: Dict[str, Union[tensor, str, int]]) -> int:
        """Get Length of Instance.

        Args:
            elem (dict): Data instance of which length is to be determined.

        Raises:
            NotImplementedError: In case elem is not a `dict` type.

        Returns:
            int: Length of instances
        """
        if isinstance(elem, dict):
            if "input_ids" in elem.keys():
                return len(elem["input_ids"])
            else:
                # return sum([len(k) for k in elem.keys()])
                return sum([len(v) if isinstance(v, str) else 0 for v in elem.values()])
        else:
            raise NotImplementedError(f"Case '{type(elem)}' not implemented")

    def classify(
        self,
        dataset: Union[Dataset, List[str]],
        batch_size: int = 100,
        show_progress_bar: bool = False,
        return_logits: bool = False,
        return_type: str = "torch",
        calibrate: Union[bool] = False,
        calibrate_samples: int = 200,
        data_collator: str = "safe",
        **kwargs: Any,
    ) -> Union[tensor, np.ndarray, List[List[float]], pd.DataFrame]:
        """Classify a Dataset.

        Classify a dataset using a prompt and a verbalizer. The classification can happen in two different steps:
        1. Dataset is already prepared:
            # TODO check if it works
            ```
                model = MaskedLM4Classification('a-model-on-hf', Verbalizer([['bad'], ['good']]))
                dataset = [e + 'It was [MASK]' for e in dataset]
                dataset = Dataset.from_dict({'text': dataset}).map(tokenizer)
                model.classify(dataset)
            ```
        2. Dataset is prepared on the fly:
                ```
                model = MaskedLM4Classification('a-model-on-hf',
                    Prompt(Key('text'), Prompt('It was '), Verbalizer([['bad'], ['good']]))
                dataset = Dataset.from_dict({'text': ["The pizza was good.", "The pizza was bad."]})
                model.classify(dataset)
            ```
        By default, calibration is applied as described in [Hu et al., 2022](https://aclanthology.org/2022.acl-long.158/),
        this can be reset by setting `calibrate` to `False`.

        Args:
            dataset (Union[Dataset, List[str]]): Dataset to be classified. In case of `List[str]`, a prompt is required for
                `prompt_or_verbalizer` upon initialization.
            batch_size (int): Batch size for inference.
            show_progress_bar (bool): Show progress bar during inference.
            return_logits (bool): Boolean determining whether or not logits will be returned.
            return_type (str): Desired return type. Must be in: ["list", "torch", "numpy", "pandas"]. Default is "torch"
            calibrate (Union[bool]): Boolean determining whether or not logits will be calibrated.
            calibrate_samples (int): Number of samples to be used for calibration. Only works if `calibrate` is set to `True`.
            data_collator (str): Data collator to be used. Must be in: ["fast", "safe"]. Default is "safe". The fast data collator
                is faster but does not truncate data if the context length is to long. .
            **kwargs: Additional arguments for the underlying huggingface-model.

        Raises:
            AssertionError: In case `return_type` is misspecified, `dataset` has the wrong type and if `prompt_or_verbalizer`
                is not set correctly.

        Returns:
            Union[tensor, np.ndarray, List[List[float]], pd.DataFrame]: Probabilities for different classes for all instances.
        """
        assert data_collator in [
            "safe",
            "fast",
        ], "`data_collator` must be: 'safe' or 'fast'"
        assert return_type in [
            "list",
            "torch",
            "numpy",
            "pandas",
        ], "`return_type` must be: 'list', 'numpy', 'torch' or 'pandas'"

        data_collator_class: Optional[
            Union[DataCollatorPrompt, DataCollatorPromptFast, DataCollatorPromptPad]
        ] = None
        pad_side: str = "left" if self._can_generate else "right"
        if isinstance(dataset, Dataset):
            if "input_ids" in dataset.column_names:
                dataset.set_format(
                    type="torch", columns=["input_ids", "attention_mask"]
                )
                data_collator_class = DataCollatorPromptPad(
                    self.tokenizer, "max_length", pad_side
                )
        if isinstance(dataset, list):
            assert [
                e for e in dataset if not isinstance(e, str)
            ] == [], "Data is not of type `List[str]`"
            assert (
                self.prompt is not None
            ), "When using data as `List[str]` a Prompt for `prompt_or_verbalizer` is required on initialization."
            dataset = Dataset.from_dict({"text": dataset})

        if data_collator_class is None:
            if data_collator == "fast":
                data_collator_class = DataCollatorPromptFast(
                    self.prompt, self.tokenizer, pad_side  # type: ignore[arg-type]
                )
            else:
                data_collator_class = DataCollatorPrompt(
                    self.prompt, self.tokenizer, pad_side  # type: ignore[arg-type]
                )

        if bool(calibrate):
            if self.calibration_probs is None:
                n: int = len(dataset)
                if calibrate_samples > n:
                    calibrate_samples = int(n // 2)
                random_indices: List[int] = random.sample(range(n), calibrate_samples)
                dataset_cali = dataset.select(random_indices)
                dataloader_cali = DataLoader(
                    dataset_cali, collate_fn=data_collator_class
                )
                self.set_contextualized_prior(dataloader_cali)

        length_sorted_idx = np.argsort([-self._text_length(inst) for inst in dataset])
        dataset = dataset.select(length_sorted_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator_class,
            shuffle=False,
        )

        collector: List[tensor] = []
        device = self.model.device
        for batch in tqdm(
            dataloader, desc="Classify...", disable=not show_progress_bar
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = self.forward(batch, calibrate=calibrate, **kwargs)
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
        else:
            return pd.DataFrame(
                output.numpy(), columns=[e[0] for e in self.verbalizer_raw]
            )

    def __del__(self) -> None:
        """Delete the model."""
        del self.model
        torch.cuda.empty_cache()


class MaskedLM4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Masked-Language-Modeling-Based Classification.

    This class can be used with all masked-language-based language models from huggingface.co.
    """

    def __init__(
        self,
        model_id: str,
        prompt_or_verbalizer: Union[Prompt, Verbalizer],
        **kwargs: Any,
    ) -> None:
        """Initialize Class.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt_or_verbalizer (Union[Prompt, Verbalizer]): An Prompt object or a Verbalizer Object. The verbalizer object
                is used, when the data is already pre-processed otherwise
                the pre-processing happens inside the Prompt class. Example:
                    1. Verbalizer:
                        ```Verbalizer([['good'], ['bad']])```
                    2. Prompt:
                        ```Prompt(Text("Classify the following with 'good' or 'bad'"), Text('text'), Verbalizer([['good'], ['bad']]))```
            **kwargs: Additional arguments for initializing the underlying huggingface-model.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True, use_fast=True
        )
        model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
        super().__init__(model, tokenizer, prompt_or_verbalizer, generate=False)


class CausalLM4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Causal-Language-Modeling-Based Classification.

    This class can be used with all causal/autoregressive language models from huggingface.co.
    """

    def __init__(
        self,
        model_id: str,
        prompt_or_verbalizer: Union[Prompt, Verbalizer],
        **kwargs: Any,
    ) -> None:
        """Initialize Class.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt_or_verbalizer (Union[Prompt, Verbalizer]): An Prompt object or a Verbalizer Object. The verbalizer object
            is used, when the data is already pre-processed otherwise
                the pre-processing happens inside the Prompt class. Example:
                    1. Verbalizer:
                        ```Verbalizer([['good'], ['bad']])```
                    2. Prompt:
                        ```Prompt(Text("Classify the following with 'good' or 'bad'"), Text('text'), Verbalizer([['good'], ['bad']]))```
            **kwargs: Additional arguments for initializing the underlying huggingface-model.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        super().__init__(model, tokenizer, prompt_or_verbalizer, generate=True)
