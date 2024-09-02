"""Promptzl, 2024.

MIT LICENSE
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

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
from .utils import DataCollatorPromptPad


class LLM4ClassificationBase(torch.nn.Module):
    """Handles the main computations like extracting the logits, calibration and returning new logits."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,  # TODO Check types
        prompt_or_verbalizer: Union[Prompt, Verbalizer],
        generate: bool,
    ):
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

        if isinstance(prompt_or_verbalizer, Prompt):
            self.prompt = prompt_or_verbalizer
            self.prompt.subinit(self.tokenizer, self._can_generate)
            self.verbalizer_raw = self.prompt.verbalizer.verbalizer
        elif isinstance(prompt_or_verbalizer, Verbalizer):
            self.verbalizer_raw = prompt_or_verbalizer.verbalizer
        else:
            raise TypeError(
                "Argument `prompt_or_verbalizer` must be of either `Prompt` or `Verbalizer`."
            )

        if not self._can_generate:
            if self.tokenizer.mask_token_id is None or not hasattr(
                self.tokenizer, "mask_token_id"
            ):
                raise ValueError(
                    "The tokenizer does not have a mask token. Please use a model that supports masked language modeling."
                )

        self.verbalizer_tok, self.i_dict = self._get_verbalizer(self.verbalizer_raw)
        self.calibration_probs: Optional[tensor] = None

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

    def _get_verbalizer(
        self,
        verbalizer: List[List[str]],
    ) -> Tuple[List[List[int]], Dict[str, List[int]]]:
        """Prepare verbalizer.

        Preprocess the verbalizer to be used in the model. The verbalizer is tokenized and the indexes are stored in a dictionary.
        The indices are further necessary to obtain the logits from the models output.

        Args:
            verbalizer (List[List[str]]): The verbalizer to be used. E.g.: `[["good"], ["bad"]]`, `[["good", "positive"], ["bad", "negative"]]`.

        Raises:
            AssertionError: In case a verbalizer with the same tokens in multiple classes is passed or in case a label word with
            multiple subtokens is used in MLM.

        Returns:
            Tuple[List[List[int]], Dict[str, List[int]]]: The tokenized verbalizer (first) and the dictionary with the indices (second) as a tuple.
        """
        tokenized: List[List[List[List[int]]]] = list(
            map(
                lambda elem: [
                    self.tokenizer(e, add_special_tokens=False)["input_ids"]
                    for e in elem
                ],
                [[[elem] for elem in e] for e in verbalizer],
            )
        )
        if not self._can_generate:
            assert [
                item for one_dim in tokenized for item in one_dim if len(item[0]) != 1
            ] == [], "Multi token word found. When using MLM-models, only one token per word is permitted."
        verbalizer_tok: List[List[int]] = [
            [item[0] for one_dim in two_dim for item in one_dim]
            for two_dim in tokenized
        ]

        counter = 0
        i_dict: Dict[str, List[int]] = {}
        for e in verbalizer:
            i_dict[e[0]] = []
            for _ in e:
                i_dict[e[0]].append(counter)
                counter += 1
        verbalizer_tok_seq: List[int] = [
            item for innerlist in verbalizer_tok for item in innerlist
        ]
        assert len(set(verbalizer_tok_seq)) == len(
            verbalizer_tok_seq
        ), "Equivalent tokens for different classes detected! This also happens if subwords are equal. Tokens must be unique for each class!"
        # TODO: Consider this in label search!
        return verbalizer_tok, i_dict

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
        # TODO: Check if single and if yes unsqueeze
        out_res: tensor = torch.cat(
            list(
                map(
                    lambda i: logits[:, self.verbalizer_tok[i]],
                    range(len(self.verbalizer_tok)),
                )
            ),
            axis=-1,
        )
        out_res = torch.nn.functional.softmax(out_res, dim=1)
        if self.calibration_probs is not None and calibrate:
            shape = out_res.shape
            out_res = out_res / (self.calibration_probs + 1e-15)
            norm = out_res.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
            out_res = out_res.reshape(shape[0], -1) / norm
            out_res = out_res.reshape(*shape)
        out_res = torch.log(out_res)
        if combine:
            out_res = torch.transpose(
                torch.stack(
                    [
                        torch.sum(out_res[:, v], axis=-1) / len(v)
                        for v in self.i_dict.values()
                    ]
                ),
                0,
                1,
            )
        return out_res
        # TODO: verbalizer for labels
        # return class_probs_combined

    def forward(
        self,
        batch: Dict[str, tensor],
        return_model_output: bool = False,
        combine: bool = True,
        calibrate: bool = False,
        **kwargs,
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
        probs: tensor = self._class_logits(logits, combine=combine, calibrate=calibrate)
        if return_model_output:
            return probs, outputs
        else:
            return probs

    def _text_length(self, elem: Dict[str, tensor]) -> int:
        """Get Length of Instance.

        Args:
            elem (dict): Data instance of which length is to be determined.

        Raises:
            NotImplementedError: In case elem is not a `dict` type.

        Returns:
            int: Length of instances
        """
        if isinstance(elem, dict):
            # if "input_ids" in elem.keys():
            return len(elem["input_ids"])
        else:
            raise NotImplementedError(f"Case '{type(elem)}' not implemented")

    def classify(
        self,
        dataset: Union[Dataset, List[str]],
        batch_size: int = 100,
        show_progress_bar: bool = False,
        return_logits: bool = False,
        return_type: str = "torch",
        calibrate: bool = True,
        **kwargs,
    ):
        """Classify a Dataset.

        Classify a dataset using a prompt and a verbalizer. The classification can happen in two different steps:
        1. Dataset is already prepared:
            # TODO check if it works
            ```
                model = MLM4Classification('a-model-on-hf', Verbalizer([['bad'], ['good']]))
                dataset = [e + 'It was [MASK]' for e in dataset]
                dataset = Dataset.from_dict({'text': dataset}).map(tokenizer)
                model.classify(dataset)
            ```
        2. Dataset is prepared on the fly:
                ```
                model = MLM4Classification('a-model-on-hf',
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
            calibrate (bool): Boolean determining whether or not logits will be calibrated.
            **kwargs: Additional arguments for the underlying huggingface-model.

        Raises:
            AssertionError: In case `return_type` is misspecified, `dataset` has the wrong type and if `prompt_or_verbalizer`
                is not set correctly.

        Returns:
            Union[tensor, np.ndarray, List[List[float]], pd.DataFrame]: Probabilities for different classes for all instances.
        """
        assert return_type in [
            "list",
            "torch",
            "numpy",
            "pandas",
        ], "`return_type` must be: 'list', 'numpy', 'torch' or 'pandas'"
        if isinstance(dataset, Dataset):
            if "input_ids" not in dataset:
                if self.prompt is not None:
                    dataset = self.prompt.prepare_dataset(dataset)
                    dataset.set_format(
                        type="torch", columns=["input_ids", "attention_mask"]
                    )
        if isinstance(dataset, list):
            assert [
                e for e in dataset if not isinstance(e, str)
            ] == [], "Data is not of type `List[str]`"
            assert (
                self.prompt is not None
            ), "When using data as `List[str]` a Prompt for `prompt_or_verbalizer` is required on initialization."
            dataset = Dataset.from_dict({"text": dataset})
            dataset = self.prompt.prepare_dataset(dataset)

        if calibrate:
            if self.calibration_probs is None:
                n: int = len(dataset)
                n_sample: int = 200 if n > 200 else n // 2
                random_indices: List[int] = random.sample(range(n), n_sample)
                dataset_cali = dataset.select(random_indices)
                dataloader_cali = DataLoader(dataset_cali)
                self.set_contextualized_prior(dataloader_cali)

        length_sorted_idx = np.argsort([-self._text_length(inst) for inst in dataset])
        dataset = dataset.select(length_sorted_idx)

        pad_side: str = "left" if self._can_generate else "right"
        data_collator = DataCollatorPromptPad(self.tokenizer, "max_length", pad_side)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=data_collator
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
        elif return_type == "pandas":
            return pd.DataFrame(
                output.numpy(), columns=[e[0] for e in self.verbalizer_raw]
            )

    def __del__(self):
        """Delete the model."""
        del self.model
        torch.cuda.empty_cache()


class MLM4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Masked-Language-Modeling-Based Classification.

    This class can be used with all masked-language-based language models from huggingface.co.
    """

    def __init__(
        self, model_id: str, prompt_or_verbalizer: Union[Prompt, Verbalizer], **kwargs
    ):
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


class CausalModel4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Causal-Language-Modeling-Based Classification.

    This class can be used with all causal/autoregressive language models from huggingface.co.
    """

    def __init__(
        self, model_id: str, prompt_or_verbalizer: Union[Prompt, Verbalizer], **kwargs
    ):
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
