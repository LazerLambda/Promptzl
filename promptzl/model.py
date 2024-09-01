"""Promptzl, 2024.

MIT LICENSE
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
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
    """Class for Pattern Exploitatoin Classificatoin.

    This class handles the underlying model and the inference mechanism. It aims to tranform a LLM into a
    classifier.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,  # TODO Check types
        prompt_or_verbalizer,  #: Union[Prompt, Verbalizer],
        generate: bool,
        *args,
        **kwargs,
    ):
        """Initialize Class.

        Build the class based on the desired use case determined by the arguments.

        :param model: The model to be used. It is a pretrained model from the Huggingface Transformers library.
        :param tokenizer: The tokenizer to be used. It is a pretrained tokenizer from the Huggingface Transformers library.
        :param verbalizer: The verbalizer to be used. It is a list of lists of strings. Each list of strings represents a class.
        :param generate: A flag to determine if the model is autoregressive and can _generate_ or not. If not, the model is treated as a masked language model.
            E.g.: `[["good"], ["bad"]]`, `[["good", "positive"], ["bad", "negative"]]`
        :param prompt: The prompt to be used. If None, the model will be used without a prompt. This case requires the data to be preprocessed.
        :param kwargs: Additional keyword arguments to be passed to the model.
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

        self.verbalizer_tok, self.i_dict = self._get_verbalizer(self.verbalizer_raw)
        self.calibration_probs: Optional[torch.tensor] = None

        if not self._can_generate:
            if self.tokenizer.mask_token_id is None or not hasattr(
                self.tokenizer, "mask_token_id"
            ):
                raise ValueError(
                    "The tokenizer does not have a mask token. Please use a model that supports masked language modeling."
                )

    def set_contextualized_prior(self, support_set: DataLoader) -> None:
        """Compute Contextualized Prior.

        Compute the contextualized prior form equation (2) based on [Hu et al., 2022](https://aclanthology.org/2022.acl-long.158/).
        A support set is used to obtain the logits of the model for the labels. The logits are then averaged and normalized (softmax) to obtain
        the prior.Function taken from [OpenPrompt])(https://thunlp.github.io/OpenPrompt/_modules/openprompt/utils/calibrate.html#calibrate) and
        adapted for our framework.

        :param support_set: The support set to be used for calibration.
        """
        all_logits: List[torch.tensor] = []
        self.model.eval()
        for batch in support_set:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            logits: torch.tensor = self.forward(
                batch, combine=False
            )  # TODO predict method with smart batching
            all_logits.append(logits)
        all_logits_combined: torch.tensor = torch.cat(all_logits, dim=0)
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

        :param verbalizer: The verbalizer to be used.
        :return: The tokenized verbalizer and the dictionary with the indexes.
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

    def _class_probs(
        self, logits: Any, combine: bool = True
    ) -> tensor:  # TODO: maybe add i_dict or in inference method
        """Get the class probabilities.

        Get the class probabilities from the logits. The logits are transformed into probabilities using the softmax function
        based on the indices.

        :param logits: The models logits from the batch.
        :param verbalizer_tok: The tokenized verbalizer.
        :return: The class probabilities.
        """
        # TODO: Check if single and if yes unsqueeze
        out_res: torch.Tensor = torch.cat(
            list(
                map(
                    lambda i: logits[:, self.verbalizer_tok[i]],
                    range(len(self.verbalizer_tok)),
                )
            ),
            axis=-1,
        )
        out_res = torch.nn.functional.softmax(out_res, dim=1)
        if self.calibration_probs is not None:
            assert self.calibration_probs is not None, "Calibration logits not set!"
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
        return_logits: bool = False,
        **kwargs,
    ) -> Union[tensor, Tuple[tensor, Any]]:
        """Forward pass.

        Perform the forward pass of the model. The model generates the output based on the input batch.

        :param batch: The input batch.
        :param return_model_output: A flag to determine if the model output should be returned.
        :param combine: A flag to determine if the probabilities for each label word should be combined.
        :return: The class probabilities.
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
            # TODO: CHeck if this is correct
            logits = outputs.logits[mask_index_batch, mask_index_tok].detach().cpu()
        probs: tensor = self._class_probs(logits, combine=combine)
        if return_model_output:
            return probs, outputs
        else:
            return probs

    def _text_length(self, e):
        if isinstance(e, dict):
            if "input_ids" in e.keys():
                return len(e["input_ids"])

    def classify(
        self,
        dataset,
        batch_size=100,
        show_progress_bar: bool = False,
        return_logits: bool = False,
        return_type: str = "torch",
        calibrate: bool = True,
        **kwargs,
    ):
        """Classify the dataset.

        Classify the dataset based on the model. The dataset is tokenized and the model is used to classify the data.

        :param dataset: The dataset to be classified.
        :param batch_size: The batch size to be used for classification.
        :param show_progress_bar: A flag to determine if the progress should be shown.
        :param **kwargs: Additional arguments for the forward function passed to the model.
        :return: The class probabilities.
        """
        assert return_type in [
            "list",
            "torch",
            "numpy",
            "pandas",
        ], "`return_type` must be: 'list', 'numpy', 'torch' or 'pandas'"
        if "input_ids" not in dataset:
            if self.prompt is not None:
                dataset = self.prompt.prepare_dataset(dataset)
                dataset.set_format(
                    type="torch", columns=["input_ids", "attention_mask"]
                )

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

        collector = []
        device = self.model.device
        for batch in tqdm(
            dataloader, desc="Classify...", disable=not show_progress_bar
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = self.forward(batch, return_logits, **kwargs)
            output = torch.nn.functional.softmax(output, dim=-1)
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
    """Docstring TODO."""

    def __init__(self, model_id, prompt_or_verbalizer, **kwargs):
        """Initialize Class."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True
        )
        model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
        super().__init__(model, tokenizer, prompt_or_verbalizer, generate=False)


class CausalModel4Classification(LLM4ClassificationBase, torch.nn.Module):
    """Docstring TODO."""

    def __init__(self, model_id, prompt_or_verbalizer, **kwargs):
        """Initialize Class."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        super().__init__(model, tokenizer, prompt_or_verbalizer, generate=True)
