from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
import torch
from datasets import Dataset, DatasetDict
from torch import Tensor, tensor
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.generation.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .prompt import Prompt
from .utils import LLM4ClassificationOutput, SystemPrompt
from .utils import calibrate as calibrate_fn


class LLM4ClassificationBase(torch.nn.Module):
    """Handles the main computations like extracting the logits, calibration and returning new logits."""

    # TODO: Class attributes

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: Prompt,
        generate: bool,
        device: Optional[str] = None,
        lower_verbalizer: bool = False,
    ) -> None:
        """**Base Class for LM-Classifiers**

        Checks correctness of input and initializes the class.

        Args:
            model (PreTrainedModel): The model to be used.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used.
            prompt (Prompt): The prompt to be used. Example usage:
                ```Txt("This text ") + Key('text') + Txt(" is ") + Vbz([['good'], ['bad']])```
                or
                ```FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))```
                More about the prompt object in :ref:`prompt-classes`.
            generate (bool): A flag to determine if the model should be able to generate.
            device (Optional[str], optional): The device to be used. Defaults to None.
            lower_verbalizer (bool, optional): A flag to determine if the verbalizer should be lowercased. Defaults to False.

        Raises:
            AssertionError: If model is not of type PreTrainedModel.
            AssertionError: If tokenizer is not of type PreTrainedTokenizerBase.
            AssertionError: If prompt is not of type Prompt.
            AssertionError: If generate is not of type bool.
            AssertionError: If device is not of type str or None.
            AssertionError: If lower_verbalizer is not of type bool.
        """
        assert isinstance(
            model, PreTrainedModel
        ), "Model must be of type PreTrainedModel"
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), "Tokenizer must be of type PreTrainedTokenizerBase"
        assert isinstance(prompt, Prompt), "Prompt must be of type Prompt"
        assert isinstance(generate, bool), "Generate must be of type bool"
        assert device is None or isinstance(
            device, str
        ), "Device must be of type str or None"
        assert isinstance(
            lower_verbalizer, bool
        ), "Lower Verbalizer must be of type bool"

        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model: PreTrainedModel = model

        self.causal: bool = generate

        self.set_prompt(prompt, lower_verbalizer=lower_verbalizer)

        if device is None and torch.cuda.is_available():
            self.device: str = "cuda"
        else:
            self.device = self.model.device
        try:
            self.model.to(self.device)
        except Exception as exp:
            self.device = self.model.device
            warn(
                f"Could not move the model to the specified device. The `device` is set to the model's current device ({self.model.device}).\n\t'->{exp}",
                category=UserWarning,
            )

        self.calibration_probs: Optional[Tensor] = None

        if self.causal:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def set_prompt(self, prompt: Prompt, lower_verbalizer: bool = False) -> None:
        """
        Can be used for initialization or updating the object.

        Args:
            prompt (Prompt): The prompt to be set.
            lower_verbalizer (bool, optional): A flag to determine if the verbalizer should be lowercased. Defaults to False.
        """
        self.prompt: SystemPrompt = SystemPrompt(
            prompt, self.tokenizer, generate=self.causal
        )
        self.verbalizer_raw: List[List[str]] = self.prompt.verbalizer.verbalizer
        self.verbalizer_dict: Optional[Dict[Union[int, str], List[str]]] = None
        if self.prompt.verbalizer.verbalizer_dict is not None:
            self.verbalizer_dict = self.prompt.verbalizer.verbalizer_dict

        self.verbalizer_indices, self.grouped_indices = self._get_verbalizer(
            self.verbalizer_raw, lower=lower_verbalizer
        )

    def _get_verbalizer(
        self,
        verbalizer_raw: List[List[str]],
        lower: bool = False,
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Build verbalizer and add improve it with lowercased words if needed or add the intermediate token.
        Add ing the previous token (' ' + 'TOKEN' = ' TOKEN') can lead to improved performance.

        Args:
            verbalizer_raw (List[List[str]]): The raw verbalizer.
            lower (bool, optional): A flag to determine if the verbalizer should be lowercased. Defaults to False.s

        Returns:
            Tuple[List[int], List[List[int]]]: The verbalizer indices and the grouped indices.
        """
        if isinstance(verbalizer_raw, list):
            assert 0 not in [
                len(item) for sublist in verbalizer_raw for item in sublist
            ], "Empty string in verbalizer detected! Please provide only non-empty strings."
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
        if not self.causal:
            if True in [
                True in [len(v) > 1 for v in e] for e in verbalizer_tokenized_raw
            ]:
                warn(
                    "Warning: Some tokens are subwords and only the first subword is used. "
                    + "This may lead to unexpected behavior. Consider using a different word.",
                    category=UserWarning,
                )
        verbalizer_tokenized: List[List[int]] = [
            [tok[0] for tok in label_tok] for label_tok in verbalizer_tokenized_raw
        ]

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
    def group_logits(logits: Tensor, grouped_indices: List[List[int]]) -> Tensor:
        """
        Combines the logits if different label words are used by taking the arithmetic mean of the logits
        for each class.

        Args:
            logits (torch.Tensor): The logits to be combined.

        Returns:
            torch.Tensor: The combined logits.
        """
        return torch.stack(
            [
                torch.stack([torch.sum(e[idx] / len(idx)) for idx in grouped_indices])
                for e in logits
            ]
        )

    def _predicted_indices_to_labels(self, predicted: Tensor) -> Tensor:
        """
        Converts the predicted indices to labels if the verbalizer dictionary is available. If return_type is set to 'torch'
        while the keys of the verbalizer dict are strings, 'return_type' is set to 'list'.

        Args:
            predicted (torch.Tensor): The predicted indices.
            return_type (str): Intended type for output.

        Returns:
            Tuple[Union[torch.Tensor, List[str]], str]: The predicted labels (either as tensor or list) and the 'return_type' variable.
        """
        if self.verbalizer_dict is not None:
            verb_kes_list: List[Union[int, str]] = list(self.verbalizer_dict.keys())
            if True in [isinstance(e, str) for e in self.verbalizer_dict.keys()]:
                predicted = [verb_kes_list[idx.item()] for idx in predicted]
            else:
                predicted = tensor([verb_kes_list[idx.item()] for idx in predicted])
            # predicted = tensor([verb_kes_list[idx.item()] for idx in predicted])
        return predicted

    def calibrate_output(
        self, output: LLM4ClassificationOutput
    ) -> LLM4ClassificationOutput:
        """
        Wrapper for the :meth:`promptzl.utils.calibrate` method that retains the types (e.g., 'torch', 'pandas' etc.)
        and returns an updated :class:`promptzl.utils.LLM4ClassificationOutput` object
        with calibrated probabilities. More about calibration is available in :ref:`calibration`.

        Args:
            output (LLM4ClassificationOutput): A :class:`promptzl.utils.LLM4ClassificationOutput`
                object with predictions and probabilites.
        Returns:
            LLM4ClassificationOutput: A :class:`promptzl.utils.LLM4ClassificationOutput`
                with calibrated probabilities and predictions. Logits are kept not altered if available.
        """
        return_type: str = "torch"
        distribution: Union[
            Tensor, np.ndarray, pd.DataFrame, pl.DataFrame, List[List[float]]
        ] = output.distribution
        if isinstance(distribution, torch.Tensor):
            pass
        elif isinstance(distribution, np.ndarray):
            return_type = "numpy"
            distribution = torch.from_numpy(distribution)
        elif isinstance(distribution, pd.DataFrame):
            return_type = "pandas"
            distribution = torch.from_numpy(distribution.values)
        elif isinstance(distribution, pl.DataFrame):
            return_type = "polars"
            distribution = torch.from_numpy(distribution.to_numpy().copy())
        elif isinstance(distribution, list):
            return_type = "list"
            distribution = tensor(distribution)

        distribution = calibrate_fn(distribution)
        predictions: Union[Tensor, List[str]] = torch.argmax(distribution, dim=-1)
        predictions = self._predicted_indices_to_labels(predictions)

        return LLM4ClassificationOutput(
            self._prepare_output(predictions, return_type, True),
            self._prepare_output(distribution, return_type, False),
        )

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ModelOutput]]:
        """**Forward Function.**

        Perform the forward pass of the model and return the logits for each class.
        This method must be implemented in a child class.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            kwargs: Additional arguments for the model.

        Raises:
            NotImplementedError: If the forward function is not implemented.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
        """
        raise NotImplementedError("Forward function must be implemented in subclass.")

    def _prepare_output(
        self, output: Tensor, return_type: str, predict_labels: bool
    ) -> Union[
        Tensor, np.ndarray, List[Union[str, float, int]], pd.DataFrame, pl.DataFrame
    ]:
        """
        Transforms data into the desired output type.

        Args:
            output (torch.Tensor): The output to be prepared. Can be the predicted tensor or the distribution tensor.
            return_type (str): The return type (Supported types are "list", "torch", "numpy", "pandas" and "polars").
            predict_labels (bool): A flag to determine the output tensor is already the tensor with the predicted labels
                or the distribution tensor.

        Returns:
            Union[torch.Tensor, np.ndarray, List[Union[str, float, int]], pd.DataFrame, pl.DataFrame]: The prepared output.
        """
        if return_type == "torch":
            return output
        elif return_type == "numpy":
            if isinstance(output, list):
                return np.asarray(output)
            else:
                return output.numpy()
        elif return_type == "list":
            if isinstance(output, list):
                return output
            else:
                return output.tolist()
        elif return_type == "polars":
            if self.verbalizer_dict is not None:
                if isinstance(output, list):
                    output = np.asarray(output)  #
                else:
                    output = output.numpy()
                return pl.DataFrame(
                    output,
                    schema=["Prediction"]
                    if predict_labels
                    else [str(e) for e in self.verbalizer_dict.keys()],
                )
            else:
                return pl.DataFrame(
                    output.numpy(),
                    schema=["Prediction"]
                    if predict_labels
                    else [e[0] for e in self.verbalizer_raw],
                )
        else:
            if self.verbalizer_dict is not None:
                if isinstance(output, list):
                    output = np.asarray(output)
                else:
                    output = output.numpy()
                return pd.DataFrame(
                    output,
                    columns=["Prediction"]
                    if predict_labels
                    else [str(e) for e in self.verbalizer_dict.keys()],
                )
            else:
                return pd.DataFrame(
                    output.numpy(),
                    columns=["Prediction"]
                    if predict_labels
                    else [e[0] for e in self.verbalizer_raw],
                )

    def _smart_forward(
        self,
        dataset: Dataset,
        batch_size: int,
        return_logits: bool,
        show_progress_bar: bool,
        return_type: str,
        temperature: float,
        **kwargs: Any,
    ) -> LLM4ClassificationOutput:
        """
        Smart batch dataset and predict labels for the dataset. Returns the results in the requested format.


        Args:
            dataset (Dataset): The dataset to be classified.
            batch_size (int): The batch size to be used.
            return_logits (bool): A flag to determine if the logits should be returned.
            show_progress_bar (bool): A flag to determine if the progress bar should be shown.
            return_type (str): The return type. Defaults to "torch". Supported types are "list",
                "torch", "numpy", "pandas" and "polars".
            temperature (float): The temperature to be used. Defaults to 1.0.
            kwargs: Additional arguments for the forward function (the model).

        Returns:
            List[tensor]: The output logits.
        """
        length_sorted_idx = np.argsort([-len(e) for e in dataset])
        dataset = dataset.select(length_sorted_idx)
        collector: List[Tensor] = []
        if return_logits:
            collector_logits: List[Tensor] = []

        self.model.eval()

        for i in trange(
            0,
            len(dataset),
            batch_size,
            desc="Classify Batches...",
            disable=not show_progress_bar,
        ):
            batch: Dict[str, Tensor] = self.prompt.get_tensors_fast(
                dataset[i : i + batch_size]
            )
            with torch.no_grad():
                logits: Tensor = self.forward(batch, **kwargs)
                logits = logits.detach().cpu()
                logits = self.group_logits(logits, self.grouped_indices)
                if temperature != 1.0:
                    output: Tensor = torch.nn.functional.softmax(
                        logits / temperature, dim=-1
                    )
                else:
                    output = torch.nn.functional.softmax(logits, dim=-1)
                collector.extend(output)
                if return_logits:
                    collector_logits.extend(logits)

        self.model.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output = torch.stack([collector[idx] for idx in np.argsort(length_sorted_idx)])
        if return_logits:
            logits = torch.stack(
                [collector_logits[idx] for idx in np.argsort(length_sorted_idx)]
            )

        predicted = torch.argmax(output, dim=-1)
        predicted = self._predicted_indices_to_labels(predicted)

        if return_logits:
            return LLM4ClassificationOutput(
                self._prepare_output(predicted, return_type, True),
                self._prepare_output(output, return_type, False),
                self._prepare_output(logits, return_type, False),
            )
        else:
            return LLM4ClassificationOutput(
                self._prepare_output(predicted, return_type, True),
                self._prepare_output(output, return_type, False),
            )

    def classify(
        self,
        data: Union[Dataset, DatasetDict],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        return_logits: bool = False,
        return_type: str = "torch",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Union[LLM4ClassificationOutput, Dict[str, LLM4ClassificationOutput]]:
        """
        Classifies the data and returns the results in the requested format. For the prediction loop, smart-batching is used,
        where the data is sorted according to the lengths of the instances and then predicted as the longest first. After the
        prediction, the data is reordered into its initial order.

            Args:
                data (Union[Dataset, Any]): The data to be classified.
                batch_size (int): The batch size to be used. Defaults to 64.
                show_progress_bar (bool): A flag to determine if the progress bar should be shown. Defaults to False.
                return_logits (bool): A flag to determine if the logits should be returned. Defaults to False.
                    If the logits are returned and a label in the verbalizer contains more than one word, the logits
                    are averaged for the label group. E.g. :code:`Vbz([['good', 'great'], ['bad']])` mean logits are computed
                    for :code:`['good', 'great']`
                return_type (str): The return type. Defaults to "torch". Supported types are "list",
                    "torch", "numpy", "pandas" and "polars".
                temperature (float): The temperature for the softmax function. Defaults to 1.0.
                kwargs: Additional arguments for the model's forward function.

            Returns:
                Any: The output logits.
        """
        assert return_type in [
            "list",
            "torch",
            "numpy",
            "pandas",
            "polars",
        ], "`return_type` must be: 'list', 'numpy', 'torch', 'polars' or 'pandas'"
        if self.verbalizer_dict is not None:
            if True in [isinstance(e, str) for e in self.verbalizer_dict.keys()]:
                assert return_type != "torch", (
                    "Verbalizer has been provided with a dictionary of the form `Dict[str, Any]."
                    " However return_type is set to 'torch'. String-tensors are not supported with this option."
                    "Please consider using a different return_type (e.g. 'list', 'numpy', 'pandas' or 'polars')."
                )
        temperature = float(temperature)
        assert temperature > 0.0, "Temperature must be greater than 0."

        if isinstance(data, Dataset):
            return self._smart_forward(
                data,
                batch_size,
                return_logits,
                show_progress_bar=show_progress_bar,
                return_type=return_type,
                temperature=temperature,
                **kwargs,
            )
        elif isinstance(data, DatasetDict):
            return_dict: Dict[str, LLM4ClassificationOutput] = {}
            for key in data.keys():
                results: LLM4ClassificationOutput = self._smart_forward(
                    data[key],
                    batch_size,
                    return_logits,
                    show_progress_bar=show_progress_bar,
                    return_type=return_type,
                    temperature=temperature,
                    **kwargs,
                )
                return_dict[key] = results
            return return_dict
        else:
            raise ValueError("Data must be of type Dataset or DatasetDict.")

    def __repr__(self) -> str:
        """String Representation."""
        prompt_str = self.prompt.__str__()
        if "\n" in prompt_str:
            prompt_str = '\n    "' + "\n     ".join(prompt_str.split("\n")) + '"'
        else:
            prompt_str = f'"{prompt_str}"'
        return (
            f"Hub-ID            : {self.model.name_or_path}\n\n"
            f"Prompt            : {prompt_str}\n\n"
            f"Verbalizer        : {self.verbalizer_raw}\n\n"
            f"Verbalizer-Indices: {self.verbalizer_indices}\n\n"
        ) + super().__repr__()

    def __str__(self) -> str:
        """String Representation."""
        return repr(self)


class MaskedLM4Classification(LLM4ClassificationBase, torch.nn.Module):
    """**Masked-LM-Based Classification**

    This class can be used with all masked-language-based language models from huggingface.co.
    """

    def __init__(
        self,
        model_id: str,
        prompt: Prompt,
        device: Optional[str] = None,
        lower_verbalizer: bool = False,
        model_args: Optional[Dict[str, Any]] = None,
        tok_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """**Masked-Language-Modeling-Based Classification**

        :code:`MaskedLM4Classification` works with all models that can be loaded through
        :code:`AutoModelForMaskedLM.from_pretrained(model_id)` with valid model_ids from
        `huggingface.co <https://huggingface.co/models?pipeline_tag=fill-mask>`_.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt (Prompt): A prompt object. Example usage:
                ```Txt("This text ") + Key('text') + Txt(" is ") + Vbz([['good'], ['bad']])```
                or
                ```FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))```
                More about the prompt object in :ref:`prompt-classes`.
            device (Optional[str]): The device to be used. Defaults to None.
            lower_verbalizer (bool): A flag to determine if the verbalizer should be enhanced with lowercased words.
                Defaults to False.
            model_args (Optional[Dict[str, Any]]): Additional arguments for initializing the underlying huggingface-model.
            tok_args (Optional[Dict[str, Any]]): Additional arguments for initializing the underlying huggingface-model.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            clean_up_tokenization_spaces=True,
            use_fast=True,
            **tok_args if tok_args is not None else {},
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_id, **model_args if model_args is not None else {}
        )
        super().__init__(
            model,
            tokenizer,
            prompt,
            generate=False,
            device=device,
            lower_verbalizer=lower_verbalizer,
        )

    def forward(
        self,
        batch: Dict[str, tensor],
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ModelOutput]]:
        """
        Perform forward pass and return logits.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            kwargs: Additional arguments for the model.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits: Optional[Tensor] = None

        outputs: ModelOutput = self.model(**batch, **kwargs)
        mask_index_batch, mask_index_tok = torch.where(
            batch["input_ids"] == self.tokenizer.mask_token_id
        )
        assert (
            mask_index_tok.shape[0] == batch["input_ids"].shape[0]
        ), "Mask token not found in input!"
        logits = outputs.logits[mask_index_batch, mask_index_tok]
        logits = logits[:, self.verbalizer_indices]

        if return_model_output:
            return logits, outputs
        else:
            return logits

    def __repr__(self) -> str:
        """String Representation."""
        return "Type              : MaskedLM4Classification\n\n" + super().__repr__()

    def __str__(self) -> str:
        """String Representation."""
        return repr(self)


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
        model_args: Optional[Dict[str, Any]] = None,
        tok_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """**Causal-LM-Based Classification**

        :code:`CausalLM4Classification` works with all models that can be loaded through
        :code:`AutoModelForCausalLM.from_pretrained(model_id)` with valid model_ids from
        `huggingface.co <https://huggingface.co/models?pipeline_tag=text-generation>`_.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt (Prompt): A prompt object. Example usage:
                ```Txt("This text ") + Key('text') + Txt(" is ") + Vbz([['good'], ['bad']])```
                or
                ```FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))```
                More about the prompt object in :ref:`prompt-classes`.
            device (Optional[str]): The device to be used. Defaults to None.
            lower_verbalizer (bool): A flag to determine if the verbalizer should be enhanced with lowercased words.
                Defaults to False.
            model_args (Optional[Dict[str, Any]]): Additional arguments for initializing the underlying huggingface-model.
            tok_args (Optional[Dict[str, Any]]): Additional arguments for initializing the underlying huggingface-model.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            clean_up_tokenization_spaces=True,
            use_fast=True,
            padding_side="left",
            **tok_args if tok_args is not None else {},
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **model_args if model_args is not None else {}
        )
        super().__init__(
            model,
            tokenizer,
            prompt,
            generate=True,
            device=device,
            lower_verbalizer=lower_verbalizer,
        )

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ModelOutput]]:
        """
        Perform forward pass and return logits.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            kwargs: Additional arguments for the model.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits: Optional[Tensor] = None

        outputs: ModelOutput = self.model(**batch, **kwargs)
        logits = outputs.logits[:, -1, self.verbalizer_indices]

        if return_model_output:
            return logits, outputs
        else:
            return logits

    def __repr__(self) -> str:
        """String Representation."""
        return "Type              : MaskedLM4Classification\n\n" + super().__repr__()

    def __str__(self) -> str:
        """String Representation."""
        return repr(self)
