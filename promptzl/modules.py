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

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: Prompt,
        generate: bool,
        device: Optional[str] = None,
        lower_verbalizer: bool = False,
        truncate: bool = True,
    ) -> None:
        """Initialize Class.

        Initialize class with the model, tokenizer, prompt, device, lower verbalizer and truncate.
        Check if all input is valid.

        Args:
            model (PreTrainedModel): The model to be used.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used.
            prompt (Prompt): The prompt to be used. Example usage:
                ```Txt("This text ") + Key('text') + Txt(" is ") + Vbz([['good'], ['bad']])```
                or
                ```FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))```
                WARNING: Using FVP can result in indexing errors as automatic truncation is not applied.
            generate (bool): A flag to determine if the model should be able to generate.
            device (Optional[str], optional): The device to be used. Defaults to None.
            lower_verbalizer (bool, optional): A flag to determine if the verbalizer should be lowercased. Defaults to False.
            truncate (bool, optional): A flag to determine if the prompt should be truncated. Defaults to True.

        Raises:
            AssertionError: If model is not of type PreTrainedModel.
            AssertionError: If tokenizer is not of type PreTrainedTokenizerBase.
            AssertionError: If prompt is not of type Prompt.
            AssertionError: If generate is not of type bool.
            AssertionError: If device is not of type str or None.
            AssertionError: If lower_verbalizer is not of type bool.
            AssertionError: If truncate is not of type bool.
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
        assert isinstance(truncate, bool), "Truncate must be of type bool"

        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model: PreTrainedModel = model

        self.causal: bool = generate

        self.set_prompt(prompt)

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

        self.verbalizer_indices, self.grouped_indices = self._get_verbalizer(
            self.verbalizer_raw, lower=lower_verbalizer
        )

        self.calibration_probs: Optional[Tensor] = None

        if self.causal:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def set_prompt(self, prompt: Prompt) -> None:
        """Set Prompt.

        Sets the prompt for the class. Can be used for initialization or updating the object.

        Args:
            prompt (Prompt): The prompt to be set.
        """
        self.prompt: SystemPrompt = SystemPrompt(
            prompt, self.tokenizer, generate=self.causal
        )
        self.verbalizer_raw: List[List[str]] = self.prompt.verbalizer.verbalizer
        self.verbalizer_dict: Optional[Dict[Union[int, str], List[str]]] = None
        if self.prompt.verbalizer.verbalizer_dict is not None:
            self.verbalizer_dict = self.prompt.verbalizer.verbalizer_dict

    def _get_verbalizer(
        self,
        verbalizer_raw: List[List[str]],
        lower: bool = False,
    ) -> Tuple[List[int], List[List[int]]]:
        """Get Verbalizer.

        Build verbalizer and add improve it with lowercased words if needed or add the intermediate token.
        Add ing the previous token (' ' + 'TOKEN' = ' TOKEN') can lead to improved performance.

        Args:
            verbalizer_raw (List[List[str]]): The raw verbalizer.
            lower (bool, optional): A flag to determine if the verbalizer should be lowercased. Defaults to False.s

        Returns:
            Tuple[List[int], List[List[int]]]: The verbalizer indices and the grouped indices.
        """
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
    def combine_logits(logits: Tensor, grouped_indices: List[List[int]]) -> Tensor:
        """Combine Logits.

        Combine the logits for different class labels by taking the arithmetic mean of the logits
        for each class label.

        Args:
            logits (tensor): The logits to be combined.

        Returns:
            tensor: The combined logits.
        """
        return torch.stack(
            [
                torch.stack([torch.sum(e[idx] / len(idx)) for idx in grouped_indices])
                for e in logits
            ]
        )

    def _predicted_indices_to_labels(self, predicted: Tensor) -> Tensor:
        """Predicted Indices to Labels.

        Convert the predicted indices to labels if the verbalizer dictionary is available.

        Args:
            predicted (tensor): The predicted indices.

        Returns:
            tensor: The predicted labels.
        """
        if self.verbalizer_dict is not None:
            verb_kes_list: List[Union[int, str]] = list(self.verbalizer_dict.keys())
            predicted = tensor([verb_kes_list[idx.item()] for idx in predicted])
        return predicted

    def calibrate_output(
        self, output: LLM4ClassificationOutput
    ) -> LLM4ClassificationOutput:
        """Calibrate Output.

        Calibrate the obtained output. Method takes the `LLM4ClassificationOutput` object and calibrates the
        distribution. The predictions are also updated to the calibrated distribution. The type of the output
        is kept the same as the input.

        Args:
            output (LLM4ClassificationOutput): The output logits.

        Returns:
            LLM4ClassificationOutput: The calibrated output logits.
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
        predictions: tensor = torch.argmax(distribution, dim=-1)
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
        """Forward Function.

        Perform the forward pass of the model and return the logits.

        Args:
            batch (Dict[str, tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            kwargs: Additional arguments for the model.

        Raises:
            NotImplementedError: If the forward function is not implemented in the subclass.

        Returns:
            Union[tensor, Tuple[tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
        """
        raise NotImplementedError("Forward function must be implemented in subclass.")

    def _prepare_output(
        self, output: Tensor, return_type: str, predict_labels: bool
    ) -> Union[
        Tensor, np.ndarray, List[Union[str, float, int]], pd.DataFrame, pl.DataFrame
    ]:
        """Prepare Output for Desired Return Type.

        Args:
            output (tensor): The output to be prepared. Can be the predicted tensor or the distribution tensor.
            return_type (str): The return type (Supported types are "list", "torch", "numpy", "pandas" and "polars").
            predict_labels (bool): A flag to determine the output tensor is already the tensor with the predicted labels
                or the distribution tensor.

        Returns:
            Union[tensor, np.ndarray, List[Union[str, float, int]], pd.DataFrame, pl.DataFrame]: The prepared output.
        """
        if return_type == "torch":
            return output
        elif return_type == "numpy":
            return output.numpy()
        elif return_type == "list":
            return output.tolist()
        elif return_type == "polars":
            if self.verbalizer_dict is not None:
                return pl.DataFrame(
                    output.numpy(),
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
                return pd.DataFrame(
                    output.numpy(),
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
        calibrate: bool,
        temperature: float,
        **kwargs: Any,
    ) -> LLM4ClassificationOutput:
        """Smart Forward.

        Smart batch dataset and predict. Return the results in the requested format.


        Args:
            dataset (Dataset): The dataset to be classified.
            batch_size (int): The batch size to be used.
            return_logits (bool): A flag to determine if the logits should be returned.
            show_progress_bar (bool): A flag to determine if the progress bar should be shown.
            return_type (str): The return type. Defaults to "torch". Supported types are "list",
                "torch", "numpy", "pandas" and "polars".
            predict_labels (bool): A flag to determine if the labels (argmax) should be returned.
            calibrate (bool): A flag to determine if the logits should be calibrated.
            temperature (float): The temperature to be used. Defaults to 1.0.
            kwargs: Additional arguments for the forward function (the model).

        Returns:
            List[tensor]: The output logits.
        """
        length_sorted_idx = np.argsort([-len(e) for e in dataset])
        dataset = dataset.select(length_sorted_idx)
        collector: List[Tensor] = []

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
                output: Tensor = self.forward(batch, **kwargs)
                output = output.detach().cpu()
                output = self.combine_logits(output, self.grouped_indices)
                if not return_logits:
                    if temperature != 1.0:
                        output = torch.nn.functional.softmax(
                            output / temperature, dim=-1
                        )
                    else:
                        output = torch.nn.functional.softmax(output, dim=-1)
                collector.extend(output)

        self.model.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output = torch.stack([collector[idx] for idx in np.argsort(length_sorted_idx)])
        if calibrate:
            output = calibrate_fn(output)

        predicted = torch.argmax(output, dim=-1)
        predicted = self._predicted_indices_to_labels(predicted)

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
        calibrate: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Union[LLM4ClassificationOutput, Dict[str, LLM4ClassificationOutput]]:
        """Classify Data.

        Classify the data and return the results in the requested format. This method is used to prepare the data
        according to the provided input format.
        Before inference, model is set into eval() mode and later reset to train() mode.

        Args:
            data (Union[Dataset, Any]): The data to be classified.
            batch_size (int): The batch size to be used. Defaults to 64.
            show_progress_bar (bool): A flag to determine if the progress bar should be shown. Defaults to False.
            return_logits (bool): A flag to determine if the logits should be returned. Defaults to False.
            return_type (str): The return type. Defaults to "torch". Supported types are "list",
                "torch", "numpy", "pandas" and "polars".
            calibrate (bool): A flag to determine if the logits should be calibrated. Defaults to False.
            temperature (float): The temperature to be used. Defaults to 1.0.
            kwargs: Additional arguments for the forward function (the model).

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
        temperature = float(temperature)
        assert temperature > 0.0, "Temperature must be greater than 0."

        if isinstance(data, Dataset):
            return self._smart_forward(
                data,
                batch_size,
                return_logits,
                show_progress_bar=show_progress_bar,
                return_type=return_type,
                calibrate=calibrate,
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
                    calibrate=calibrate,
                    temperature=temperature,
                    **kwargs,
                )
                return_dict[key] = results
            return return_dict
        else:
            raise ValueError("Data must be of type Dataset or DatasetDict.")


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
        model_args: Optional[Dict[str, Any]] = None,
        tok_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Class.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt (Prompt): A prompt object. Example usage:
                ```Txt("This text ") + Key('text') + Txt(" is ") + Vbz([['good'], ['bad']])```
                or
                ```FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))```
                WARNING: Using FVP can result in indexing errors as automatic truncation is not applied.
            device (Optional[str]): The device to be used. Defaults to None.
            lower_verbalizer (bool): A flag to determine if the verbalizer should be enhanced with lowercased words.
                Defaults to False.
            truncate (bool): A flag to determine if the prompt should be truncated. Defaults to True.
            model_args (Optional[Dict[str, Any]]): Additional arguments for initializing the underlying huggingface-model.
            tok_args (Optional[Dict[str, Any]]): Additional arguments for initializing the underlying huggingface-model.
            **kwargs: Additional arguments for initializing the underlying huggingface-model.
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
            truncate=truncate,
        )

    def forward(
        self,
        batch: Dict[str, tensor],
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ModelOutput]]:
        """Forward Function.

        Perform the forward pass of the model and return the logits.

        Args:
            batch (Dict[str, tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            kwargs: Additional arguments for the model.

        Returns:
            Union[tensor, Tuple[tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
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
        model_args: Optional[Dict[str, Any]] = None,
        tok_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Class.

        Args:
            model_id (str): Valid model identifier for huggingface.co.
            prompt (Prompt): A prompt object. Example usage:
                ```Txt("This text ") + Key('text') + Txt(" is ") + Vbz([['good'], ['bad']])```
                or
                ```FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))```
                WARNING: Using FVP can result in indexing errors as automatic truncation is not applied.
            device (Optional[str]): The device to be used. Defaults to None.
            lower_verbalizer (bool): A flag to determine if the verbalizer should be enhanced with lowercased words.
                Defaults to False.
            truncate (bool): A flag to determine if the prompt should be truncated. Defaults to True.
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
            truncate=truncate,
        )

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ModelOutput]]:
        """Forward Function.

        Perform the forward pass of the model and return the logits.

        Args:
            batch (Dict[str, tensor]): The input batch.
            return_model_output (bool): A flag to determine if the model output should be returned.
            kwargs: Additional arguments for the model.

        Returns:
            Union[tensor, Tuple[tensor, Any]]: Output logits or output logits and output from model (if `return_model_output` is set).
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits: Optional[Tensor] = None

        outputs: ModelOutput = self.model(**batch, **kwargs)
        logits = outputs.logits[:, -1, self.verbalizer_indices]

        if return_model_output:
            return logits, outputs
        else:
            return logits
