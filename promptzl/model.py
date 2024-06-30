"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import tensor
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .pattern import Pattern


class LLM4ForPatternExploitationClassification(torch.nn.Module):
    """Class for Pattern Exploitatoin Classificatoin.

    This class handles the underlying model and the inference mechanism. It aims to tranform a LLM into a
    classifier.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,  # TODO Check types
        verbalizer: List[List[str]],
        generate: bool,
        prompt: Optional[Pattern] = None,
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
        self.verbalizer_raw: List[List[str]] = verbalizer
        self.use_pattern: bool = prompt is not None

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model: PreTrainedModel = model

        self._can_generate: bool = generate
        if not self._can_generate:
            if self.tokenizer.mask_token_id is None:
                raise ValueError(
                    "The tokenizer does not have a mask token. Please use a model that supports masked language modeling."
                )

        if not self._can_generate:
            if not hasattr(self.tokenizer, "mask_token_id"):
                raise ValueError(
                    "The tokenizer does not have a mask token. Please use a model that supports masked language modeling."
                )  # TODO test this case

        if self.use_pattern:
            self.prompt: Optional[Pattern] = prompt
            self.prompt_tok: Any = self.prompt.tokenize(self.tokenizer)

        self.verbalizer_tok, self.i_dict = self._get_verbalizer(
            verbalizer, self.tokenizer
        )

        self.calibration_logits: Optional[torch.tensor] = None

    # def _load_model(self, model_id: str, **kwargs) -> PreTrainedModel:
    #     model: Optional[PreTrainedModel] = None
    #     try:
    #         model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    #     except:
    #         pass
    #         # TODO: logging?

    #     if model is not None:
    #         return model

    #     try:
    #         model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
    #     except Exception as e:
    #         raise ValueError(f"Model {model_id} is not supported! Error:\n{e}")
    #     return model

    def calibrate(self, support_set: Any) -> None:  # TODO: Add detailed description.
        """Calibrate the model.

        Calibration based on https://aclanthology.org/2022.acl-long.158/. Get the distribution from
        the distribution. After obtaining the distribution, a flag will be set to calibrate during
        inference. Function taken form [OpenPrompt])().  TODO: Finde URL

        :param support_set: The support set to be used for calibration.
        """
        # TODO
        all_logits: List[torch.tensor] = []
        self.model.eval()
        for batch in support_set:
            batch = {k:v.to(self.model.device) for k, v in batch.items()}
            logits: torch.tensor = self.forward(batch)
            all_logits.append(logits.detach())
        all_logits = torch.cat(all_logits, dim=0)
        self.calibration_logits = all_logits.mean(dim=0)

    def _get_verbalizer(
        self, verbalizer: List[List[str]], tokenizer: Any
    ) -> Tuple[List[List[List[List[int]]]]]:
        """Prepare verbalizer.

        Preprocess the verbalizer to be used in the model. The verbalizer is tokenized and the indexes are stored in a dictionary.
        The indices are further necessary to obtain the logits from the models output.

        :param verbalizer: The verbalizer to be used.
        :param tokenizer: The tokenizer to be used.
        :return: The tokenized verbalizer and the dictionary with the indexes.
        """
        verbalizer_tok: List[List[List[List[int]]]] = list(
            map(
                lambda elem: [
                    tokenizer(e, add_special_tokens=False)["input_ids"][0][0]
                    for e in elem
                ],
                [[[elem] for elem in e] for e in verbalizer],
            )
        )
        counter = 0
        i_dict: Dict[Any, Any] = {}
        for i, e in enumerate(verbalizer):
            i_dict[i] = []
            for _ in e:
                i_dict[i].append(counter)
                counter += 1
        verbalizer_tok_seq: List[int] = [
            item for innerlist in verbalizer_tok for item in innerlist
        ]
        assert len(set(verbalizer_tok_seq)) == len(
            verbalizer_tok_seq
        ), "Equivalent tokens for different classes detected! This also happens if subwords are equal. Tokens must be unique for each class!"
        return verbalizer_tok, i_dict

    def _class_probs(
        self, scores: Any, verbalizer_tok: List[List[List[List[int]]]]
    ) -> tensor:  # TODO: maybe add i_dict or in inference method
        """Get the class probabilities.

        Get the class probabilities from the logits. The logits are transformed into probabilities using the softmax function
        based on the indices.

        :param scores: The logits from the model.
        :param verbalizer_tok: The tokenized verbalizer.
        :return: The class probabilities.
        """
        out_res: torch.Tensor = torch.cat(
            list(
                map(lambda i: scores[:, verbalizer_tok[i]], range(len(verbalizer_tok)))
            ),
            axis=-1,
        )
        if self.calibration_logits is not None:
            pass  # TODO
        out_res = torch.nn.functional.softmax(out_res, dim=1)
        return out_res
        # TODO: verbalizer for labels
        # class_probs_combined: Dict[str, torch.Tensor] = {k:torch.sum(out_res[:, v], axis=-1) for k, v in i_dict.items()}
        # return class_probs_combined

    def forward(self, batch: Dict[str, tensor], return_model_output: bool = False) -> Union[tensor, Tuple[tensor, Any]]:
        """Forward pass.

        Perform the forward pass of the model. The model generates the output based on the input batch.

        :param batch: The input batch.
        :return: The class probabilities.
        """
        if self._can_generate:
            if self.use_pattern:
                pass  # TODO
                # batch = self.prompt.get_prompt_batch(batch)
                # # TODO Temperature
                # outputs = self.model.generate(
                #     input_ids=batch,
                #     output_scores=True,
                #     return_dict_in_generate=True,
                #     max_new_tokens=2,
                # )
                # if return_model_output:
                #     return outputs, outputs
                # return outputs
            else:
                outputs: GenerateDecoderOnlyOutput = self.model.generate(
                    **batch,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,  # TODO temperature
                )
                probs: tensor = self._class_probs(
                    outputs.scores[0].cpu(), self.verbalizer_tok
                )
                if return_model_output:
                    return probs, outputs
                else:
                    return probs
        else:
            mask_index = torch.where(
                batch["input_ids"] == self.tokenizer.mask_token_id
            )[1]
            assert (
                mask_index.shape[0] == batch["input_ids"].shape[0]
            ), "Mask token not found in input!"
            outputs = self.model(**batch)
            logits = outputs.logits[range(mask_index.shape[0]), mask_index].cpu()
            probs = self._class_probs(logits, self.verbalizer_tok)
            return probs

    def __del__(self):
        """Delete the model."""
        del self.model
        torch.cuda.empty_cache()
