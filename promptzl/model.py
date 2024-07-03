"""Promptzl, 2024.

MIT LICENSE
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import tensor
from torch.utils.data import DataLoader
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
            assert self.prompt is not None, "Prompt is None!"
            self.prompt_tok: Any = (
                self.prompt.tokenize(self.tokenizer)
                if self.prompt is not None
                else None
            )

        self.verbalizer_tok, self.i_dict = self._get_verbalizer(verbalizer)
        self.calibration_probs: Optional[torch.tensor] = None

    def set_contextualized_prior(self, support_set: DataLoader) -> None:
        """Compute Contextualized Prior.

        Compute the contextualized prior form equation (2) based on [Hu et al., 2022](https://aclanthology.org/2022.acl-long.158/).
        A support set is used to obtain the logits of the model for the labels. The logits are then averaged and normalized (softmax) to obtain
        the prior.Function taken from [OpenPrompt])(https://thunlp.github.io/OpenPrompt/_modules/openprompt/utils/calibrate.html#calibrate) and
        adapted for our framework.

        :param support_set: The support set to be used for calibration.
        """
        # TODO
        all_logits: List[torch.tensor] = []
        self.model.eval()
        for batch in support_set:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            logits: torch.tensor = self.forward(batch, combine=False)
            all_logits.append(logits.detach())
        all_logits_combined: torch.tensor = torch.cat(all_logits, dim=0)
        all_logits_combined = all_logits_combined.mean(dim=0)
        self.calibration_probs = torch.nn.functional.softmax(
            all_logits_combined, dim=-1
        )

    # def calibrate(self, labels_logits: torch.tensor) -> torch.tensor:
    #     """Calibrate the logits.

    #     Calibrate the logits based on the contextualized prior. The logits are normalized (softmax) and multiplied by the prior.

    #     :param labels_logits: The logits to be calibrated.
    #     :return: The calibrated logits.
    #     """
    #     assert self.calibration_probs is not None, "Calibration logits not set!"
    #     return labels_logits

    def _get_verbalizer(
        self, verbalizer: List[List[str]]
    ) -> Tuple[List[List[int]], Dict[str, List[int]]]:
        """Prepare verbalizer.

        Preprocess the verbalizer to be used in the model. The verbalizer is tokenized and the indexes are stored in a dictionary.
        The indices are further necessary to obtain the logits from the models output.

        :param verbalizer: The verbalizer to be used.
        :return: The tokenized verbalizer and the dictionary with the indexes.
        """
        verbalizer_tok: List[List[int]] = list(
            map(
                lambda elem: [
                    self.tokenizer(e, add_special_tokens=False)["input_ids"][0][0]
                    for e in elem
                ],
                [[[elem] for elem in e] for e in verbalizer],
            )
        )

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
        print(i_dict)
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
            out_res = out_res / self.calibration_probs
            out_res = torch.nn.functional.softmax(out_res, dim=1)
        # TODO: Sum multiple tokens together
        if combine:
            out_res = torch.transpose(
                torch.stack(
                    [torch.sum(out_res[:, v], axis=-1) for v in self.i_dict.values()]
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
                    **kwargs,
                )
                logits = outputs.scores[0].detach().cpu()
        else:
            mask_index = torch.where(
                batch["input_ids"] == self.tokenizer.mask_token_id
            )[1]
            assert (
                mask_index.shape[0] == batch["input_ids"].shape[0]
            ), "Mask token not found in input!"
            outputs = self.model(**batch)
            logits = (
                outputs.logits[range(mask_index.shape[0]), mask_index].detach().cpu()
            )

        probs: tensor = self._class_probs(logits, combine=combine)
        if return_model_output:
            return probs, outputs
        else:
            return probs

    def __del__(self):
        """Delete the model."""
        del self.model
        torch.cuda.empty_cache()
