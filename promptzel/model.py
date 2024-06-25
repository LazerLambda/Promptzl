from .pattern import Pattern, Prompt, DataKey, Mask
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from typing import Callable, List, Union, Any, Tuple, Dict, Optional
from torch import tensor
import os
import torch



class LLM4ForPatternExploitationClassification(torch.nn.Module):

    # TODO Option no pattern
    def __init__(self, pretrained_model_name_or_path: Union[str, os.PathLike], verbalizer: List[List[str]], prompt: Optional[Pattern] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbalizer_raw: List[List[str]] = verbalizer
        self.use_pattern: bool = prompt is not None

        self.tokenizer: Any = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model: Any = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)

        if self.use_pattern:
            self.prompt: Callable[[str], str] = prompt
            self.prompt_tok: Any = self.prompt.tokenize(self.tokenizer)
        
        self.verbalizer_tok, self.i_dict = self._get_verbalizer(verbalizer, self.tokenizer)
    

    def calibrate(self, support_set: Any):
        # TODO
        pass
 

    def _get_verbalizer(self, verbalizer: List[List[str]], tokenizer: Any) -> Tuple[List[List[List[List[int]]]]]:
        verbalizer_tok: List[List[List[List[int]]]] = list(map(lambda elem: [tokenizer(e, add_special_tokens=False)['input_ids'][0][0] for e in elem], [[[elem] for elem in e] for e in verbalizer]))
        counter = 0
        i_dict: Dict[Any, Any] = {}
        for i, e in enumerate(verbalizer):
            i_dict[i] = []
            for _ in e:
                i_dict[i].append(counter)
                counter += 1
        verbalizer_tok_seq: List[int] = [item for innerlist in verbalizer_tok for item in innerlist]
        assert len(set(verbalizer_tok_seq)) == len(verbalizer_tok_seq), "Equivalent tokens for different classes detected! This also happens if subwords are equal. Tokens must be unique for each class!"
        return verbalizer_tok, i_dict


    def _class_probs(self, scores: Any, verbalizer_tok: List[List[List[List[int]]]], i_dict: Any) -> tensor:
        out_res: torch.Tensor = torch.cat(list(map(lambda i: scores[:, verbalizer_tok[i]], range(len(verbalizer_tok)))), axis=-1)
        out_res = torch.nn.functional.softmax(out_res, dim=1)
        return out_res
        # TODO: verbalizer for labels
        # class_probs_combined: Dict[str, torch.Tensor] = {k:torch.sum(out_res[:, v], axis=-1) for k, v in i_dict.items()}
        # return class_probs_combined


    def forward(self, batch):
        if self.use_pattern:
            batch = self.prompt.get_prompt_batch(batch)
            # TODO Temperature
            outputs = self.model.generate(input_ids=batch, output_scores=True, return_dict_in_generate=True, max_new_tokens=2)
            return outputs
        else:
            outputs: GenerateDecoderOnlyOutput = self.model.generate(**batch, output_scores=True, return_dict_in_generate=True, max_new_tokens=1)
            probs: tensor = self._class_probs(outputs.scores[0].cpu(), self.verbalizer_tok, self.i_dict)
            return probs
            # return {v[0]:probs[:, i] for i, v in enumerate(self.verbalizer_raw)}

    def __del__(self):
        del self.model
        torch.cuda.empty_cache()