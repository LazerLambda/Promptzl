import operator

from datasets import Dataset
from functools import reduce
from torch import tensor
from typing import Any, Dict, List, Union, Optional, Tuple
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from .prompt import Prompt, TKy, IKy, Vbz, Txt


class SystemPrompt:

    def __init__(self, prompt: Prompt, tokenizer: PreTrainedTokenizerFast, truncate: bool = True, mlm: bool = True):
        self.prompt: Prompt = prompt
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.truncate: bool = truncate
        self.mlm: bool = mlm


        assert isinstance(prompt, Prompt), "`prompt` must be of type Prompt."
        assert isinstance(tokenizer, PreTrainedTokenizerFast), "`tokenizer` must be of type PreTrainedTokenizer."
        assert isinstance(truncate, bool), "`truncate` must be of type bool."
        assert isinstance(mlm, bool), "`mlm` must be of type bool."

        # TODO: Check if prompt includes key
        # TODO: Check if vbz is at the end for causal
        try:
            self.verbalizer: Vbz = [e for e in self.prompt.collector if isinstance(e, Vbz)][0]
        except:
            raise ValueError(f"No verbalizer found in prompt:\n\t'-> {str(prompt)}")
        
        if len([e for e in self.prompt.collector if isinstance(e, (TKy, IKy))]) < 1:
            raise ValueError(f"No key found in prompt. Please provide a `TKy` key!\n\t'-> {str(prompt)}")
        
        self.intermediate_token = None
        if mlm:
            if self.tokenizer.mask_token_id is None or not hasattr(
                self.tokenizer, "mask_token_id"
            ):
                raise ValueError("Tokenizer does not have a mask token. Please provide a tokenizer with a mask token.")
        
        # TODO: TEST THIS and check if subsequent token is also important.
        for i in range(0, len(self.prompt.collector) - 1):
            if isinstance(self.prompt.collector[i], (Txt)) and isinstance(self.prompt.collector[i + 1], Vbz):
                self.intermediate_token = self.prompt.collector[i].text[-1]

        prefix_suffix: Tuple[List[int], List[int]] = self.get_prefix_suffix()
        self.prefix: List[int] = prefix_suffix[0]
        self.suffix: List[int] = prefix_suffix[1]
        n_pre_suffix: int = 0

        if len(self.prefix) > 0:
            n_pre_suffix += 1

        if len(self.suffix) > 0:
            n_pre_suffix += 1

        self.dict_ds: Dict[str, Dataset] = {k:None for k in [e.key for e in self.prompt.collector if isinstance(e, (IKy, TKy))]}

        self.template_prmpt: List[Union[List[int], TKy, IKy, Vbz]] = [self._tokenize_txt(e) if isinstance(e, Txt) else e for e in self.prompt.collector]
 
        only_keys: int = len([None for e in self.prompt.collector if isinstance(e, (IKy, TKy))])
        used_tokens: int = int(self.tokenizer.model_max_length - sum([len(e) for e in self.template_prmpt if isinstance(e, list)]) - n_pre_suffix) # Account for prefix and suffix tokens
        if mlm:
            self.max_len_keys: int = int((used_tokens - 1) // only_keys) if only_keys != 0 else used_tokens
        else:
            self.max_len_keys = int(used_tokens // only_keys) if only_keys != 0 else used_tokens


    def _tokenize_txt(self, elem: Any) -> Any:
        return self.tokenizer(elem.text, add_special_tokens=False)['input_ids']

    def _tokenize_data_by_key(self, elem: List[Dict[str, str]], key: str) -> List[List[int]]:
        if self.truncate:
            return self.tokenizer(elem[key], add_special_tokens=False, max_length=self.max_len_keys, truncation=True)['input_ids']
            # return [e[0:self.max_len_keys] for e in self.tokenizer(elem[key], add_special_tokens=False)['input_ids']]
        else:
            return self.tokenizer(elem[key], add_special_tokens=False)['input_ids']


    def get_prefix_suffix(self):
        ex = list(set(self.tokenizer.vocab.keys()) - set(self.tokenizer.all_special_tokens))[-1]
        ex_id = self.tokenizer.vocab[ex]
        example = self.tokenizer(ex)['input_ids']

        prefix = []
        suffix = []
        found = False
        for e in example:
            if not found and e != ex_id and len(prefix) == 0:
                prefix = [e]
            if e == ex_id:
                found = True
                continue
            if found:
                suffix = [e]
                break
        return prefix, suffix


    def __str__(self):
        return str(self.prompt)

    def prepare_and_tokenize_dataset(self, data: Dict[str, List[Union[str, Any]]]) -> List[List[int]]:

        n: int = len(data[list(data.keys())[0]])

        # Prepare prompts in batch
        tknzd_prmpt: List[Union[List[int], TKy, IKy, Vbz]] = [[e] * n if not isinstance(e, (Vbz, IKy, TKy)) else e for e in self.template_prmpt]

        # Tokenize data by key # TODO: max lengths
        tknzd_prmpt = [self._tokenize_data_by_key(data, e.key) if isinstance(e, (TKy, IKy)) else e for e in tknzd_prmpt]

        # Add Mask if necessary TODO: different case for autoregressive decoding
        if self.mlm:
            tknzd_prmpt = [[[self.tokenizer.mask_token_id]] * n if isinstance(e, Vbz) else e for e in tknzd_prmpt]
        else:
            tknzd_prmpt = [e for e in tknzd_prmpt  if not isinstance(e, Vbz)]

        # Add Prefix and Suffix
        tknzd_prmpt = [[self.prefix] * n] + tknzd_prmpt + [[self.suffix] * n]

        # Remove superfluous tokens
        tknzd_prmpt = [e for e in tknzd_prmpt if len(e) > 0]
    
        # Add lists together horizontically
        tknzd_prmpt = [reduce(operator.add, e) for e in zip(*tknzd_prmpt)]

        return tknzd_prmpt
    
    def pad_and_stack(self, data: List[List[int]]) -> Dict[str, tensor]:
        n: int = max([len(e) for e in data])
        if self.mlm:
            input_ids: tensor = tensor([e + [self.tokenizer.pad_token_id] * (n - len(e)) for e in data])
            attention_mask: tensor = tensor([[1] * len(e) + [0] * (n - len(e)) for e in data])
        else:
            input_ids: tensor = tensor([[self.tokenizer.pad_token_id] * (n - len(e)) + e for e in data])
            attention_mask: tensor = tensor([[0] * (n - len(e)) + [1] * len(e) for e in data])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


    def get_tensors(self, data: Dict[str, List[Union[str, Any]]]) -> Dict[str, tensor]:
        tknzd_data: List[List[int]] = self.prepare_and_tokenize_dataset(data)
        return self.pad_and_stack(tknzd_data)
