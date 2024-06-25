
from typing import Any, Dict, List

class Placeholder:
    def __init__(self, text: str):
        self.text: str = text

    def tokenize(self, tokenizer):
        return tokenizer(self.text)['input_ids']


class DataKey(Placeholder):
    def __init__(self, text: str):
        super().__init__(text)

    def tokenize(self, *args, **kwargs):
        return []


class Prompt(Placeholder):
    def __init__(self, text: str, autospaces: bool = True):
        if autospaces:
            if text[-1] != " ":
                text = text + " "
            if text[0] != " ":
                text = " " + text
        super().__init__(text)


class Mask(Placeholder):    
    def __init__(self):
        super().__init__("MASK")
    
    def tokenize(self, tokenizer):
        if tokenizer.mask_token_id is not None:
            return [tokenizer.mask_token_id]
        else:
            return []


class Pattern:

    def __init__(self, pattern: List[Placeholder]):
        assert len(list(filter(lambda x: isinstance(x, Mask), pattern))) <= 1, "Exactly one mask token must be present in the pattern!"
        self.pattern: List[Placeholder] = pattern
        self.prompt_len: int = -1
        self.prompt_tokenized: Any = []
    

    def tokenize(self, tokenizer: callable):
        self.prompt_tokenized: Any = [e.tokenize(tokenizer) for e in self.pattern]
        self.prompt_len = sum(map(len, self.prompt_tokenized))
        return self.prompt_tokenized
    

    def get_prompt_single(self, elem: Dict[str, Any]):
        return [item for sublist in list(map(lambda e: elem[e[0].text] if isinstance(e[0], DataKey) else e[1], zip(self.pattern, self.prompt_tokenized))) for item in sublist]
    

    def get_prompt_batch(self, batch: Any):
        return [self.get_prompt_single(e) for e in batch]