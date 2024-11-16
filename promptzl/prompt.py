"""Promptzl, 2024.

MIT LICENSE
"""

from typing import List


class Prompt:
    def __init__(self, collector: list):
        self.collector = collector

    def __add__(self, other):
        self.collector.append(other)
        return Prompt(self.collector)

    def __str__(self):
        return "".join([str(e) for e in self.collector])

    def __fn_str__(self, tokenizer):
        return "".join([e.__fn_str__(tokenizer) for e in self.collector])

    def __repr__(self):
        return self.__str__()

    def prompt_fun(self, tokenizer):
        return lambda e: self.__fn_str__(tokenizer) % e


class Txt(Prompt):
    def __init__(self, text: str = " "):
        self.text: str = text
        super().__init__([self])

    def __str__(self):
        return self.text

    def __fn_str__(self, *args):
        return self.text

    def __repr__(self):
        return self.__str__()


class TKy(Prompt):
    def __init__(self, key: str = "text"):
        self.key: str = key
        super().__init__([self])

    def __str__(self):
        return f"<{self.key}>"

    def __fn_str__(self, *args):
        return "%s"

    def __repr__(self):
        return self.__str__()


class IKy(Prompt):
    def __init__(self, key: str):
        self.key: str = key
        super().__init__([self])

    def __str__(self):
        return f"[{self.key}]"

    def __fn_str__(self, *args):
        return f"[IMAGE]"

    def __repr__(self):
        return self.__str__()


class Vbz(Prompt):
    def __init__(self, verbalizer: List[List[str]]):
        self.verbalizer: List[List[str]] = verbalizer
        super().__init__([self])

    def __str__(self):
        # return f"[{','.join([f"['{str(e[0])}',...]" for e in self.verbalizer])}]"
        return (
            "<Vbz: ["
            + ", ".join(['["%s",...]' % str(elem[0]) for elem in self.verbalizer])
            + "]>"
        )

    def __fn_str__(self, tokenizer):
        return f"{tokenizer.mask_token}"

    def __repr__(self):
        return self.__str__()
