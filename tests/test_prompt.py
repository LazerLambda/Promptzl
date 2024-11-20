import os
import sys

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
import promptzl
from promptzl import *
from promptzl.utils import SystemPrompt

model_id_gen = "sshleifer/tiny-gpt2"
model_id_mlm = "nreimers/BERT-Tiny_L-2_H-128_A-2"

sample_data = [
    "The pizza was horribe and the staff rude. Won't recommend.",
    "The pasta was undercooked and the service was slow. Not going back.",
    "The salad was wilted and the waiter was dismissive. Avoid at all costs.",
    "The soup was cold and the ambiance was noisy. Not a pleasant experience.",
    "The burger was overcooked and the fries were soggy. I wouldn't suggest this place.",
    "The sushi was not fresh and the staff seemed uninterested. Definitely not worth it.",
    "The steak was tough and the wine was sour. A disappointing meal.",
    "The sandwich was bland and the coffee was lukewarm. Not a fan of this café.",
    "The dessert was stale and the music was too loud. I won't be returning.",
    "The chicken was dry and the vegetables were overcooked. A poor dining experience.",
]

model_id_gen = "sshleifer/tiny-gpt2"
model_id_mlm = "nreimers/BERT-Tiny_L-2_H-128_A-2"


def test_init():
    prompt = Txt("Hello World! ") + Key('test') + Vbz([['good'], ['bad']])
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen)
    SystemPrompt(prompt, tokenizer, mlm=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm)
    SystemPrompt(prompt, tokenizer)

    with pytest.raises(Exception):
        SystemPrompt(None, tokenizer)

    with pytest.raises(Exception):
        SystemPrompt(prompt, None)

    with pytest.raises(Exception):
        SystemPrompt(prompt, tokenizer, truncate=-1)

    with pytest.raises(Exception):
        SystemPrompt(prompt, tokenizer, mlm=-1)
    
    with pytest.raises(Exception):
        prompt = Txt("Hello World! ") + Vbz([['good'], ['bad']])
        SystemPrompt(prompt, tokenizer)

    with pytest.raises(Exception):
        prompt = Txt("Hello World! ") + Key('test')
        SystemPrompt(prompt, tokenizer)

def test_equal_tokens_error():
    pass

def test_multiple_tokens_warning_mlm():
    pass

def test_str_method():
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    assert str(prompt) == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    assert str(prompt) == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"

    systemprompt = SystemPrompt(prompt, AutoTokenizer.from_pretrained(model_id_mlm))
    assert str(systemprompt) == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"

    systemprompt = SystemPrompt(prompt, AutoTokenizer.from_pretrained(model_id_gen), mlm=False)
    assert str(systemprompt) == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"

def test_repr_method():
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    assert prompt.__repr__() == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    assert prompt.__repr__() == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"
    assert ''.join([e.__repr__() for e in prompt.collector]) == "Test <a> [a] <Vbz: [[\"bad\",...], [\"good\",...]]>"


def test_fn_str_method():
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen, padding_side="left")
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm)
    assert isinstance(prompt.__fn_str__(tokenizer), str)

    assert callable(prompt.prompt_fun(tokenizer))


def test_vbz_w_dict():
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz({0: ['bad'], 1: ['good']})
    assert prompt.collector[-1].verbalizer_dict == {0: ['bad'], 1: ['good']}
    assert prompt.collector[-1].verbalizer == [['bad'], ['good']]

    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    assert prompt.collector[-1].verbalizer_dict == None
    assert prompt.collector[-1].verbalizer == [['bad'], ['good']]

    with pytest.raises(ValueError):
        Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz(0)


# import os
# import sys

# import pytest
# import torch
# from datasets import Dataset
# from transformers import AutoModelForMaskedLM, AutoTokenizer

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # Now we can import the promptzel package
# from promptzl import *
# from promptzl.prompt import get_prompt


# # class TestPrompt():


# #     def test_basic_initialization(self):
# #         Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))

# #     def test_string_representation(self):
# #         prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
# #         assert len(str(prompt)) > 0
# #         assert len(prompt.__repr__()) > 0
    
# #     def test_subinit_causal(self):
# #         tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
# #         prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
# #         prompt.subinit(tokenizer, generate=True)

# #     def test_subinit_masked(self):
# #         tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
# #         prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
# #         prompt.subinit(tokenizer, generate=False)

# #     def test_get_text_masked(self):
# #         tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
# #         prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
# #         prompt.subinit(tokenizer, generate=False)
# #         assert prompt.get_text({'a': 'Test'}) == f"Test Test {tokenizer.mask_token}"

# #     def test_get_text_causal(self):
# #         tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', clean_up_tokenization_spaces=True)
# #         prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
# #         prompt.subinit(tokenizer, generate=True)
# #         assert prompt.get_text({'a': 'Test'}) == f"Test Test "

# #     def test_get_text_masked_error(self):
# #         with pytest.raises(AssertionError):
# #             Prompt(Text('Test'), None, Verbalizer([['bad'], ['good']]))

# #         with pytest.raises(AssertionError):
# #             Prompt(Text('Test'), Key('a'))

# #         # TODO: What?
# #         tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
# #         prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))



# #     # TODO: Test dataset['test'] with only Verbalizer -> Error

# # def test_get_prompt_init():
# #     prompt = get_prompt("asd %s askl %m ödj %s", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])
# #     assert isinstance(prompt, Prompt)
# #     prompt = get_prompt("asd %s asklödj %s", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])
# #     assert isinstance(prompt, Prompt)
# #     prompt = get_prompt("asd %s asklödj", "text", verbalizer=[["<VERBALIZER>"]])
# #     assert isinstance(prompt, Prompt)
    

# # def test_get_prompt_mask_plcholder_guards():
# #     with pytest.raises(AssertionError):
# #         get_prompt("asd %s askl %m ödj %s askl %m ödj", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])
    
# #     with pytest.raises(AssertionError):
# #         get_prompt("asd %s askl %m ödj %m %s", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])

# # def test_get_prompt_key_list_guards():
# #     with pytest.raises(ValueError):
# #         get_prompt("asd %s asklödj %s", ["", "text_b"], verbalizer=[["<VERBALIZER>"]])
    
# #     with pytest.raises(TypeError):
# #         get_prompt("asd %s asklödj", 0, verbalizer=[["<VERBALIZER>"]])

# # def test_get_prompt_place_holders_guards():
# #     with pytest.raises(AssertionError):
# #         get_prompt("asd %s asklödj", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])

# # def test_addability():
# #     prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init()
# #     prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init(sep=" ")
# #     prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init(truncate_data=True)
# #     prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init(sep=" ", truncate_data=True)
# #     assert str(prompt) == """Text <Data-Key: 'a'> <Verbalizer: [["bad",...], ["good",...]]>"""
