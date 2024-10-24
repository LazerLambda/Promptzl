import os
import sys

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
from promptzl import *
from promptzl.prompt import get_prompt


class TestPrompt():


    def test_basic_initialization(self):
        Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))

    def test_string_representation(self):
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        assert len(str(prompt)) > 0
        assert len(prompt.__repr__()) > 0
    
    def test_subinit_causal(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=True)

    def test_subinit_masked(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=False)

    def test_get_text_masked(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=False)
        assert prompt.get_text({'a': 'Test'}) == f"Test Test {tokenizer.mask_token}"

    def test_get_text_causal(self):
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2', clean_up_tokenization_spaces=True)
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=True)
        assert prompt.get_text({'a': 'Test'}) == f"Test Test "

    def test_get_text_masked_error(self):
        with pytest.raises(AssertionError):
            Prompt(Text('Test'), None, Verbalizer([['bad'], ['good']]))

        with pytest.raises(AssertionError):
            Prompt(Text('Test'), Key('a'))

        # TODO: What?
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))



    # TODO: Test dataset['test'] with only Verbalizer -> Error

def test_get_prompt_init():
    prompt = get_prompt("asd %s askl %m ödj %s", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])
    assert isinstance(prompt, Prompt)
    prompt = get_prompt("asd %s asklödj %s", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])
    assert isinstance(prompt, Prompt)
    prompt = get_prompt("asd %s asklödj", "text", verbalizer=[["<VERBALIZER>"]])
    assert isinstance(prompt, Prompt)
    

def test_get_prompt_mask_plcholder_guards():
    with pytest.raises(AssertionError):
        get_prompt("asd %s askl %m ödj %s askl %m ödj", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])
    
    with pytest.raises(AssertionError):
        get_prompt("asd %s askl %m ödj %m %s", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])

def test_get_prompt_key_list_guards():
    with pytest.raises(ValueError):
        get_prompt("asd %s asklödj %s", ["", "text_b"], verbalizer=[["<VERBALIZER>"]])
    
    with pytest.raises(TypeError):
        get_prompt("asd %s asklödj", 0, verbalizer=[["<VERBALIZER>"]])

def test_get_prompt_place_holders_guards():
    with pytest.raises(AssertionError):
        get_prompt("asd %s asklödj", ["text_a", "text_b"], verbalizer=[["<VERBALIZER>"]])

def test_addability():
    prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init()
    prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init(sep=" ")
    prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init(truncate_data=True)
    prompt = (Text('Text') + Key('a') + Verbalizer([['bad'], ['good']])).init(sep=" ", truncate_data=True)
    assert str(prompt) == """Text <Data-Key: 'a'> <Verbalizer: [["bad",...], ["good",...]]>"""
