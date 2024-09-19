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


class TestPrompt():


    def test_basic_initialization(self):
        Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))

    def test_string_representation(self):
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        assert len(str(prompt)) > 0
        assert len(prompt.__repr__()) > 0
    
    def test_subinit_causal(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2')
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=True)

    def test_subinit_masked(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2')
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=False)

    def test_get_text_masked(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2')
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=False)
        assert prompt.get_text({'a': 'Test'}) == f"Test Test {tokenizer.mask_token}"

    def test_get_text_causal(self):
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))
        prompt.subinit(tokenizer, generate=True)
        assert prompt.get_text({'a': 'Test'}) == f"Test Test "

    def test_get_text_masked_error(self):
        with pytest.raises(AssertionError):
            Prompt(Text('Test'), None, Verbalizer([['bad'], ['good']]))

        with pytest.raises(AssertionError):
            Prompt(Text('Test'), Key('a'))

        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2')
        prompt = Prompt(Text('Test'), Key('a'), Verbalizer([['bad'], ['good']]))


