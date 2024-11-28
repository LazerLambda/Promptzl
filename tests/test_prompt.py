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

    assert callable(prompt._prompt_fun(tokenizer))


def test_vbz_w_dict():
    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz({0: ['bad'], 1: ['good']})
    assert prompt.collector[-1].verbalizer_dict == {0: ['bad'], 1: ['good']}
    assert prompt.collector[-1].verbalizer == [['bad'], ['good']]

    prompt = Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz([['bad'], ['good']])
    assert prompt.collector[-1].verbalizer_dict == None
    assert prompt.collector[-1].verbalizer == [['bad'], ['good']]

    with pytest.raises(ValueError):
        Txt('Test ') + Key('a') + Txt(" ") + Img('a') + Txt(" ") + Vbz(0)


def test_fpv():
    prompt = FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))
    assert prompt._prompt_fun()({'text': 'test'}) == "test It was "
    assert str(prompt) == "<FVP>"
    assert prompt.__repr__() == "<FVP>"