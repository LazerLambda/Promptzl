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

def test_batch_padding_mlm():
    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm)
    as_should = tokenizer([e + str(tokenizer.mask_token) for e in sample_data], padding=True, truncation="longest_first", return_tensors="pt")
    prompt = Key("text") + Vbz([["bad", "horrible"], ["good"]])
    systemprompt = SystemPrompt(prompt, tokenizer)
    output = systemprompt.get_tensors({'text': sample_data})
    assert torch.equal(output['input_ids'], as_should['input_ids'])

def test_batch_padding_gen():
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    as_should = tokenizer(sample_data, padding=True, truncation="longest_first", return_tensors="pt")
    prompt = Key("text") + Vbz([["bad", "horrible"], ["good"]])
    systemprompt = SystemPrompt(prompt, tokenizer, mlm=False)
    output = systemprompt.get_tensors({'text': sample_data})
    assert torch.equal(output['input_ids'], as_should['input_ids'])

def test_exceeding_length_mlm():
    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm)
    prompt = Key("text") + Vbz([["bad", "horrible"], ["good"]])
    systemprompt = SystemPrompt(prompt, tokenizer)
    with pytest.warns(UserWarning):
        output = systemprompt.get_tensors_fast({'text': ["a " * 10000 + "a"] *4})
    assert output['input_ids'].shape[0] == 4
    assert output['input_ids'].shape[1] <= tokenizer.model_max_length

    prompt = Key("text_a") + Vbz([["bad", "horrible"], ["good"]]) + Key("text_b")
    systemprompt = SystemPrompt(prompt, tokenizer)
    with pytest.warns(UserWarning):
        output = systemprompt.get_tensors_fast({"text_a": ["a " * 10000 + "a"] * 4, "text_b": ["b " * 10000 + "b"] * 4})
    assert output['input_ids'].shape[0] == 4
    assert output['input_ids'].shape[1] <= tokenizer.model_max_length

    prompt = Key("text_a") + Vbz([["bad", "horrible"], ["good"]]) + Key("text_b") + Key("text_c")
    systemprompt = SystemPrompt(prompt, tokenizer)
    with pytest.warns(UserWarning):
        output = systemprompt.get_tensors_fast({
            "text_a": ["a " * 10000 + "a"] * 4,
            "text_b": ["b " * 10000 + "b"] * 4,
            "text_c": ["c " * 10000 + "c"] * 4})
    assert output['input_ids'].shape[0] == 4
    assert output['input_ids'].shape[1] <= tokenizer.model_max_length

def test_exceeding_length_gen():
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = Key("text") + Vbz([["bad", "horrible"], ["good"]])
    systemprompt = SystemPrompt(prompt, tokenizer, mlm=False)
    with pytest.warns(UserWarning):
        output = systemprompt.get_tensors_fast({'text': ["a " * 10000 + "a"] *4})
    assert output['input_ids'].shape[0] == 4
    assert output['input_ids'].shape[1] <= tokenizer.model_max_length

    prompt = Key("text_a") + Vbz([["bad", "horrible"], ["good"]]) + Key("text_b")
    systemprompt = SystemPrompt(prompt, tokenizer, mlm=False)
    with pytest.warns(UserWarning):
        output = systemprompt.get_tensors_fast({"text_a": ["a " * 10000 + "a"] * 4, "text_b": ["b " * 10000 + "b"] * 4})
    assert output['input_ids'].shape[0] == 4
    assert output['input_ids'].shape[1] <= tokenizer.model_max_length

    prompt = Key("text_a") + Vbz([["bad", "horrible"], ["good"]]) + Key("text_b") + Key("text_c")
    systemprompt = SystemPrompt(prompt, tokenizer, mlm=False)
    with pytest.warns(UserWarning):
        output = systemprompt.get_tensors_fast({
            "text_a": ["a " * 10000 + "a"] * 4,
            "text_b": ["b " * 10000 + "b"] * 4,
            "text_c": ["c " * 10000 + "c"] * 4})
    assert output['input_ids'].shape[0] == 4
    assert output['input_ids'].shape[1] <= tokenizer.model_max_length
