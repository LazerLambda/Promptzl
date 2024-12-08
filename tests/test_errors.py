import os
import sys

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import promptzl
from promptzl import *

model_id_gen = "sshleifer/tiny-gpt2"
model_id_mlm = "nreimers/BERT-Tiny_L-2_H-128_A-2"

def test_vbz_init_errors():
    
    with pytest.raises(AssertionError):
        Vbz([["bad"], "good", "wonderful", "great"])
    
    with pytest.raises(AssertionError):
        Vbz([[2],[ "good", "wonderful", "great"]])


def test_tokenizer_wo_mask_mlm():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    model = AutoModelForMaskedLM.from_pretrained(model_id_mlm)
    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm, clean_up_tokenization_spaces=True)
    tokenizer.mask_token_id = None
    with pytest.raises(ValueError):
        LLM4ClassificationBase(model=model, tokenizer=tokenizer, prompt=prompt, generate=False)

def test_multiple_subwords_warning():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad worse", "horrible"], ["good"]])
    with pytest.warns():
        MaskedLM4Classification(model_id_mlm, prompt)

def test_forward_base_error():
    prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
    model = AutoModelForCausalLM.from_pretrained(model_id_gen)
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
    test = LLM4ClassificationBase(model=model, tokenizer=tokenizer, prompt=prompt, generate=True)
    with pytest.raises(NotImplementedError):
        test.forward({'test': torch.tensor([1, 2, 3])})

def test_temp_greater_zero_error():
    prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": ["sample_data"] * 3})
    with pytest.raises(AssertionError):
        model.classify(dataset, temperature=0)

def test_wrong_type_classify_error():
    prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt
    )
    with pytest.raises(ValueError):
        model.classify(["This will be allowed in a later version :)"])

def test_vbz_causal_error():
    prompt = Txt("This is a test ") + Vbz([["bad"], ["good", "wonderful", "great"]]) + Key("text")
    with pytest.raises(AssertionError):
        promptzl.CausalLM4Classification(
            model_id_gen,
            prompt=prompt
        )

def test_not_one_verbalizer_error():
    prompt = Txt("This is a test ") + Key("text")
    with pytest.raises(ValueError):
        prompt._get_verbalizer()

    prompt = Txt("This is a test ") + Vbz([["bad"], ["good", "wonderful", "great"]]) + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
    with pytest.raises(ValueError):
        prompt._get_verbalizer()

def test_fvp_error():
    prompt = FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))
    with pytest.raises(NotImplementedError):
        prompt.__fn_str__(AutoTokenizer.from_pretrained(model_id_mlm, clean_up_tokenization_spaces=True))
    
    with pytest.raises(ValueError):
        FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]])) + Txt("asdf")

    with pytest.raises(ValueError):
        Txt("asdf") + FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))

def test_fvp_input_lenght_error():
    prompt = FVP(lambda e: f"{e['text']} It was ", Vbz([["bad", "horrible"], ["good"]]))
    model = promptzl.CausalLM4Classification(
        model_id_gen,
        prompt=prompt
    )

    dataset = Dataset.from_dict({"text": ["a " * 10000 + "a"]})
    with pytest.raises(ValueError):
        model.classify(dataset)

def test_empty_verb_labels_errors():
    prompt = Txt("") + Vbz([[""], ["good", "wonderful", "great"]]) + Key("text")
    with pytest.raises(AssertionError):
        promptzl.CausalLM4Classification(
            model_id_gen,
            prompt=prompt
        )

    prompt = Txt("") + Vbz({0: [""], 1: ["good", "wonderful", "great"]}) + Key("text")
    with pytest.raises(AssertionError):
        promptzl.CausalLM4Classification(
            model_id_gen,
            prompt=prompt
        )

    prompt = Txt("") + Vbz([[""], ["good", "wonderful", "great"]]) + Key("text")
    with pytest.raises(AssertionError):
        promptzl.MaskedLM4Classification(
            model_id_mlm,
            prompt=prompt
        )

    prompt = Txt("") + Vbz({0: [""], 1: ["good", "wonderful", "great"]}) + Key("text")
    with pytest.raises(AssertionError):
        promptzl.MaskedLM4Classification(
            model_id_mlm,
            prompt=prompt
        )

def test_error_prompt_too_long():
    prompt = Txt("a " * 2000) + Key("text") + Vbz([["bad"], ["good"]])


    with pytest.raises(ValueError):
        promptzl.MaskedLM4Classification(
            model_id_mlm,
            prompt=prompt
        )

    with pytest.raises(ValueError):
        promptzl.CausalLM4Classification(
            model_id_gen,
            prompt=prompt
        )