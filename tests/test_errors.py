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


def test_polars_pandas_warning_no_dict():

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": ["sample_data"] * 3})
    with pytest.warns(UserWarning):
        output = model.classify(dataset, use_dataset_keys_in_results=True, return_type="polars")
        assert output.columns == ['bad', 'good']

    with pytest.warns(UserWarning):
        output = model.classify(dataset, use_dataset_keys_in_results=True, return_type="pandas")
        assert output.columns.to_list() == ['bad', 'good']

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
        