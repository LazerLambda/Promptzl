import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from datasets import Dataset
from torch import tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.generation.utils import ModelOutput

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
from promptzl import *

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

def test_combine_function():

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])

    test = CausalLM4Classification("sshleifer/tiny-gpt2", prompt)
    grouped_indices = [[0, 1], [2]]

    combined = LLM4ClassificationBase.group_logits(tensor([[1,3,7], [2,4,8]]), grouped_indices)
    assert torch.all(combined == tensor([[2., 7.], [3., 8.]]))

    combined = LLM4ClassificationBase.group_logits(tensor([[1,3,7], [2,4,8]]), grouped_indices)
    assert torch.all(combined == tensor([[2., 7.], [3., 8.]]))

def test_set_prompt_function():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])

    test = CausalLM4Classification("sshleifer/tiny-gpt2", prompt)
    assert test.verbalizer_raw == [["bad", "horrible"], ["good"]]

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad"], ["good"]])
    test.set_prompt(prompt)
    assert test.verbalizer_raw == [["bad"], ["good"]]

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad"], ["good"], ['neutral']])
    test.set_prompt(prompt)
    assert test.verbalizer_raw == [["bad"], ["good"], ['neutral']]


    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    test = MaskedLM4Classification("nreimers/BERT-Tiny_L-2_H-128_A-2", prompt)
    assert test.verbalizer_raw == [["bad", "horrible"], ["good"]]

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad"], ["good"]])
    test.set_prompt(prompt)
    assert test.verbalizer_raw == [["bad"], ["good"]]

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad"], ["good"], ['neutral']])
    test.set_prompt(prompt)
    assert test.verbalizer_raw == [["bad"], ["good"], ['neutral']]

def test_calibrate():
    ex = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
    answer = tensor([[0.2500, 0.7500],
        [0.5000, 0.5000],
        [0.7500, 0.2500]])
    assert torch.allclose(answer, calibrate(ex))

def test_get_verbalizer():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    test = CausalLM4Classification("sshleifer/tiny-gpt2", prompt)
    vbz_idx, grp_idx = test._get_verbalizer([["Bad", "Horrible"], ["Good"]], lower=True)
    assert grp_idx == [[0, 1, 2, 3], [4, 5]]
    assert len(grp_idx[0]) == 4 and len(grp_idx[1]) == 2
    assert len(vbz_idx) == 6
    vbz_idx, grp_idx = test._get_verbalizer([["bad"], ["good"]])
    assert grp_idx == [[0], [1]]
    assert len(vbz_idx) == 2
    assert len(grp_idx[0]) == 1 and len(grp_idx[1]) == 1

    test = MaskedLM4Classification("nreimers/BERT-Tiny_L-2_H-128_A-2", prompt)
    # lower case not tested, as this tokenizer is uncased
    vbz_idx, grp_idx = test._get_verbalizer([["bad"], ["good"]])
    assert grp_idx == [[0], [1]]
    assert len(vbz_idx) == 2
    assert len(grp_idx[0]) == 1 and len(grp_idx[1]) == 1

def test_forward_function():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    test = CausalLM4Classification(model_id_gen, prompt)
    batch_size=2
    for i in range(0, len(sample_data), batch_size):
        batch = test.prompt.get_tensors({'text': sample_data[i:i+batch_size]})
        output = test.forward(batch).detach().cpu()
        output = torch.nn.functional.softmax(output, dim=-1)
        assert torch.sum(output).round().item() == float(batch_size)
        output, model_output = test.forward(batch, return_model_output=True)
        output = torch.nn.functional.softmax(output.detach().cpu(), dim=-1)
        assert isinstance(model_output, ModelOutput)
        assert torch.sum(output).round().item() == float(batch_size)

        _, _ = test.forward(batch, return_model_output=True)

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    test = MaskedLM4Classification(model_id_mlm, prompt)
    batch_size=2
    for i in range(0, len(sample_data), batch_size):
        batch = test.prompt.get_tensors({'text': sample_data[i:i+batch_size]})
        output = test.forward(batch).detach().cpu()
        output = torch.nn.functional.softmax(output, dim=-1)
        assert torch.sum(output).round().item() == float(batch_size)
        _, model_output = test.forward(batch, return_model_output=True)
        assert isinstance(model_output, ModelOutput)
        assert torch.sum(output.detach().cpu()).round().item() == float(batch_size)

        _, _ = test.forward(batch, return_model_output=True)


def test_calibrate_output():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])

    model = CausalLM4Classification(
        model_id_gen,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": sample_data[0:4]})

    otp = model.classify(dataset, return_type="torch")
    pred_cali = calibrate(otp.distribution)
    otp_ = model.calibrate_output(otp)
    assert isinstance(otp_.predictions, torch.Tensor)
    assert torch.allclose(otp_.distribution, pred_cali)
    assert isinstance(otp_.distribution, torch.Tensor)

    otp = model.classify(dataset, return_type="list")
    pred_cali = calibrate(torch.tensor(otp.distribution))
    otp_ = model.calibrate_output(otp)
    assert isinstance(otp_.predictions, list)
    assert torch.allclose(torch.tensor(otp_.distribution), pred_cali)
    assert isinstance(otp_.distribution, list)

    otp = model.classify(dataset, return_type="pandas")
    pred_cali = calibrate(torch.tensor(otp.distribution.values))
    otp_ = model.calibrate_output(otp)
    assert isinstance(otp_.predictions, pd.DataFrame)
    assert torch.allclose(torch.tensor(otp_.distribution.values), pred_cali)
    assert isinstance(otp_.distribution, pd.DataFrame)

    otp = model.classify(dataset, return_type="numpy")
    pred_cali = calibrate(torch.from_numpy(otp.distribution))
    otp_ = model.calibrate_output(otp)
    assert isinstance(otp_.predictions, np.ndarray)
    assert torch.allclose(torch.from_numpy(otp_.distribution), pred_cali)
    assert isinstance(otp_.distribution, np.ndarray)

    otp = model.classify(dataset, return_type="polars")
    pred_cali = calibrate(torch.from_numpy(otp.distribution.to_numpy().copy()))
    otp_ = model.calibrate_output(otp)
    assert isinstance(otp_.predictions, pl.DataFrame)
    assert torch.allclose(torch.from_numpy(otp_.distribution.to_numpy().copy()), pred_cali)
    assert isinstance(otp_.distribution, pl.DataFrame)