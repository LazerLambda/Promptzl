import os
import sys

import pytest
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
import promptzl
from promptzl import *

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


def test_simple_causal_class_prompt():

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])

    model = promptzl.CausalLM4Classification(
        model_id_gen,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": sample_data})
    model.classify(dataset)
    otpt = model.classify(dataset, predict_labels=True)
    assert int(torch.sum(otpt).item()) == len(dataset)
    model.classify(dataset, batch_size=2)
    model.classify(dataset, batch_size=2, temperature=0.5)
    model.classify(dataset, batch_size=2, calibrate=True)
    model.classify(dataset, batch_size=2, show_progress_bar=True)
    model.classify(dataset, return_type="list")
    model.classify(dataset, return_type="pandas")
    model.classify(dataset, return_type="numpy")
    model.classify(dataset, return_type="polars")

    dataset = DatasetDict({
        'train': Dataset.from_dict({"text": sample_data}),
        'test': Dataset.from_dict({"text": sample_data})})
    model.classify(dataset)

    model.prompt.truncate = False
    model.classify(dataset)
    # model.classify(sample_data)
    # model.classify(sample_data, calibrate=True, calibrate_samples=100)
    # model.classify(sample_data, calibrate=True, calibrate_samples=2)

def test_simple_mlm_class_prompt():

    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])

    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": sample_data})
    model.classify(dataset)
    otpt = model.classify(dataset, predict_labels=False)
    assert int(torch.sum(otpt).item()) == len(dataset)
    model.classify(dataset, batch_size=2)
    model.classify(dataset, batch_size=2, temperature=0.5)
    model.classify(dataset, batch_size=2, calibrate=True)
    model.classify(dataset, batch_size=2, show_progress_bar=True)
    model.classify(dataset, return_type="list")
    model.classify(dataset, return_type="pandas")
    model.classify(dataset, return_type="numpy")
    model.classify(dataset, return_type="polars")

    dataset = DatasetDict({
        'train': Dataset.from_dict({"text": sample_data}),
        'test': Dataset.from_dict({"text": sample_data})})
    model.classify(dataset)

    model.prompt.truncate = False
    model.classify(dataset)

def test_w_o_truncation():
    prompt = Key("text") + Txt(". It was ") + Vbz([["bad", "horrible"], ["good"]])
    dataset = Dataset.from_dict({"text": sample_data})

    model = promptzl.CausalLM4Classification(
        model_id_gen,
        prompt=prompt,
        truncate=True
    )
    model.classify(dataset)
    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt,
        truncate=True
    )
    model.classify(dataset)

def test_w_fvp():
    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm)
    mask_token = tokenizer.mask_token
    prompt = FVP(lambda e: f"{e['text']} It was [MASK]", Vbz([["bad", "horrible"], ["good"]])) 
    dataset = Dataset.from_dict({"text": sample_data})

    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt,
        truncate=True
    )
    model.classify(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_id_gen)
    prompt = FVP(lambda e: f"{e['text']}. It was ",Vbz([["bad", "horrible"], ["good"]])) 
    model = promptzl.CausalLM4Classification(
        model_id_gen,
        prompt=prompt,
        truncate=True
    )
    model.classify(dataset)

def test_w_vbz_dict():

    prompt = Key("text") + Txt(". It was ") + Vbz({0: ["bad", "horrible"], 1: ["good"]})
    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": sample_data})
    output = model.classify(dataset, use_dict_keys=True, return_type="polars", predict_labels=False)
    assert output.columns == ['0', '1']

    output = model.classify(dataset, use_dict_keys=True, return_type="polars")
    assert output.columns == ['Prediction']

    output = model.classify(dataset, use_dict_keys=True, return_type="pandas", predict_labels=False)
    assert output.columns.to_list() == [0, 1]

    output = model.classify(dataset, use_dict_keys=True, return_type="pandas")
    assert output.columns.to_list() == ['Prediction']

def test_pred_classes_directly():
    prompt = Key("text") + Txt(". It was ") + Vbz({0: ["bad", "horrible"], 1: ["good"]})
    model = promptzl.MaskedLM4Classification(
        model_id_mlm,
        prompt=prompt
    )
    dataset = Dataset.from_dict({"text": sample_data})
    model.classify(dataset, batch_size=2, predict_labels=True)
    model.classify(dataset, return_type="list", predict_labels=True)
    model.classify(dataset, return_type="pandas", predict_labels=True)
    model.classify(dataset, return_type="numpy", predict_labels=True)
    model.classify(dataset, return_type="polars", predict_labels=True)
