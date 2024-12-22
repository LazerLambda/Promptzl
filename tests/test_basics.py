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
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])

    model = AutoModelForCausalLM.from_pretrained(model_id_gen)
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
    LLM4ClassificationBase(model=model, tokenizer=tokenizer, device=device, prompt=prompt, generate=True)
    LLM4ClassificationBase(model=model, tokenizer=tokenizer, device="cpu", prompt=prompt, generate=True)

    model = AutoModelForMaskedLM.from_pretrained(model_id_mlm)
    tokenizer = AutoTokenizer.from_pretrained(model_id_mlm, clean_up_tokenization_spaces=True)
    LLM4ClassificationBase(model=model, tokenizer=tokenizer, device=device, prompt=prompt, generate=False)
    LLM4ClassificationBase(model=model, tokenizer=tokenizer, device="cpu", prompt=prompt, generate=False)

    with pytest.raises(Exception):
        model = AutoModelForCausalLM.from_pretrained(model_id_gen)
        prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
        promptzl.LLM4ClassificationBase(
            model=model,
            tokenizer=1,
            prompt=prompt,
            generate=False
        )

    with pytest.raises(Exception):
        tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
        prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
        promptzl.LLM4ClassificationBase(
            model=None,
            tokenizer=tokenizer,
            prompt=prompt,
            generate=False
        )

    with pytest.raises(Exception):
        model = AutoModelForCausalLM.from_pretrained(model_id_gen)
        tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
        promptzl.LLM4ClassificationBase(
            model=model,
            tokenizer=tokenizer,
            prompt=2,
            generate=False
        )

    with pytest.raises(Exception):
        prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
        model = AutoModelForCausalLM.from_pretrained(model_id_gen)
        tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
        promptzl.LLM4ClassificationBase(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generate=-1
        )

    with pytest.raises(Exception):
        prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
        model = AutoModelForCausalLM.from_pretrained(model_id_gen)
        tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
        promptzl.LLM4ClassificationBase(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generate=False,
            lower_verbalizer=-1
        )


def init_promptzl(model_id, generate):
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, clean_up_tokenization_spaces=True
    )
    if torch.cuda.is_available():
        if not generate:
            model = AutoModelForMaskedLM.from_pretrained(model_id)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", load_in_8bit=True
            )
    else:
        if not generate:
            model = AutoModelForMaskedLM.from_pretrained(model_id)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)

    prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
    test = promptzl.LLM4ClassificationBase(
        model,
        tokenizer,
        prompt=prompt,
        generate=generate,
    )
    return test, device


def test_str_module():
    model = MaskedLM4Classification(model_id_mlm, prompt=Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]]))
    assert isinstance(str(model), str)
    assert isinstance(repr(model), str)

    model = MaskedLM4Classification(model_id_mlm, prompt=Txt("This is a\ntest ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]]))
    assert isinstance(str(model), str)
    assert isinstance(repr(model), str)

    model = CausalLM4Classification(model_id_gen, prompt=Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]]))
    assert isinstance(str(model), str)
    assert isinstance(repr(model), str)

    model = CausalLM4Classification(model_id_gen, prompt=Txt("This is a\ntest ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]]))
    assert isinstance(str(model), str)
    assert isinstance(repr(model), str)

    model = AutoModelForCausalLM.from_pretrained(model_id_gen)
    tokenizer = AutoTokenizer.from_pretrained(model_id_gen, clean_up_tokenization_spaces=True)
    prompt = Txt("This is a test ") + Key("text") + Vbz([["bad"], ["good", "wonderful", "great"]])
    tmp = LLM4ClassificationBase(model=model, tokenizer=tokenizer, prompt=prompt, generate=True)
    assert isinstance(str(tmp), str)
    assert isinstance(repr(tmp), str)
