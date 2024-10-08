import os
import sys

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
import promptzl


class TestError:

    sample_data = [
        "The pizza was horribe and the staff rude. Won't recommend.",
        "The pasta was undercooked and the service was slow. Not going back.",
        "The salad was wilted and the waiter was dismissive. Avoid at all costs."
    ]

    def test_simple_causal_class_prompt(self):

        tokenizer = AutoTokenizer.from_pretrained(
            "nreimers/BERT-Tiny_L-2_H-128_A-2", clean_up_tokenization_spaces=True
        )
        tokenizer.mask_token_id = None
        model = AutoModelForMaskedLM.from_pretrained("nreimers/BERT-Tiny_L-2_H-128_A-2")

        with pytest.raises(ValueError):
            test = promptzl.LLM4ClassificationBase(
                model,
                tokenizer,
                prompt_or_verbalizer=promptzl.Verbalizer(
                    [["bad"], ["good"]]
                ),
                generate=False,
            )

    def test_simple_causal_class_prompt(self):

        tokenizer = AutoTokenizer.from_pretrained(
            "nreimers/BERT-Tiny_L-2_H-128_A-2", clean_up_tokenization_spaces=True
        )
        tokenizer.mask_token_id = None
        model = AutoModelForMaskedLM.from_pretrained("nreimers/BERT-Tiny_L-2_H-128_A-2")

        with pytest.raises(ValueError):
            promptzl.LLM4ClassificationBase(
                model,
                tokenizer,
                prompt_or_verbalizer=promptzl.Verbalizer(
                    [["bad"], ["good"]]
                ),
                generate=False,
            )

    def test_type_prompt_or_verbalizer_error(self):
        with pytest.raises(TypeError):
            promptzl.CausalLM4Classification(
                "sshleifer/tiny-gpt2",
                prompt_or_verbalizer="Test"
            )

    def test_length_function_error(self):
        with pytest.raises(NotImplementedError):
            model = promptzl.CausalLM4Classification(
                "sshleifer/tiny-gpt2",
                prompt_or_verbalizer=promptzl.Verbalizer([["bad"], ["good"]]),
            )
            model._text_length("Test")

    def test_data_collator_error(self):
        with pytest.raises(TypeError):
            model = promptzl.MaskedLM4Classification(
                "nreimers/BERT-Tiny_L-2_H-128_A-2",
                prompt_or_verbalizer="Test"
            )
            dataset = Dataset.from_dict(
                {"text_a": ["a " * 1000 + "a"] * 3, "text_b": ["b " * 1000 + "b"] * 3}
            )
            model.classify(dataset, data_collator="test")