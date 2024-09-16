import os
import sys

import pytest
import torch
from datasets import Dataset

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
import promptzl


class TestClassification:

    sample_data = [
        "The pizza was horribe and the staff rude. Won't recommend.",
        "The pasta was undercooked and the service was slow. Not going back.",
        "The salad was wilted and the waiter was dismissive. Avoid at all costs.",
        # "The soup was cold and the ambiance was noisy. Not a pleasant experience.",
        # "The burger was overcooked and the fries were soggy. I wouldn't suggest this place.",
        # "The sushi was not fresh and the staff seemed uninterested. Definitely not worth it.",
        # "The steak was tough and the wine was sour. A disappointing meal.",
        # "The sandwich was bland and the coffee was lukewarm. Not a fan of this café.",
        # "The dessert was stale and the music was too loud. I won't be returning.",
        # "The chicken was dry and the vegetables were overcooked. A poor dining experience.",
    ]

    def test_simple_causal_class_prompt(self):
        with pytest.raises(AssertionError):
            promptzl.CausalLM4Classification(
                "sshleifer/tiny-gpt2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text"), promptzl.Text(". It was")
                ),
            )

        with pytest.raises(AssertionError):
            promptzl.CausalLM4Classification(
                "sshleifer/tiny-gpt2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text"),
                    promptzl.Verbalizer([["bad"], ["good"]]),
                    promptzl.Text(". It was"),
                ),
            )

        model = promptzl.CausalLM4Classification(
            "sshleifer/tiny-gpt2",
            prompt_or_verbalizer=promptzl.Prompt(
                promptzl.Key("text"),
                promptzl.Text(". It was"),
                promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
            ),
        )
        dataset = Dataset.from_dict({"text": self.sample_data})
        model.classify(dataset)
        otpt = model.classify(dataset)
        assert int(torch.sum(otpt).item()) == len(dataset)
        model.classify(dataset, batch_size=2)
        model.classify(dataset, batch_size=2, show_progress_bar=True)
        model.classify(dataset, return_type="list")
        model.classify(dataset, return_type="pandas")
        model.classify(dataset, return_type="numpy")
        model.classify(self.sample_data)
        model.classify(self.sample_data, calibrate=True, calibrate_samples=100)
        model.classify(self.sample_data, calibrate=True, calibrate_samples=2)

    def test_simple_mlm_class_prompt(self):
        with pytest.raises(AssertionError):
            promptzl.MaskedLM4Classification(
                "nreimers/BERT-Tiny_L-2_H-128_A-2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text"), promptzl.Text(". It was")
                ),
            )

        1

    def test_simple_mlm_class_prompt_w_multiple(self):

        model = promptzl.MaskedLM4Classification(
            "nreimers/BERT-Tiny_L-2_H-128_A-2",
            prompt_or_verbalizer=promptzl.Prompt(
                promptzl.Key("text_a"),
                promptzl.Text(". It was"),
                promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
                promptzl.Key("text_b"),
            ),
        )
        dataset = Dataset.from_dict(
            {"text_a": self.sample_data, "text_b": self.sample_data[::-1]}
        )
        otpt = model.classify(dataset)
        assert int(torch.sum(otpt).item()) == len(dataset)

    def test_simple_causal_class_prompt_w_multiple(self):

        model = promptzl.CausalLM4Classification(
            "nreimers/BERT-Tiny_L-2_H-128_A-2",
            prompt_or_verbalizer=promptzl.Prompt(
                promptzl.Key("text_a"),
                promptzl.Text(". It was"),
                promptzl.Key("text_b"),
                promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
            ),
        )
        dataset = Dataset.from_dict(
            {"text_a": self.sample_data, "text_b": self.sample_data[::-1]}
        )
        otpt = model.classify(dataset)
        assert int(torch.sum(otpt).item()) == len(dataset)

    # def test_sequence_length_restriction_mlm(self):
    #     model = promptzl.MaskedLM4Classification(
    #         "nreimers/BERT-Tiny_L-2_H-128_A-2",
    #         prompt_or_verbalizer=promptzl.Prompt(
    #             promptzl.Key("text_a"),
    #             promptzl.Text(". It was"),
    #             promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
    #             promptzl.Key("text_b"),
    #         ),
    #     )
    #     dataset = Dataset.from_dict(
    #         {"text_a": ["a " * 1000 + "a"] * 3, "text_b": ["b " * 1000 + "b"] * 3}
    #     )
    #     model.classify(dataset, data_collator="safe")

    def test_missing_mask_long_sequence(self):
        with pytest.raises(AssertionError):
            model = promptzl.MaskedLM4Classification(
                "nreimers/BERT-Tiny_L-2_H-128_A-2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text_a"),
                    promptzl.Text(". It was"),
                    promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
                    promptzl.Key("text_b"),
                ),
            )
            dataset = Dataset.from_dict(
                {"text_a": ["a " * 1000 + "a"] * 3, "text_b": ["b " * 1000 + "b"] * 3}
            )
            model.classify(dataset, data_collator="fast")

    def test_sequence_length_restriction_causal(self):
        model = promptzl.CausalLM4Classification(
            "sshleifer/tiny-gpt2",
            prompt_or_verbalizer=promptzl.Prompt(
                promptzl.Key("text_a"),
                promptzl.Text(". It was"),
                promptzl.Key("text_b"),
                promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
            ),
        )
        dataset = Dataset.from_dict(
            {"text_a": ["a " * 1000 + "a"] * 3, "text_b": ["b " * 1000 + "b"] * 3}
        )
        model.classify(dataset, data_collator="fast")