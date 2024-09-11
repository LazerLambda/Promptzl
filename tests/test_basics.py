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


class TestPromptzel:

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

    def _init_promptzl(self, model_id, generate):
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

        test = promptzl.LLM4ClassificationBase(
            model,
            tokenizer,
            prompt_or_verbalizer=promptzl.Verbalizer(
                [["bad"], ["good", "wonderful", "great"]]
            ),
            generate=generate,
        )
        return test, device

    def test_sample_wo_pattern_autoregressive(self):
        model_id = "sshleifer/tiny-gpt2"
        promptzl, device = self._init_promptzl(model_id, True)

        dataset = Dataset.from_dict({"text": self.sample_data})

        promptzl.tokenizer.padding_side = "left"
        if promptzl.tokenizer.pad_token is None:
            promptzl.tokenizer.pad_token = promptzl.tokenizer.eos_token

        promptzl.tokenizer.pad_token_id = promptzl.tokenizer.convert_tokens_to_ids(
            promptzl.tokenizer.pad_token
        )

        # Tokenize the dataset
        def tokenize_function(examples):
            return promptzl.tokenizer(
                list(
                    map(
                        lambda e: e[0] + e[1],
                        zip(
                            examples["text"],
                            [" Overall, it was "] * len(examples["text"]),
                        ),
                    )
                ),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
        dataloader = DataLoader(tokenized_dataset, batch_size=100)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = promptzl.forward(batch)
            assert output.shape[0] == len(batch["input_ids"]) and output.shape[1] == 2
            pytest.approx(len(batch), torch.sum(output), abs=0.1)

    def test_sample_wo_pattern_mlm(self):
        model_id = "nreimers/BERT-Tiny_L-2_H-128_A-2"
        promptzl, device = self._init_promptzl(model_id, False)

        dataset = Dataset.from_dict({"text": self.sample_data})

        # Tokenize the dataset
        def tokenize_function(examples):
            return promptzl.tokenizer(
                list(
                    map(
                        lambda e: e[0] + e[1],
                        zip(
                            examples["text"],
                            [" This review is [MASK] "] * len(examples["text"]),
                        ),
                    )
                ),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
        dataloader = DataLoader(tokenized_dataset, batch_size=100)
        promptzl.model.to(device)

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = promptzl.forward(batch)
            assert output.shape[0] == len(batch["input_ids"]) and output.shape[1] == 2
            pytest.approx(len(batch), torch.sum(output), abs=0.1)

    def test_sample_wo_pattern_mlm_no_masks(self):
        model_id = "nreimers/BERT-Tiny_L-2_H-128_A-2"

        promptzl, device = self._init_promptzl(model_id, False)

        dataset = Dataset.from_dict({"text": self.sample_data})
        promptzl.tokenizer.pad_token_id = promptzl.tokenizer.convert_tokens_to_ids(
            promptzl.tokenizer.pad_token
        )

        # Tokenize the dataset
        def tokenize_function(examples):
            return promptzl.tokenizer(
                list(
                    map(
                        lambda e: e[0] + e[1],
                        zip(
                            examples["text"],
                            [" This review is"] * len(examples["text"]),
                        ),
                    )
                ),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
        dataloader = DataLoader(tokenized_dataset, batch_size=100)
        promptzl.model.to(device)

        with pytest.raises(Exception):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = promptzl.forward(batch)

    def test_sample_wo_pattern_mlm_no_mask_token(self):
        device = "cpu"
        model_id = "nreimers/BERT-Tiny_L-2_H-128_A-2"

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, clean_up_tokenization_spaces=True
        )
        tokenizer.mask_token = None
        tokenizer.mask_token_id = None
        model = AutoModelForMaskedLM.from_pretrained(model_id)

        with pytest.raises(Exception):
            test = promptzl.LLM4ClassificationBase(
                model, tokenizer, [["bad"], ["good"]], generate=False
            )

    def test_calibration(self):
        model_id = "nreimers/BERT-Tiny_L-2_H-128_A-2"
        promptzl, device = self._init_promptzl(model_id, False)

        dataset = Dataset.from_dict({"text": self.sample_data})

        # Tokenize the dataset
        def tokenize_function(examples):
            return promptzl.tokenizer(
                list(
                    map(
                        lambda e: e[0] + e[1],
                        zip(
                            examples["text"],
                            [" This review is [MASK] "] * len(examples["text"]),
                        ),
                    )
                ),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
        dataloader = DataLoader(tokenized_dataset, batch_size=100)
        promptzl.model.to(device)
        promptzl.set_contextualized_prior(dataloader)

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = promptzl.forward(batch)
            assert output.shape[0] == len(batch["input_ids"]) and output.shape[1] == 2
            pytest.approx(len(batch), torch.sum(output), abs=0.1)

    def test_equal_tokens_error(self):
        with pytest.raises(AssertionError):
            promptzl.CausalLM4Classification(
                "sshleifer/tiny-gpt2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text"),
                    promptzl.Text(". It was"),
                    promptzl.Verbalizer([["bad"], ["bad"]]),
                ),
            )
        with pytest.raises(AssertionError):
            promptzl.MaskedLM4Classification(
                "nreimers/BERT-Tiny_L-2_H-128_A-2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text"),
                    promptzl.Text(". It was"),
                    promptzl.Verbalizer([["good"], ["good"]]),
                ),
            )

    def test_multiple_tokens_error_mlm(self):
        with pytest.raises(AssertionError):
            promptzl.MaskedLM4Classification(
                "nreimers/BERT-Tiny_L-2_H-128_A-2",
                prompt_or_verbalizer=promptzl.Prompt(
                    promptzl.Key("text"),
                    promptzl.Text(". It was"),
                    promptzl.Verbalizer([["bad", "very bad"], ["good"]]),
                ),
            )

    def test_simple_mlm_classification_wo_prompt(self):
        model = promptzl.MaskedLM4Classification(
            "nreimers/BERT-Tiny_L-2_H-128_A-2",
            promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
        )
        dataset = Dataset.from_dict({"text": self.sample_data})

        # Tokenize the dataset
        def tokenize_function(examples):
            return model.tokenizer(
                list(
                    map(
                        lambda e: e[0] + e[1],
                        zip(
                            examples["text"],
                            [" This review is [MASK] "] * len(examples["text"]),
                        ),
                    )
                ),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
        model.classify(tokenized_dataset)
        model.classify(tokenized_dataset, batch_size=2)
        model.classify(tokenized_dataset, batch_size=2, show_progress_bar=True)
        model.classify(tokenized_dataset, return_type="list")
        model.classify(tokenized_dataset, return_type="pandas")
        model.classify(tokenized_dataset, return_type="numpy")

        del model

    def test_simple_autoreg_classification_wo_prompt(self):
        model = promptzl.CausalLM4Classification(
            "sshleifer/tiny-gpt2", promptzl.Verbalizer([["bad", "horrible"], ["good"]])
        )
        dataset = Dataset.from_dict({"text": self.sample_data})

        # Tokenize the dataset
        def tokenize_function(examples):
            return model.tokenizer(
                list(
                    map(
                        lambda e: e[0] + e[1],
                        zip(
                            examples["text"],
                            [" This review is [MASK] "] * len(examples["text"]),
                        ),
                    )
                ),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
        model.classify(tokenized_dataset)
        model.classify(tokenized_dataset, batch_size=2)
        model.classify(tokenized_dataset, batch_size=2, show_progress_bar=True)
        model.classify(tokenized_dataset, return_type="list")
        model.classify(tokenized_dataset, return_type="pandas")
        model.classify(tokenized_dataset, return_type="numpy")

        del model

    def test_forward_function(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/tiny-gpt2", clean_up_tokenization_spaces=True
        )
        model = promptzl.CausalLM4Classification(
            "sshleifer/tiny-gpt2",
            prompt_or_verbalizer=promptzl.Prompt(
                promptzl.Key("text_a"),
                promptzl.Text(". It was"),
                promptzl.Key("text_b"),
                promptzl.Verbalizer([["bad", "horrible"], ["good"]]),
            ),
        )
        model.forward(batch=tokenizer('This is a test.', return_tensors="pt"), return_model_output=True)
        model.forward(batch=tokenizer('This is a test.', return_tensors="pt"), return_model_output=False)