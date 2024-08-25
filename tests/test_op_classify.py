import operator
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


class TestOPClassify:

    sample_data = [
        "Albert Einstein was one of the greatest intellects of his time.",
        "The film was badly made.",
    ]

    def _init_promptzl(self, model_id, generate):
        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
            model, tokenizer, [["bad"], ["good", "wonderful", "great"]], generate
        )
        return test, device

    def test_sample_op_tutorial(self):
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
                            [" It was [MASK] "] * len(examples["text"]),
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
            output = promptzl.forward(batch, return_logits=True)
            expected_output_op = torch.tensor([[-2.8208, -1.5331], [-0.4941, -2.2750]])
            assert output == pytest.approx(expected_output_op, abs=1e-4)
