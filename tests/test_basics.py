import operator
import os
import sys

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
import promptzl


class TestPromptzel:
    # def test_sample(self):
    #     pattern = promptzel.Pattern([promptzel.DataKey("data"), promptzel.Prompt("Is this good or bad?"), promptzel.Mask()])
    #     test = promptzel.LLM4ForPatternExploitationClassification("distilbert/distilgpt2", [["good"], ["bad"]], pattern)
    #     x = pattern.get_prompt_single({'data': test.tokenizer.encode("Das war nicht gut!")})
    #     print(x)
    #     print(test.tokenizer.decode(x))

    def test_sample_wo_pattern(self):
        test = promptzl.LLM4ForPatternExploitationClassification(
            "sshleifer/tiny-gpt2",
            verbalizer=[["bad"], ["good"]],
            device_map="auto",
            load_in_8bit=True,
        )
        reviews = [
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

        # Convert the list to a Hugging Face Dataset
        dataset = Dataset.from_dict({"text": reviews})

        test.tokenizer.padding_side = "left"
        # Assuming 'test.tokenizer' is your tokenizer object
        if test.tokenizer.pad_token is None:
            test.tokenizer.pad_token = test.tokenizer.eos_token

        # Now, you can safely set the pad_token_id (though it should be automatically set by the above step)
        test.tokenizer.pad_token_id = test.tokenizer.convert_tokens_to_ids(
            test.tokenizer.pad_token
        )

        # Tokenize the dataset
        def tokenize_function(examples):
            return test.tokenizer(
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

        # test.model.to('cuda')

        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            output = test.forward(batch)
            pytest.approx(len(batch), torch.sum(output), abs=0.1)
