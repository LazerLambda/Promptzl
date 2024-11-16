# import os
# import sys

# import pytest
# import torch
# from datasets import Dataset
# from transformers import AutoModelForMaskedLM, AutoTokenizer

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # Now we can import the promptzel package
# from promptzl.utils import *


# class TestUtils:


#     def test_data_collator_pad_init(self):
#         tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2', clean_up_tokenization_spaces=True)
#         data_collator = DataCollatorPromptPad(tokenizer, padding='max_length', padding_side='left')

#     # TODO: Test combine and test max lengths

#     # TODO Test concat empty tensor

#     # TODO Test List[str] input with safe datacollator (keys_in_prompt)


# TODO test left right padding 
# TODO test attention mask etc
# TODO check parity with tokenizer
# TODO CHeck length limitations