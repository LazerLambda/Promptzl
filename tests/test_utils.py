import os
import sys

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
from promptzl.utils import *


class TestUtils:


    def test_data_collator_pad_init(self):
        tokenizer = AutoTokenizer.from_pretrained('nreimers/BERT-Tiny_L-2_H-128_A-2')
        data_collator = DataCollatorPromptPad(tokenizer, padding='max_length', padding_side='left')