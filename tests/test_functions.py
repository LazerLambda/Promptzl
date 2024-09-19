import os
import sys

import pytest
import torch
from datasets import Dataset
from torch import tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import the promptzel package
from promptzl import *


def test_combine_function():
    test = CausalLM4Classification("sshleifer/tiny-gpt2", Verbalizer([['1', '2'], ['3']])) 
    combined = test._combine_logits(tensor([[1,3,7], [2,4,8]]))
    assert torch.all(combined == tensor([[2., 7.], [3., 8.]]))

    test = MaskedLM4Classification("nreimers/BERT-Tiny_L-2_H-128_A-2", Verbalizer([['1', '2'], ['3']])) 
    combined = test._combine_logits(tensor([[1,3,7], [2,4,8]]))
    assert torch.all(combined == tensor([[2., 7.], [3., 8.]]))


# TODO: Test get_verbalizer
