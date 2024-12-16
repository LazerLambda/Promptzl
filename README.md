<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/badge/License-Apache-yellow.svg)][#github-license]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=promptzl)][#docs-package]
![Tests Passing](https://github.com/lazerlambda/promptzl/actions/workflows/python-package.yml/badge.svg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/promptzl?logo=pypi&style=flat)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/promptzl?logo=pypi&style=flat)][#pypi-package]

[#github-license]: https://github.com/LazerLambda/Promptzl/blob/main/LICENSE.md
[#docs-package]: https://promptzl.readthedocs.io/en/latest/
[#pypi-package]: https://pypi.org/project/promptzl/
<!--- BADGES: END --->


<!-- TODO -->
# <p style="text-align: center;">Pr🥨mptzl</p>

Promptzl is an easy-to-use library for turning state-of-the-art LLMs into old-school
pytorch-based, zero-shot classifiers based on the 🤗-transformers library.

   - 💪 Zero-shot classification
   - 🤖 All [causal](https://huggingface.co/models?pipeline_tag=text-generation) and [masked](https://huggingface.co/models?pipeline_tag=fill-mask) LMs from the 🤗-hub are available
   - ⚡ Fast and easy to use (just a few lines of code needed. Check the example below!)
   - 📦 Promptzl works with batches
   - 🔎 All models are transparent on your device
   - 📈 Properties of old-school classifiers included
   - 🚀 No need for proprietary APIs

Check out more in the [official documentation.](https://promptzl.readthedocs.io/en/latest/)

## Installation


`pip install -U promptzl`

## Getting Started

In just a few lines of code, you can transform a LLM of choice into an old-school classifier with all it's desirable properties:

```{python}
from promptzl import *
from datasets import load_dataset
import torch

dataset = load_dataset("mteb/amazon_polarity")['test'].select(range(1000))

prompt = FVP(lambda e:\
    f"""
    Product Review Classification into categories 'positive' or 'negative'.

    'Good value
    
    I love Curve and this large bottle offers great value. Highly recommended.'='positive'
    'Edge of Danger
    
    1 star - only because that's the minimum. This book shows that famous people can publish anything.'='negative'

    '{e['text']}'=""", Vbz({0: ["negative"], 1: ["positive"]}))

model = CausalLM4Classification(
    'HuggingFaceTB/SmolLM2-1.7B',
    prompt=prompt)

output = model.classify(dataset, show_progress_bar=True, batch_size=1).predictions
sum([int(prd == lbl) for prd, lbl in zip(output, torch.tensor(dataset['label']))]) / len(output)
0.92
```

For more detailed tutorials, check out the [official documentation](https://promptzl.readthedocs.io/en/latest/)!

