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

Turn state-of-the-art LLMs into zero-shot PyTorch classifiers in just a few lines of code.

Promptzl offers:
   - 🤖 Zero-shot classification with LLMs
   - 🤗 Turning `causal <https://huggingface.co/models?pipeline_tag=text-generation>`_ and `masked <https://huggingface.co/models?pipeline_tag=fill-mask>`_ LMs into classifiers without any training
   - 📦 Batch processing on your device for efficiency
   - 🚀 Speed-up over calling an online API
   - 🔎 Transparency and accessibility by using the model locally
   - 📈 Distribution over the classes
   - ✂️ No need to extract the predictions from the answer.

Check out more in the [official documentation.](https://promptzl.readthedocs.io/en/latest/)

## Installation


`pip install -U promptzl`

## Getting Started

In just a few lines of code, you can transform a LLM of choice into an old-school classifier with all it's desirable properties:

Import necessary dependencies and initialize an example dataset:
```{python}
from datasets import load_dataset
from promptzl import *
from sklearn.metrics import accuracy_score

dataset = load_dataset("mteb/amazon_polarity")['test'].select(range(1000))
```

Define a prompt for guiding the language model to the correct predictions:
```{python}
prompt = FVP(lambda e:\
    f"""
    Product Review Classification into categories 'positive' or 'negative'.

    'Good value
    
    I love Curve and this large bottle offers great value. Highly recommended.'='positive'
    'Edge of Danger
    
    1 star - only because that's the minimum. This book shows that famous people can publish anything.'='negative'

    '{e['text']}'=""", Vbz({0: ["negative"], 1: ["positive"]}))
```

Initialize a model:
```{python}
model = CausalLM4Classification(
    'HuggingFaceTB/SmolLM2-1.7B',
    prompt=prompt)
```

Classify the data:
```{prompt}
output = model.classify(dataset, show_progress_bar=True, batch_size=1)
accuracy_score(dataset['label'], output.predictions)
0.935
```

For more detailed tutorials, check out the [documentation](https://promptzl.readthedocs.io/en/latest/)!

