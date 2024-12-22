<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/badge/License-Apache-yellow.svg)][#github-license]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=promptzl)][#docs-package]
![Tests Passing](https://github.com/lazerlambda/promptzl/actions/workflows/python-package.yml/badge.svg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/promptzl?logo=pypi&style=flat)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/promptzl?logo=pypi&style=flat)][#pypi-package]

[#github-license]: https://github.com/LazerLambda/Promptzl/blob/main/LICENSE.md
[#docs-package]: https://promptzl.readthedocs.io/en/stable/
[#pypi-package]: https://pypi.org/project/promptzl/
<!--- BADGES: END --->


<!-- TODO -->
<h1 align="center">Pr🥨mptzl</h1>

Turn state-of-the-art LLMs into zero<sup>+</sup>-shot PyTorch classifiers in just a few lines of code.

Promptzl offers:
   - 🤖 Zero<sup>+</sup>-shot classification with LLMs
   - 🤗 Turning [causal](https://huggingface.co/models?pipeline_tag=text-generation>) and [masked](https://huggingface.co/models?pipeline_tag=fill-mask>) LMs into classifiers without any training
   - 📦 Batch processing on your device for efficiency
   - 🚀 Speed-up over calling an online API
   - 🔎 Transparency and accessibility by using the model locally
   - 📈 Distribution over labels
   - ✂️ No need to extract the predictions from the answer.

For more information, check out the [**official documentation**.](https://promptzl.readthedocs.io/en/latest/)

## Installation


`pip install -U promptzl`

## Getting Started

In just a few lines of code, you can transform a LLM of choice into an old-school classifier with all it's desirable properties:

Set up the dataset:
```python
from datasets import Dataset

dataset = Dataset.from_dict(
    {
        'text': [
            "The food was absolutely wonderful, from preparation to presentation, very pleasing.",
            "The service was a bit slow, but the food made up for it. Highly recommend the pasta!",
            "The restaurant was too noisy and the food was mediocre at best. Not worth the price.",
        ],
        'label': [1, 1, 0]
    }
)
```

Define a prompt for guiding the language model to the correct predictions:
```python
from promptzl import FnVbzPair, Vbz
prompt = FnVbzPair(
    lambda e: f"""Restaurant review classification into categories 'positive' or 'negative'.

    'Best pretzls in town!'='positive'
    'Rude staff, horrible food.'='negative'

    '{e['text']}'=""",
    Vbz({0: ["negative"], 1: ["positive"]}))
```

Initialize a model:
```python
from promptzl import CausalLM4Classification
model = CausalLM4Classification(
    'HuggingFaceTB/SmolLM2-1.7B',
    prompt=prompt)
```

Classify the data:
```python
from sklearn.metrics import accuracy_score
output = model.classify(dataset, show_progress_bar=True, batch_size=1)
accuracy_score(dataset['label'], output.predictions)
1.0
```

For more detailed tutorials, check out the [documentation](https://promptzl.readthedocs.io/en/latest/)!

