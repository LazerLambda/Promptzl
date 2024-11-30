<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/badge/License-MIT-yellow.svg)][#github-license]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=promptzl)][#docs-package]
![Tests Passing](https://github.com/lazerlambda/promptzl/actions/workflows/python-package.yml/badge.svg)

[#github-license]: https://github.com/LazerLambda/Promptzl/blob/main/LICENSE.md
[#docs-package]: https://promptzl.readthedocs.io/en/latest/
<!--- BADGES: END --->



# <p style="text-align: center;">Pr🥨mptzl (Under Development)</p>

Promptzl is a simple library for turning LLMs into traditional PyTorch-based classifiers using the 🤗 Transformers library.

Classify large datasets quickly and easily while maintaining full control!

## Installation

Download this repository, navigate to the folder and run:
`pip install .`

### Getting Started

In just a few lines of code, you can transform a LLM of choice into an old-school classifier with all it's desirable properties:
```{python}
    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

    model = MaskedLM4Classification("roberta-large", prompt)
    output = model.classify(dataset['test'], show_progress_bar=True).predictions
    sum([int(prd == lbl) for prd, lbl in zip(output, dataset['test']['label'])]) / len(output)
```

## Installation (Dev)

`pip install -e .`

`pip install -r test-requirements.txt`
