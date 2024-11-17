# <p style="text-align: center;">Pr🥨mptzl</p>

Promptzl is a simple library for turning LLMs into old-school
pytorch-based classifiers based on the 🤗-transformers library.

## Installation

Download this repository, navigate to the folder and run:
`pip install .`

### Getting Started

In just a few lines of code, you can transform a LLM of choice into an old-school classifier with all it's desirable properties:
```{python}
    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")

    verbalizer = Vbz({1: ["World"], 2: ["Sports"], 3: ["Business"], 4: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + TKy()

    model = MaskedLM4Classification("roberta-large", prompt, trust_remote_code=True)
    output = model.classify(dataset['test'], show_progress_bar=True)
```

## Installation (Dev)

`pip install -e .`
`pip install -r test-requirements.txt`
