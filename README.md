# <p style="text-align: center;">Pr🥨mptzl</p>

Zero/Few-Shot classifications with prompts and LLMs. Turn your LLM into a classifier!

## Installation

Download this repository, navigate to the folder and run:
`pip install .`

### Getting Started

In just a few lines of code, you can transform a LLM of choice into an old-school classifier with all it's desirable properties:
```{python}
    from promptzl import *
    from datasets import Dataset

    # Dataset
    Dataset.from_dict({'text': ["The pizza was good.", "The pizza was bad."]})

    # Verbalizer (define label words)
    verbalizer = Prompt(Key('text'), Text('It was '), Verbalizer([['bad'], ['good']]))

    # Inference
    model = MLM4Classification('a-hf-model', verbalizer)
    model.classify(dataset)
```

## Installation (Dev)

`pip install -e .`
`pip install -r test-requirements.txt`
