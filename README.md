# <p style="text-align: center;">Pr🥨mptzl</p>

Zero/Few-Shot classifications with prompts and LLMs. Turn your LLM into a classifier!

## What is it?

Promptzl turns LLMs into old school classifiers that can be used for zero<sup>+</sup>-shot classification or distillation.

The resulting model is a pytorch-based classifier where you are in charge of. No API-fees, no intransparency.


## Installation

Download this repository, navigate to the folder and run:
`pip install .`


## How does it work?

LLMs output a value for each token that is predicted from which a distribution is formed that determines which token is chosen.

When asking the model a simple question based on a given instance, only few tokens are important. For example, for classifying the sentence 'The pizza was bad. It was [MASK]', the only tokens of interest are good and bad.

At this point, it is possible to extract the LLM's logits to determine which token is more likely to be predicted. 

To make it work, a prompt and a verbalizer a required. The prompt transforms the problem into a prediction task of one token and the verbalizer selects the tokens of interest.

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

### Further Reads
 - [Schick and Schütze 2020: Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification](https://aclanthology.org/2020.coling-main.488/)
 - [Schick and Schütze 2021: Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference](https://aclanthology.org/2021.eacl-main.20/)
 - [Hu et al. 2022: Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://aclanthology.org/2022.acl-long.158/)

## Installation (Dev)

`pip install -e .`
`pip install -r test-requirements.txt`
