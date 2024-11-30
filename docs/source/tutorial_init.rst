Tutorial - Basic Usage
======================

Using promptzl is simple. Here's a basic example of how to use it to classify text with a masked language model:

.. code-block:: python

    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

    model = MaskedLM4Classification("roberta-large", prompt)
    output = model.classify(dataset['test'], show_progress_bar=True).predictions
    sum([int(prd == lbl) for prd, lbl in zip(output, dataset['test']['label'])]) / len(output)
    0.7986842105263158

This is a simple example, but promptzl can be used for more complex tasks as well.