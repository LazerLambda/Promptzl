Tutorial - Basic Usage
======================

Using promptzl is simple. Here's a basic example of how to use it to classify text with a masked language model:

.. code-block:: python

    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")

    verbalizer = Vbz({1: ["World"], 2: ["Sports"], 3: ["Business"], 4: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + TKy()

    model = MaskedLM4Classification("roberta-large", prompt, trust_remote_code=True)
    output = model.classify(dataset['test'], show_progress_bar=True)


Now the output can be evaluated. For example, to calculate the accuracy of the model:
.. code-block:: python

    import torch
    labels = dataset['test']['label']
    sum([int(truth == pred) for truth, pred in zip(labels, torch.argmax(output, -1))]) / len(labels)
    0.7986842105263158

This is a simple example, but promptzl can be used for more complex tasks as well.