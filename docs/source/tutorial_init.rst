Tutorial - Basic Usage
======================

Promptzl can be used in two ways: with `masked language models <https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling>`_ 
and with `causal language models <https://huggingface.co/docs/transformers/en/tasks/language_modeling>`_.

After running :code:`pip install promptzl` it is possible to run the following examples.

Masked Language Models
----------------------
Using promptzl is simple. Here's a basic example (from the work of Schick and Schütze) of how to use it to classify text with a *masked language model*:

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


Causal Language Models
----------------------
This is a simple example, but promptzl can be used for more complex tasks as well.

It is also possible to use a *causal language model* in a similar way. Here, it is also possible to leverage the
fine-tuned nature of novel LLMs, that are aimed at interacting with the user, making finding an appropriate prompt easier.
Thus, it is possible to instruct the model to generate a certain output, which further improves the classification performance:

.. code-block:: python

    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

    model = CausalLM4Classification("TODO", prompt)

However, finding a good prompt can be challenging. The next section will discuss how to establish a good workflow with this package.