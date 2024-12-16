.. _tutorial_tldr:

Tutorial - Basic Usage
======================

Promptzl can be used in two ways: with `masked language models <https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling>`_ 
and with `causal language models <https://huggingface.co/docs/transformers/en/tasks/language_modeling>`_.

After running :code:`pip install promptzl` it is possible to run the following examples.

Causal Language Models
----------------------

It is also possible to use a *causal language model* in a similar way. Here, it is also possible to leverage the
fine-tuned nature of novel LLMs, that are aimed at interacting with the user, making finding an appropriate prompt easier.
Thus, it is possible to instruct the model to generate a certain output, which further improves the classification performance:

.. code-block:: python

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

It is also possible to use *Prompt-Element-Objects* as it will be shown in the following example. Using *Prompt-Element-Objects* (see :ref:`prompt-element-objects`)
is safer, as it automatically truncates the prompt to the maximum length of the model. This is especially useful when using
smaller models where the context length is limited.


Masked Language Models
----------------------
Here's a basic example (from `Schick and Schütze., 2020 <https://aclanthology.org/2021.eacl-main.20>`_) of how to classify text with a *masked language model*:

.. code-block:: python

    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")['test']

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

    model = MaskedLM4Classification("roberta-large", prompt)
    output = model.classify(dataset, show_progress_bar=True).predictions
    sum([int(prd == lbl) for prd, lbl in zip(output, dataset['label'])]) / len(output)
    0.7986842105263158

This is a simple example, but promptzl can be used for more complex tasks as well.