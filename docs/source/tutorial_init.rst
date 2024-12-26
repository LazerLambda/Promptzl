.. _tutorial_tldr:

Tutorial - Basic Usage
======================

Promptzl can be used in two ways: with `masked language models <https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling>`_ 
and with `causal language models <https://huggingface.co/docs/transformers/en/tasks/language_modeling>`_.

After running :code:`pip install -U promptzl` it is possible to run the following examples.

Causal Language Models
----------------------

All causal models from the 🤗-transformers library can be used for classification tasks. The idea is to guide the model to produce the correct output by
providing a prompt that contains the information about the classification task and condens the classification into a single word at the end of the prompt.
In the following, we will see an example of how a base model without fine-tuning is used for classification:

.. code-block:: python

    from datasets import load_dataset
    from promptzl import FnVbzPair, Vbz, CausalLM4Classification
    from sklearn.metrics import accuracy_score

    dataset = load_dataset("mteb/amazon_polarity")['test'].select(range(1000))

    prompt = FnVbzPair(lambda e:\
        f"""
        Product Review Classification into categories 'positive' or 'negative'.

        'Good value
        
        I love Curve and this large bottle offers great value. Highly recommended.'='positive'
        'Edge of Danger
        
        1 star - only because that's the minimum. This book shows that famous people can publish anything.'='negative'

        '{e['text']}'=""",
        Vbz({0: ["negative"], 1: ["positive"]}))

    model = CausalLM4Classification(
        'HuggingFaceTB/SmolLM2-1.7B',
        prompt=prompt)

    output = model.classify(dataset, show_progress_bar=True, batch_size=8)
    accuracy_score(dataset['label'], output.predictions)
    0.935

It is also possible to use *Prompt-Element-Objects* as shown in the following example. Using *Prompt-Element-Objects* (see :ref:`prompt-element-objects`)
is safer, as it automatically truncates the prompt to the maximum model length, which is especially useful when using
smaller models where the context length is limited.


Masked Language Models
----------------------

Here's a basic example (from `Schick and Schütze., 2020 <https://aclanthology.org/2021.eacl-main.20>`_) of how to classify text with a *masked language model*.
Instead of using :ref:`functoin_verbalizer_pair`, we use *prompt-element-objects* to construct the prompt as they truncate the data if it exceeds the
model's context length.

.. code-block:: python

    from datasets import load_dataset
    from promptzl import Key, Txt, Vbz, MaskedLM4Classification
    from sklearn.metrics import accuracy_score

    dataset = load_dataset("SetFit/ag_news")['test']

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

    model = MaskedLM4Classification("roberta-large", prompt)
    output = model.classify(dataset, show_progress_bar=True)
    accuracy_score(dataset['label'], output.predictions)
    0.7986842105263158
