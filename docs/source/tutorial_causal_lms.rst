Tutorial: Causal-Language-Model-Based Classifiers
=================================================

Causal langauge models are trained to predict the next token in a sequence based on the previous context (next-token objective). This requires us to always set the verbalizer
at the end of the sequence. The prompt is crucial for the performance of the model and the training objective of the underlying model must be taken into account.
In the current open-source causal LLM ecosystem, there are two groups of causal LMs, the base models, the ones which where only trained on the next-token objective and the
ones that were further fine-tuned on instructions to act like assistants like ChatGPT. Both models can be used for classifications but the characterstics of each nex-token predictions
must be kept in mind. Usually the latter models have a name indicating the further tuning like '-instruct' (:code:`HuggingFaceTB/SmolLM2-1.7B` vs. :code:`HuggingFaceTB/SmolLM2-1.7B-Instruct`
where the latter is fine-tuned).

Example Dataset
---------------

As promtzl is intended to be used along the huggingface transformers and dataset libraries, we first have to initilaize an example dataset. For this
tutorial, we will use the IMDB dataset which is a binary classification task. The dataset is loaded as follows:

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("mteb/imdb")['test'].select(range(1000))

For the sake of brevity, we will only use the first 1000 examples of the dataset.

Defining a Prompt
-----------------

As we are interested in classifying only positive or negative samples, we need to define a prompt that guides the model to produce the desired output and
verbalizer (TODO link to verbalizer in background) that extracts the logits of interest. The prompt is defined as follows:

.. code-block:: python

    from promptzl import *

    prompt = FVP(
        lambda e: f"""
        Movie Review Classification into categories 'positive' or 'negative'.

        '{e['text']}'='""",
        Vbz({0: ["negative"], 1: ["positive"]})
    )

..note::
    It is also possible to use *Prompt-Element-Objects* (see :ref:`prompt-element-objects`) and also to provide a list of label words to the verbalizer.
    However, using a dict allows us to return keys of the dict in the predictions list eventually.

Loading the Model
-----------------

After we have defined the prompt, we can load the model. In this case, we will use the causal language model :code:`HuggingFaceTB/SmolLM2-1.7B` which is a causal language model
that is also fairly small so it can fit on even smaller GPUs. The model is loaded as follows:

.. code-block:: python

    model = CausalLM4Classification(
        'HuggingFaceTB/SmolLM2-1.7B',
        prompt=prompt
    )

Nice, now we have set up everything and can start to calssify the dataset!

Classifying the Dataset
-----------------------

To classify the dataset, we can use the :code:`classify` method of the model. The method returns an object that contains the predictions and the distribution.
It is also possible to get the (combined) logits for each class, however the default behavior only returns predictions and distributions. The method is called as follows:

.. code-block:: python

    output = model.classify(dataset)

..note::
    It is also possible to show a progress bar by setting the :code:`show_progress_bar` parameter to :code:`True`
    and set the :code:`batch_size` to a desired value if the model does not fit on the GPU.

Calibration is usually not necessary in causal models.

Evaluation of the Predictions
-----------------------------

After we have classified the dataset, we can evaluate the predictions. The predictions are stored in the :code:`output` object and can be accessed as follows:

.. code-block:: python

    from sklearn.metrics import accuracy_score

    accuracy_score(dataset['label'], output.predictions)

..note::
    When using only a list of lists of label words in the verbalizer, it might be first necessary to adjust the predictions to the values used in the dataset.
    In this case, the predictions refer to the indices of the lists in the verbalizer.
    E.g.: :code:`[['negative'], ['positive']]` will produce predictions in the form of zeros and ones.


Using LLAMA
-----------

A model like LLAMA might need further arguments for initialization. These arguments can be passed  when initializing the model. In this example,
we use quantization and an access token for the huggingface hub:

.. code-block:: python

    import torch
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = CausalLM4Classification(
        "meta-llama/Meta-Llama-3.1-8B",
        prompt=prompt,
        tok_args = {"token":"<YOUR TOKEN>"},
        model_args = {"device_map":'auto', "quantization_config":bnb_config, "token":"<YOUR TOKEN>"})


The arguments :code:`tok_args` and :code:`model_args` are used to pass additional arguments when calling the :code:`from_pretrained` method under the hood.


Using a Fine-Tuned/Chatbot Model
--------------------------------

As mentioned previously, there are also many fine-tuned models available that are tuned to act like assistants similar to ChatGPT. These models
can also be used but require a a different approach. First, it is strongly recommended to explore the behavior of the model given a prompt. In this
example, we will use the :code:`HuggingFaceH4/zephyr-7b-beta` model.

As the objective is not to predict the next token but to be a helpfull assistant, we first need to look at the behavior when generating text.
We can do this quite easily by using the :code:`pipeline` method of the transformers library:

.. code-block:: python

    from transformers import pipeline

    model = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

    model(dataset[0]['text'] + "Is this a positive or negative review? Answer with 'positive' or 'negative'.")


Producing multiple outputs, we will see that then model is tuned to produce first two newsline characters, so we need to adapt our prompt accordingly:

.. code-block::python

    prompt = FVP(
        lambda e: f"""

        Product Review Classification into categories 'positive' or 'negative'.

        {e['text']}

        Is this a positive or negative review? Answer with 'positive' or 'negative'.\n\n""",
        Vbz({0: ["negative"], 1: ["positive"]})
    )

and initialize the model:

.. code-block:: python

    model = CausalLM4Classification(
        'HuggingFaceH4/zephyr-7b-beta',
        prompt=prompt
    )

Now, we can again classify the dataset and evaluate the predictions as shown above.

.. code-block:: python

    output = model.classify(dataset)

    accuracy_score(dataset['label'], output.predictions)