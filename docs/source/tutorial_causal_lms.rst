.. _tutorial_causal_lms:

Tutorial - Causal-Language-Model-Based Classifiers
==================================================

Causal language models (CLMs) are trained to predict the next token in a sequence based on the previous context (next-token objective),
which requires us always to set the verbalizer at the end of the sequence. The prompt is crucial for the model's performance, and the
underlying model's training objective must be considered.

In the current open-source causal LLM ecosystem, there are two groups of causal LMs:
the base models, the ones that were only trained on the next-token objective, and those that were further fine-tuned on instructions to act like
assistants akin to ChatGPT. Both models can be used for classifications, but the characteristics of each must be kept in mind. Usually, the latter
models have a name indicating the further tuning like '-instruct' (:code:`HuggingFaceTB/SmolLM2-1.7B` vs. :code:`HuggingFaceTB/SmolLM2-1.7B-Instruct`
where the latter is fine-tuned).

Example Dataset
---------------

As promptzl is intended to be used along the 🤗-transformers and -dataset libraries, we first have to initialize an example dataset.
We will use the IMDB dataset for this tutorial, which is a binary classification task. The dataset is loaded as follows:

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("mteb/imdb")['test'].select(range(1000))

For brevity, we will only use the first 1000 examples of the dataset.

Defining a Prompt and a Verbalizer
----------------------------------

As we are interested in classifying only positive or negative samples, we need to define a prompt that guides the model to produce the desired output and
verbalizer (how the verbalizer works is described in :ref:`formal-definition`) that extracts the logits of interest. The prompt is defined as follows:

.. code-block:: python

    from promptzl import *

    prompt = FVP(
        lambda e: f"""
        Movie Review Classification into categories 'positive' or 'negative'.

        '{e['text']}'='""",
        Vbz({0: ["negative"], 1: ["positive"]})
    )

.. note::
    It is also possible to use *Prompt-Element-Objects* (see :ref:`intuition-and-definition`) or provide a list of label words to the :code:`Vbz` object.
    However, using a dictionary allows us to return the dictionary keys to the predictions list.

Finding a Good Prompt
^^^^^^^^^^^^^^^^^^^^^
**Constructing the prompt in a form that guides the model to predict only the tokens we are interested in (label words) with a high probability is crucial**. 

If labeled data is available, it is possible to leverage in-context learning and enhance the prompt with these samples, which can be sampled from the dataset.
In the following, we will take a look at an example of a natural language inference task, **independent of the IMDB task further used in this tutorial**,
where a premise and a hypothesis are given and the model must classify into the categories of :code:`entailment`, :code:`neutral` or :code:`contradiction`:

.. code-block:: python

    f"""
    Natural Language Inference. Given is a premise and a hypothesis which must be classified as 'entailment', 'neutral' and 'contradiction'.

    Premise: 'How do you know? All this is their information again.' - Hypothesis: 'This information belongs to them.'='entailment'
    Premise: 'But a few Christian mosaics survive above the apse is the Virgin with the infant Jesus, with the Archangel Gabriel to the right (his companion Michael, to the left, has vanished save for a few feathers from his wings).' - Hypothesis: 'Most of the Christian mosaics were destroyed by Muslims.  '='neutral'
    Premise: 'At the end of Rue des Francs-Bourgeois is what many consider to be the city's most handsome residential square, the Place des Vosges, with its stone and red brick facades.' - Hypothesis: 'Place des Vosges is constructed entirely of gray marble.'='contradiction'

    Premise:  '{e['premise']}' - Hypothesis: '{e['hypothesis']}'='"""

This prompt works as follows: First, a natural language description of the task and the categories the model has to label the instances to is given.
Then, some examples (one for each class) are provided to show the model how to classify the data (if labeled data is available).
Finally, the data is provided in the same form as the previous examples but is cut off at the point where the label for the sought observation
is to be predicted. With the previous examples, the model is now more likely to predict a label of code:`entailment`, :code:`neutral` or :code:`contradiction`.


Loading the Model
-----------------

After we have defined the prompt, we can load the model. In this case, we will use the CLM :code:`HuggingFaceTB/SmolLM2-1.7B`, which is relatively
small and can also fit on smaller GPUs. The model is loaded as follows:

.. code-block:: python

   model = CausalLM4Classification(
        'HuggingFaceTB/SmolLM2-1.7B',
        prompt=prompt
    )

We have set up everything and can start classifying the dataset.

Classifying the Dataset
-----------------------

To classify the dataset, we can use the model's :code:`classify` method. This method returns an :class:`promptzl.utils.LLM4ClassificationOutput` object containing the predictions and the distribution.
It is also possible to get the logits for each class. The method is called as follows:

.. code-block:: python

    output = model.classify(dataset)

.. note::
    It is also possible to show a progress bar by setting the :code:`show_progress_bar` parameter to :code:`True`
    and set the :code:`batch_size` to a desired value if the model does not fit on the GPU.

:ref:`Calibration` is usually not necessary in causal models.

Evaluation of the Predictions
-----------------------------

After we have classified the dataset, we can evaluate the predictions. The predictions are stored in the :code:`output` object and can be accessed as follows:

.. code-block:: python

    from sklearn.metrics import accuracy_score

    accuracy_score(dataset['label'], output.predictions)

.. note::
    When using List[List[str]] instead of Dict[str, List[str]] in the verbalizer, it might be necessary first to adjust the predictions to the values used in the dataset.
    In this case, the predictions refer to the indices of the lists in the verbalizer.
    E.g.: :code:`[['negative'], ['positive']]` will produce predictions in the form of zeros and ones.


Using Proprietary Models
------------------------

A model like LLAMA might need further arguments for initialization. These arguments can be passed when initializing the model. In this example,
we use quantization and an access token for the Hugging Face hub:

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

.. _tutorial_causal_lms_fine_tuned:

Using a Fine-Tuned/Chatbot Model
--------------------------------

As mentioned previously, many fine-tuned models are also available that are tuned to act like assistants similar to ChatGPT. These models
can also be used but require a different approach. Firstly, it is strongly recommended to explore the model's behavior given a prompt. In
this example, we will use the :code:`HuggingFaceH4/zephyr-7b-beta` model.

As the objective is not to predict the next token but to be a helpful assistant, we first need to examine the behavior when generating text.
We can do this quite easily by using the :code:`pipeline` method of the transformers library.

.. code-block:: python

    from transformers import pipeline

    model = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

    model(dataset[0]['text'] + "Is this a positive or negative review? Answer with 'positive' or 'negative'.")


Producing multiple outputs, we will see that the model is tuned to predict first two newline characters, so we need to adapt our prompt accordingly:

.. code-block:: python

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

    from sklearn.metrics import accuracy_score

    output = model.classify(dataset)

    accuracy_score(dataset['label'], output.predictions)