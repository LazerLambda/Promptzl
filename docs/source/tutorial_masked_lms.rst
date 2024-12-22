.. _tutorial_masked_lms:

Tutorial - Masked-Language-Model-Based Classifiers
==================================================

While causal language models predict the next token in an autoregressive manner from left to right, masked language
models (MLMs) can predict a token based on the surrounding context. A word in the sequence is masked, and the model
is trained to predict this masked token based on the context to the left and right. This technique can be leveraged to
turn pre-trained MLMs into zero-shot classifiers by turning the data into cloze-question tasks.

Example Dataset
---------------

As promtzl is intended to be used along the 🤗-transformers and -dataset libraries, we first have to initialize an example dataset. For this
tutorial, we will use the AGNews dataset. In this task, the model must label news into broader categories:

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")['test'].select(range(1000))

We will only use the first 1000 examples of the dataset for brevity.

Defining a Prompt and a Verbalizer
----------------------------------

During pre-training, the model is only fed raw text, from which it is trained to predict the missing token. **No further fine-tuning is applied,
so the model cannot be instructed, unlike fine-tuned CLMs like ChatGPT**. Thus, the task must be constructed so that just one word in the
sequence condenses the information about the classification task while maintaining its resemblance to the training data.

In AG News, we have the categories 'World', 'Sports', 'Business', and 'Tech', so we can initialize the verbalizer:

.. code:: python

    from promptzl import *

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})

For the prompt we can use the following pattern:

.. code:: python

    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

Here, we use the *prompt-element-objects* as MLM models usually have a significantly shorter context length, and we
further need to add the respective mask token from the tokenizer where the *prompt-element-objects* automatically take care of.

However, this is just one selected prompt; it is also possible to define further prompts:

.. code:: python

    prompt = Txt('[ Topic: ') + Vbz(vbz_agnews) + Txt('] ') + Key('text')
    prompt1 = Txt("A ") + Vbz(vbz_agnews) + Txt(' news: ') + Key('text')
    prompt2 = Key('text') + Txt(' This topic is about ') + Vbz(vbz_agnews) + Txt('.')
    prompt3 = Txt("[Category:") + Vbz(vbz_agnews) + Txt('] ') + Key('text')

All these prompts stem from `Schick and Schütze, 2020 <https://aclanthology.org/2021.eacl-main.20>`_ where they found
that the :code:`prompt` works the best for this task, so we will further only use this prompt.

.. note::
    As just one mask token is to be predicted, it is crucial to define a verbalizer where the label words
    translate to only single tokens, as it can sometimes happen that more complex words are tokenized into
    different subtokens.

Loading the Model
-----------------

After loading the dataset and defining the verbalizer and prompt, we can finally load the model:

.. code:: python

   model = MaskedLM4Classification(
        'roberta-large',
        prompt=prompt
    )

Classifying the Dataset
-----------------------

...and start to classify the dataset:

.. code-block:: python

    output = model.classify(dataset)

.. note::
    It is also possible to show a progress bar by setting the :code:`show_progress_bar` parameter to :code:`True`
    and set the :code:`batch_size` to a desired value if the model does not fit on the GPU.

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

Calibration
-----------

It has been found that some tokens are generally less likely to be predicted, causing the model to be biased towards more often recurring tokens in
the label word set (more details in :ref:`calibration`). To counteract this, it is possible to calibrate the output. Here, the probabilities are averaged
and used to assess the prediction probability in the context of the predicted word's overall average probability. As we can see in the following example,
this can lead to a stronger overall performance:

.. code-block:: python

    pred_cali = model.calibrate_output(output).predictions
    sum([int(prd == lbl) for prd, lbl in zip(pred_cali, dataset['test']['label'])]) / len(pred_cali)
    # 0.8315789473684211

Furthermore, it is also possible to use the :meth:`~promptzl.utils.calibrate` method that can be used with 
a tensor of probabilities.

