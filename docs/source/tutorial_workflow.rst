Tutorial - Workflow
===================

Many real-world problems include datasets that are not labeled and it is hence not possible to use supervised learning.
Yet LLMs have shown remarkable performance in common-sense knowledge and can be used to classify text without labeled data
using text generation. However, text generation through proprietary APIs is often slow and does not return desirable properties
of a classifier like the softmax distribution, access to weights or a faster batch mode as only one token is of interest.

Promptzl tackles this problem by providing a simple interface to leverage the power of LLMs for classification tasks.



Masked vs Causal LMs
--------------------

.. Pro MLM Pro Causal
.. context left and right, - context from left
.. - cloze task, difficult to find prompt ,+ easier to find prompt

While causal LMs like ChatGPT are well known for their strong performance in recent years, masked language models were strong contenders
for many natural language tasks as well (and still are). The main difference between the two is that masked LMs predict a token given a context
in both directions, while a causal LM predicts a token given a context only based on previous tokens (which can also be a prompt) at the end of
the sequence.


- Masked LMs can predict a token based on the contex:

   :code:`"Hello World! Promptzl is a [MASK] library."`

- Causal LMs can predict the next token based on the previous sequence:

   :code:`"Hello World! Promptzl is a [NEXT-TOKEN]"`

Hence, it is necessary to construct the prompts differently for the two model families. The prompt is crucial for the performance of the model.
The aim of this library is to extract the classifcation solely on this prediction, thus the quality of the prompt is of great importance.


The MaskedLM4Classification Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~promptzl.MaskedLM4Classification` class is build upon the transformers library and can be used with all available models that can be used in :code:`AutoModelForMaskedLM`.
The class :meth:`~promptzl.MaskedLM4Classification.forward` method is overwritten to return the logits of the position of the masked token. The verbalizer is used to extract the tokens of interest.

The CausalLM4Classification Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~promptzl.CausalLM4Classification` class is build upon the transformers library and can be used with all available models that can be used in :code:`AutoModelForCausalLM`.
It returns the logits of the token which is predicted next and extracts the tokens of interest with the verbalizer.

The application of the two classes is the same. The only difference is the model family that is used.

Finding a Good Prompt
---------------------

Finding a good prmopt can be challenging. It depends on which model family is used, as the prediction is different (as described above).
It is recommended to first run the model with a few examples and then adjust the prompt accordingly. For example, the model 'Zephyr' usually
generates two new line characters before starting with the answer. This little detail can be crucial for the classification task as the model
might first heavily weigh on predicting these formal characters while the actual answer is generated at a later stage of the sequence.

Example:
 SHOW \n\n and without \n\n

Another crucial impact comes through certain properties of the tokenizer. If there is no space between characters, the tokenizer might prepend
an internal special character for the tokenization process. This can lead to a different prediction than expected. For example, the tokenizer of
'roberta-large' ....

As we see here, it is crucial to first examine which characters the model produces with a certain prompt. This can be tested in the following way:


TODO

.. test if verbalizer labels are just ONE token!


Tuning the Prompt (if Validation Data is Available)
---------------------------------------------------

In the case that validation data is available, it is possible (and recommended) to tune the prompt and investigate shortcomings.

Tips for Causal LMs:
^^^^^^^^^^^^^^^^^^^^

 -  In the case of causal language models, that were fine-tuned to be assistants, it is possible to provide examples in the prompt
    or describe what the task is about. 
 -  The specific output can be defined in the prompt (e.g. "Only return 'Positive' or 'Negative'").

Tips for Masked LMs:
^^^^^^^^^^^^^^^^^^^^

 -  The model is not trained to answer questions or assist the user. It is trained to predict the masked token given a context.
    Thus providing a context in which the model finds it very easy to predict the masked token based on the data is crucial. A good
    example comes from the work of Schick and Schütze (cite):

    .. code-block:: python

        verbalizer = Vbz([["World"], ["Sports"], ["Business"], ["Tech"]])
        prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()


Calibration
-----------

It has been found (`Zhao et al., 2021 <https://arxiv.org/abs/2102.09690>`_, `Hu et al., 2022 <https://aclanthology.org/2022.acl-long.158>`_) that
some tokens are more likely to be predicted than others. This can lead to a bias in the classification task. To counteract this, it is possible to
calibrate the output. Here, the probabilities are averaged and used to assess the prediction probability in context of the overall probability 
of the word being predicted. As we can see in the following example, this can lead to a stronger overall performance:

.. code-block:: python

    from promptzl import *
    from datasets import load_dataset

    dataset = load_dataset("SetFit/ag_news")

    verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
    prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()

    model = MaskedLM4Classification("roberta-large", prompt)
    output = model.classify(dataset['test'], show_progress_bar=True)
    predictions = output.predictions

    sum([int(prd == lbl) for prd, lbl in zip(predictions, dataset['test']['label'])]) / len(predictions)
    # 0.7986842105263158

    pred_cali = model.calibrate_output(output).predictions
    sum([int(prd == lbl) for prd, lbl in zip(pred_cali, dataset['test']['label'])]) / len(pred_cali)
    # 0.8315789473684211

Here, the :meth:`~promptzl.LLM4ClassificationBase.calibrate_output` can be used to calibrate the output. 
The method returns a new :class:`~promptzl.LLM4ClassificationOutput` object with the calibrated predictions
and the calibrated distributions in the same format.

Furthermore, it is also possible to use the :meth:`~promptzl.utils.calibrate` method that can be used with 
a tensor of probabilities.


Limitations
-----------

While performance is often strong in few class settings, the model stops to function when there are two many classes. This is due to the fact
that only a certain amount of detail can be expressed by one token. A good example is the Banking-77 dataset, where the model has to predict
77 classes with multiple labels sharing the same tokens.
