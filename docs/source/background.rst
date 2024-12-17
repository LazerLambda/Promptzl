Background
==========


Masked vs Causal LMs
--------------------

Language models (LM) are used to predict a specific token based on a vocabulary. For each token in the vocabulary, logits 
are computed from which a probability distribution is derived. This distribution is used to predict a token of interest. There are 
two main types of language models: masked language models (MLMs) and causal language models (CLMs).
The main difference between the two is that masked LMs predict a token given a context in both directions, while a causal LM
predicts a token given a context only based on previous tokens (which can also be a prompt) at the end of the sequence.


- Masked LMs can predict a token based on the contex:

   :code:`"Hello World! Promptzl is a [MASK] library."`

- Causal LMs can predict the next token based on the previous sequence:

   :code:`"Hello World! Promptzl is a [NEXT-TOKEN]"`

Hence, it is necessary to construct the prompts differently for the two model families. The prompt is crucial for the performance of the model.
The aim of this library is to extract the classifcation solely on this prediction, thus the quality of the prompt is of great importance.

.. _intuition-and-definition:

Intuition and Definition
------------------------

.. _intuition:

Intuition
^^^^^^^^^

While this is the main task of a language model is to predict a token to form natural language, it can
also be used for classification tasks by only extracting the logits of the tokens of interest. For example, if we want to
classify a text into the categories 'positive' or 'negative', we can enhance the data with a prompt that guides the model
to produce a specific output. Depending on the training of the model, this can be done by transforming the text into a cloze-task
or predicting the next token in the sequence which is also the label for the task.

In the previous section, the examples: :code:`"Hello World! Promptzl is a [MASK] library."` and :code:`"Hello World! Promptzl is a [NEXT-TOKEN]"`
where used to exemplify how language modeling can work. Here, we can already see how classification can work in this context. 
The model is asked to predict the token of interest and we can extract the label-token of interest. For example in :code:`"Hello World! Promptzl is a [MASK] library."`
we can already classify into :code:`good` and :code:`bad` just by extracting the logits at the indices for :code:`good` and :code:`bad` from the vocabulary
and get a distribution using the softmax function.


.. _formal-definition:

Formal Definition
^^^^^^^^^^^^^^^^^

The idea has been pioniereed by `Schick and Schütze, 2020a <https://aclanthology.org/2021.eacl-main.20>`_ where further few-shot learning was applied to improve the
classfication capabilities of masked-language models. However, even RoBERTa showed proficient zero-shot classification capabilities which is the focus of promptzl.

To transform an LM into a classifier, a **verbalizer** (:math:`\mathcal{V}`) must be defined which maps
the label words tokens (:math:`\mathcal{L}`) to the indices in the vocabulary (:math:`V`) allowing us to use the respective logits of the label words (:math:`\mathcal{V}: \mathcal{L} \rightarrow V`).

For the case of a verbalizer that only uses :code:`good` and :code:`bad`, the definition is the following:

.. math::

   \mathcal V(l) = \begin{cases}
			V_{\text{good}_0}, & l = \text{good}\\
         V_{\text{bad}_0}, & l = \text{bad}
		 \end{cases}

The function (:math:`\mathcal M(t| x)`) yields the logit for token (:math:`t`) given context (:math:`x`). Thus (:math:`\mathcal M(\mathcal V(l)| x)`)
can be used to get the logits for each label word using the verbalizer function. To further guide the model to predict the tokens of interest,
it is necessary to enhance the input text with the (:math:`\text{Prompt}`) function.

For obtaining the distribution, the softmax function can be applied:

.. math::

   \mathbb P(l) = \frac{\mathcal M(\mathcal V(l)| \text{Prompt}(x))}{\sum_{l' \in \mathcal L} \mathcal M(\mathcal V(l')| \text{Prompt}(x))}

This function passes the data through the model, the verbalizer extracts the logits of interest for each class and then a softmax is applied
to obtain the probabilities of the classes.
Additionally, it is also possible to use multiple label words where the logits can be averaged as introduced by `Schick and Schütze, 2020b <https://aclanthology.org/2020.coling-main.488/>`_,
which is left out here for the sake of brevity.


The Point for Causal Language Models
------------------------------------

However, using masked langauge models (MLMs) is limited to the fact that the prompt must be engineered in a very specific way to form a cloze-task
and MLMs are also usually limited by a short context size. With the advent of large open-source language models post ChatGPT, that usually contain
strong knowledge capabilities and also usually have longer context lengths, it is possible to use causal language models (CLMs) to classify text as well.
In MLMs it is also a problem to use words that are tokenized into multiple tokens as there is usually one MASK token. `Schick and Schütze, 2021 <https://aclanthology.org/2021.naacl-main.185/>`_
solved this by using multiple tokens and thus also predicting multiple tokens. However, this might lead to slower inference as many forward passes
are necessary when filling one token after another.

Here, in-context-learning can also be applied to further boost performance. It is also possible to use words that are tokenized into multiple tokens as only
the first tokens must be distinct for classification.
The *strength of the open source ecosystem* of large causal langauge models, is a strong point to use them for classification tasks as well.

.. _calibration:

Calibration
-----------

It has been found (`Zhao et al., 2021 <https://arxiv.org/abs/2102.09690>`_, `Hu et al., 2022 <https://aclanthology.org/2022.acl-long.158>`_) that
some tokens are more likely to be predicted than others. This can lead to a bias in the classification task. To counteract this, it is possible to
calibrate the output. Here, the probabilities are averaged and used to assess the prediction probability in context of the overall probability 
of the word being predicted. As we can see in the following example, this can lead to a stronger overall performance:


.. TODO: Get math for this


.. _further-reading:

Further Reading
---------------

Papers
^^^^^^

- `Schick and Schütze, 2020a <https://aclanthology.org/2021.eacl-main.20>`_
- `Schick and Schütze, 2020b <https://aclanthology.org/2020.coling-main.488/>`_
- `Schick and Schütze, 2021 <https://aclanthology.org/2021.naacl-main.185/>`_
- `Zhao et al., 2021 <https://arxiv.org/abs/2102.09690>`_
- `Hu et al., 2022 <https://aclanthology.org/2022.acl-long.158>`_
- `Ding et al., 2022 <https://aclanthology.org/2022.acl-demo.10/>`_

Repositories
^^^^^^^^^^^^

- `Repository with prompting papers <https://github.com/thunlp/PromptPapers>`_
- `OpenPrompt (similar library) <https://github.com/thunlp/OpenPrompt>`_