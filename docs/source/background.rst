.. _background:

Background
==========

Language models are computational systems designed to understand, generate, and manipulate human language.
They are trained on vast amounts of text data to predict the likelihood of a specific token given a context.
This prediction can be modified to form a classifier, which is explained in this section in more detail.

Masked vs. Causal LMs
---------------------

Language models (LM) predict a specific token from a vocabulary based on a particular context. Logits are computed for
each token in the vocabulary, from which a probability distribution is derived. This distribution is used to predict a token of interest. There are 
two main types of language models: masked language models (MLMs) and causal language models (CLMs).
The main difference between the two is that MLMs predict a token given a context in both directions (see example), while a CLM
predicts a token given only the previous context:


- Masked LMs can predict a token based on the left and right context:

   :code:`"Hello World! Promptzl is a [MASK] library."`

- Causal LMs can predict the next token based on the previous sequence:

   :code:`"Hello World! Promptzl is a [NEXT-TOKEN]`

.. _intuition-and-definition:

Idea and Definition
-------------------

.. _intuition:

Idea
^^^^

While the main task of a language model is to predict a token to form a human-readable sequence, it can also be used for classification
tasks by only extracting the logits of the tokens of interest. For example, we want to classify a text into the categories 'good' or
'bad'. In that case, we can enhance the data with a prompt that guides the model in producing the output 'good' or 'bad'.
Depending on the model's pre-training, this can be done by transforming the text into a cloze-task
or predicting the final token in the sequence, where the missing token serves as the label for the classification task.

Previously, the examples: :code:`"Hello World! Promptzl is a [MASK] library."` and :code:`"Hello World! Promptzl is a [NEXT-TOKEN]"`
where used to exemplify how language modeling can work. Here, we can already see how classification can work in this context. 
The model is asked to predict the token of interest, and we can extract the respective logits for the classes we are interested in.
In :code:`"Hello World! Promptzl is a [MASK] library."`, we can select the logits for :code:`good` and :code:`bad` and compute the
distribution by applying the softmax function.


.. _formal-definition:

Formal Definition
^^^^^^^^^^^^^^^^^

The idea was pioneered by `Schick and Schütze, 2020a <https://aclanthology.org/2021.eacl-main.20>`_ where further few-shot learning was
applied to improve the classification capabilities of masked-language models. However, even RoBERTa demonstrated proficient zero-shot
classification abilities, confirming that LLMs possess strong capabilities in this area.

To transform an LLM into a classifier, a **verbalizer** (:math:`\mathcal{V}`) must be defined which maps
the label words tokens (:math:`\mathcal{L}`) to the indices in the vocabulary (:math:`V`), allowing us to use the respective logits of
the label words (:math:`\mathcal{V}: \mathcal{L} \rightarrow V`).

For the case of a verbalizer that only uses :code:`good` and :code:`bad`, the definition is the following:

.. math::

   \mathcal V(l) = \begin{cases}
			V_{\text{good}_0}, & l = \text{good}\\
         V_{\text{bad}_0}, & l = \text{bad}
		 \end{cases}

The function (:math:`\mathcal M(t| x)`) yields the logit for token (:math:`t`) given context (:math:`x`). Thus (:math:`\mathcal M(\mathcal V(l)| x)`)
can be used to get the logits for each label word using the verbalizer function. To further guide the model to predict the tokens of interest,
enhancing the input text with the (:math:`\text{Prompt}`) function is necessary. The prompt function augments the data to produce the desired output.
In case of a sentiment classification task, where the labels are 'good' and 'bad', the prompt function might look like:

.. math::

   \text{Prompt}(x) = \left [x, \text{ It was [MASK].} \right ]

For obtaining the distribution, the softmax function can be applied:

.. math::

   \mathbb P(l|x)_{\text{Prompt}} = \frac{\mathcal M(\mathcal V(l)| \text{Prompt}(x))}{\sum_{l' \in \mathcal L} \mathcal M(\mathcal V(l')| \text{Prompt}(x))}

This function passes the data through the model, the verbalizer extracts the logits of interest for each class, and then a softmax is applied
to get the probabilities of the classes.
Additionally, it is also possible to use multiple label words where the logits can be averaged as introduced by `Schick and Schütze, 2020b <https://aclanthology.org/2020.coling-main.488/>`_,
which is left out here for the sake of brevity.

The Point for Causal Language Models
------------------------------------

However, using MLMs is limited to the fact that the prompt must be engineered in a particular way to form a cloze-task
and MLMs are also usually limited by a short context size (except `ModernBERT <https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb>`_).
With the advent of large open-source language models post ChatGPT, which typically contain strong real-world knowledge capabilities
and have longer context lengths, it is now easier to use CLMs to classify text.

In MLMs, it is also a problem to use words that consist of multiple tokens (e.g., the `RoBERTa <https://arxiv.org/abs/1907.11692>`_
tokenizer will tokenize "Transportation" into :code:`[19163, 41067]`), as there is usually one MASK token for prediction.
`Schick and Schütze, 2021 <https://aclanthology.org/2021.naacl-main.185/>`_ solved this by using multiple mask-tokens and thus also
predicting in multiple forward passes, which has the downside of leading to slower inference.

Another point for CLMs is the possibility of applying in-context learning. The prompt provides a few labeled samples explaining how the model
classifies specific data points. The downside is that it requires labeled examples. Still, a more detailed explanation or instructions might also
be beneficial when using CLMs, especially when they are further tuned to work as an assistant like ChatGPT.

Furthermore, the variety and size of the open-source ecosystem for CLMs are also strong points for using CLMs.

.. _calibration:

Calibration
-----------

It has been found (`Zhao et al., 2021 <https://arxiv.org/abs/2102.09690>`_) that some tokens are more likely to be predicted than others,
which can lead to a bias in the classification task. For example, in a sentiment classification task, the token 'good' might be predicted
with a higher probability than 'bad' even though the input is obviously negative. However, it is possible to calibrate the output.

A contextualized prior is computed, capturing the probabilities based on the prompt for the tokens we are interested in. This allows us
to set the predictions in relation to the average probabilities in this dataset. The contextualized prior is defined as:

.. math::

   \overline{\mathbb{P}(l|X)} = \frac{1}{n} \sum_{x \in X} \mathbb{P}(l|x)_{\text{Prompt}}


Where :math:`n` is the number of samples. This step has been done in advance in `Zhao et al., 2021 <https://arxiv.org/abs/2102.09690>`_ and
`Hu et al., 2022 <https://aclanthology.org/2022.acl-long.158>`_. However, in this library, the probabilities are computed posteriorly
based on the predictions. This step omits the estimation of the probabilities in advance.

The predictions can be calibrated as follows using the contextualized prior:

.. math::

   \tilde{\mathbb P(l)} = \frac{\mathbb P(l|x)_{\text{Prompt}}}{\overline{\mathbb{P}(l|X)}} / \left [ \sum_{l' \in \mathcal L} \frac{\mathbb P(l'|x)_{\text{Prompt}}}{\overline{\mathbb{P}(l'|X)}} \right ]

The function :meth:`~promptzl.utils.calibrate` can be used to calibrate the raw predictions (as tensors). Additionally,
the entire :class:`promptzl.utils.LLM4ClassificationOutput` object can be calibrated as well using :meth:`promptzl.modules.LLM4ClassificationBase.calibrate_output`.
`Hu et al., 2022 <https://aclanthology.org/2022.acl-long.158>`_ showed that contextualized calibration can lead to a stronger performance in MLMs.


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