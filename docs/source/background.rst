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


Intuition
---------
While this is the main task of a language model is to predict a token to form natural language, it can
also be used for classification tasks by only extracting the logits of the tokens of interest. For example, if we want to
classify a text into the categories 'positive' or 'negative', we can enhance the data with a prompt that guides the model
to produce a specific output. Depending on the training of the model, this can be done by transforming the text into a cloze-task
or predicting the next token in the sequence which is also the label for the task.

.. TODO EXample


Formal Definition
-----------------
The idea has been pioniereed by Schick and Schütze in their work on ## where further few-shot learning was applied to improve the
classfication capabilities of masked-language models. To transform an LM into a classifier, a **verbalizer** must be defined which maps
the tokens of interest to the logits of the model.

.. Vbz = {'positive' -> Voc_{positive}, 'negative' -> Voc_{negative}}

It is also necessary to define a prompt that guides the model to produce the desired output.
This technique is refered to as a Prompt-Verbalizer-Pair (PVP), which is used to obtain a function of the following form:

.. P_V(y = Pattern(x)) = softmax(Vbz(M(Pattern(x))))

where x is the input text, y is a class and Pattern is the function that transforms the output into a cloze-task.

.. P_V(y = Pattern(x)) = softmax(Vbz(M(Pattern(x))))

This function passes the data through the model, the verbalizer extracts the logits of interest for each class and then a softmax is applied
to obtain the probabilities of the classes.
Additionally, it is also possible to use multiple label words where the logits can be averaged as introduced by (Schick schütze later).
More links to these branch of works, is provided in ##.


Argument for Causal Language Models
-----------------------------------
However, using masked langauge models (MLMs) is limited to the fact that the prompt must be engineered in a very specific way to form a cloze-task
and MLMs are also usually limited by a short context size. With the advent of large open-source language models post ChatGPT, that usually contain
strong knowledge capabilities and also usually have longer context lengths, it is possible to use causal language models (CLMs) to classify text as well.
In MLMs it is also a problem to use words that are tokenized into multiple tokens as there is usually one MASK token. Schick schütze solved this by using multiple tokens
and thus also predicting multiple tokens. However, this might lead to slower inference as many forward passes are necessary when filling one token after another.

Here, in-context-learning can also be applied to further boost performance. It is also possible to use words that are tokenized into multiple tokens as only
the first tokens must be distinct for classification.
The *strength of the open source ecosystem* of large causal langauge models, is a strong point to use them for classification tasks as well.


Further Reading
---------------