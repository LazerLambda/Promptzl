Prompt Classes
==============

The prompt classes are used to construct the prompt in a easy way by simply concatenating the objects through the `+` operator.
There are two main approaches to initialize the Classifier object: 

Using the *Prompt-Element-Objects*
----------------------------------

 - :class:`~promptzl.Txt` for text representation, i.e. :code:`Txt("I give you the following text: ")`
 - :class:`~promptzl.Key` for the keys that refer to a column in the corresponding huggingface dataset, i.e. :code:`Key('text')`
   (the key 'text' is the default key and can usually be omitted) or use multiple Key objects if multiple columns are used
   (e.g. :code:`Txt("Premise: ") + Key('premise') + Txt("\n\nHypothesis") + Key('nHypothesis')`)
 - :class:`~promptzl.Vbz` for the verbalizer, i.e. :code:`Vbz({0: ["Good"], 1: ["Bad"]})`
   The verbalizer object is requried to be in every prompt as it is used to extract the tokens of interest from the logits over the
   vocabulary. It is also possible to use multiple words for a class.

   .. note::
      Using prompt-elements-objects has the advantage of truncating the final prompt in case the context length is exceeded. In this case,
      the data at the position of the keys is truncated so that the other parts of the prompt still remain within the context window.
      This is especially important for masked language models, as the masked token **must** be in text.
 
Using the *Function-Verbalizer-Pair Class* (FVP)
------------------------------------------------

 - :class:`~promptzl.FVP` for the function-verbalizer-pair, i.e. :code:`FVP(lambda x: f"{x['text']}\n", {0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})`
   The FVP object is used to extract the tokens of interest from the logits over the vocabulary. The function is used to extract the
   tokens from the dataset.

   .. note::
      Using the FVP object is easier to write but also requires more vigilance as the function must accept a dictionary with the keys
      refering to the mention in the function. It is also not possible to truncate the prompt on-the-fly which can result in indexing errors.
      For masked language models, the masked token **must** be set manually in the function.



Examples
--------

In the following some examples are given on how to use the prompt classes:


Prompt-Element-Objects
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from promptzl import *

   verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
   prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()


It is possible to print the prompt after initializing it to see how the final prompt looks like. Verbalizer and
key objects are also represented as following:


.. code-block:: python

   from promptzl import *

   prompt = (
      Txt("I give you a movie review. Classify the sentiment! Here is the review:\n\n") +
      Key() +
      Txt("\n\nIs this review negative or positive?") +
      Vbz([['Negative','negative'], ['Positive', 'positive']]))
   prompt ==\
   """I give you a movie review. Classify the sentiment! Here is the review:

   <text>

   Is this review negative or positive?<Vbz: [["Negative",...], ["Positive",...]]>"""


Function-Verbalizer-Pair
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from promptzl import *

   verbalizer = Vbz({0: ["good"], 1: ["bad"]})
   prompt = FVP(lambda x: f"{x['text']}\nIt was", verbalizer)

When using masked_language_modeling, the masked token must be set manually in the function. This can be done as following:

.. code-block:: python

   from transformers import AutoTokenizer
   from promptzl import *

   tok = AutoTokenizer.from_pretrained("<an available model>")
   verbalizer = Vbz({0: ["good"], 1: ["bad"]})
   prompt = FVP(lambda x: f"{x['text']}\nIt was {tok.mask_token}", verbalizer)

Documentation
-------------

.. automodule:: promptzl.prompt
   :members:
   :undoc-members:
   :exclude-members: Img, Prompt
   :show-inheritance: