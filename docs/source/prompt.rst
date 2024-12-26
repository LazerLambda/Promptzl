.. _prompt-classes:

Prompt Classes
==============

Defining a prompt is crucial for effectively guiding the model in predicting the desired output. Promptzl offers two
approaches to defining dynamic prompts that serve as templates!

*Prompt-element-objects* are a safer but also more difficult-to-write option, while a *function-verbalizer-pair* is more
straightforward but might lead to errors more easily.

.. _prompt-element-objects:

*Prompt-Element-Objects*
------------------------
 - :class:`~promptzl.Txt` is used for text representation, e.g., :code:`Txt("I give you the following text: ")`
 - :class:`~promptzl.Key` is used for the keys that refer to a column in the corresponding 🤗-dataset, i.e. :code:`Key('text')`
   (the key ‘text’ is the default key and can usually be omitted) or use multiple Key objects if multiple columns are used
   (e.g. :code:`Txt("Premise: ") + Key('premise') + Txt("\\n\\nHypothesis: ") + Key('hypothesis') + Vbz(...`)
 - :class:`~promptzl.Vbz` for the verbalizer, e.g., :code:`Vbz({0: ["Good"], 1: ["Bad"]})`
   The verbalizer object is required to be in every prompt as it is used to extract the tokens of interest from
   the logits over the vocabulary. It is also possible to use multiple words for a class.

   .. note::

      Using prompt-elements-objects **offers the upside of truncating the prompt if the context length is exceeded and automatically
      adding the MASK token when using MLMs**. When truncation is applied, the data filled into the Key-placeholders is truncated to
      avoid cutting off crucial parts of the appended prompt. The context length is usually not a problem when dealing with modern LLMs;
      hence, an FnVbzPair prompt can be used instead of prompt-element objects.

.. _functoin_verbalizer_pair:

*Function-Verbalizer-Pair Class* (FnVbzPair)
--------------------------------------------

:class:`~promptzl.FnVbzPair` stands for the function-verbalizer-pair. In contrast to *prompt-element-objects*, a function constructing the prompt is directly defined with a verbalizer.
e.g., :code:`FnVbzPair(lambda x: f"{x['text']}\n", {0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})`.
The function receives a dictionary where the keys must refer to the columns in the dataset, and the values correspond to the respective observations.
The FnVbzPair class inherits from the prompt class and can be used for initializing the classifier classes (see. :class:`promptzl.modules.MaskedLM4Classification`
and :class:`promptzl.modules.CausalLM4Classification`)

.. note::

   Using the FnVbzPair object is easier to write but also requires more vigilance as the function must adhere to the requirements of the dataset. It is
   also impossible to truncate the prompt on the fly, which can result in indexing errors. The masked token **must** be set manually in the function when
   using MLMs.


Examples
--------

Here are some examples of handling the different approaches when constructing the prompt object.

Prompt-Element-Objects
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from promptzl import *

   verbalizer = Vbz({0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Tech"]})
   prompt = Txt("[Category:") + verbalizer + Txt("] ") + Key()


It is possible to print the prompt after initializing it to see what the final prompt looks like. Verbalizer and
key objects are also represented as follows:


.. code-block:: python

   from promptzl import *

   prompt = (
      Txt("I give you a movie review. Classify the sentiment! Here is the review:\n\n") +
      Key() +
      Txt("\n\nIs this review negative or positive?") +
      Vbz([['Negative','negative'], ['Positive', 'positive']]))
   prompt
   """I give you a movie review. Classify the sentiment! Here is the review:

   <text>

   Is this review negative or positive?<Vbz: [["Negative",...], ["Positive",...]]>"""


Function-Verbalizer-Pair
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from promptzl import *

   verbalizer = Vbz({0: ["good"], 1: ["bad"]})
   prompt = FnVbzPair(lambda x: f"{x['text']}\nIt was", verbalizer)

When using masked-language-modeling, the masked token must be set manually in the function, which can be done as follows:

.. code-block:: python

   from transformers import AutoTokenizer
   from promptzl import *

   tok = AutoTokenizer.from_pretrained("<an available model>")
   verbalizer = Vbz({0: ["good"], 1: ["bad"]})
   prompt = FnVbzPair(lambda x: f"{x['text']}\nIt was {tok.mask_token}", verbalizer)

Documentation
-------------

.. automodule:: promptzl.prompt
   :members:
   :undoc-members:
   :exclude-members: Img, Prompt
   :show-inheritance: