Promptzl Documentation
======================

Promptzl is an easy-to-use library for turning state-of-the-art LLMs into old-school
pytorch-based, zero-shot classifiers based on the 🤗-transformers library.

   - 💪 Zero-shot classification
   - 🤖 `Causal <https://huggingface.co/models?pipeline_tag=text-generation>`_ and `masked <https://huggingface.co/models?pipeline_tag=fill-mask>`_ LMs from the Hugging Face Hub are available
   - ⚡ Fast and easy to use (just a few lines of code needed)
   - 📦 Promptzl works with batches
   - 🔎 All models are transparent on your device
   - 📈 Properties of old-school classifiers included
   - 🚀 No need for proprietary APIs

Links
-----

 .. centered::
   `GitHub <https://github.com/LazerLambda/Promptzl>`_ | `PyPI <https://pypi.org/project/promptzl>`_ | `GitHub Issue Tracker <https://github.com/LazerLambda/Promptzl/issues/new>`_

Installation
------------

.. code-block:: bash

   pip install -U promptzl

How does it Work?
-----------------

Languages models predict a token given a certain context by calculating a distribution over the vocabulary. When classifying sentences,
only few tokens are relevant for the classification task. Extracting the tokens' logits and forming a distribution over them, allows turning
the LLM into a classifier. This is what Promptzl does. A simple example can be found here :ref:`tutorial_tldr`.


.. image:: _static/carbon_promptzl.png
   :align: center
   :scale: 50 %

This approach has been shown effective in mulitple research works (`Schick and Schütze, 2021 <https://aclanthology.org/2021.eacl-main.20/>`_; Schick et al., 2021, Hu et al., 2022).

Background
----------
.. toctree::
   :maxdepth: 1

   background


Documentation
-------------

.. toctree::
   :maxdepth: 1

   prompt
   lm_classifiers
   utils

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorial_init
   tutorial_causal_lms.rst
   tutorial_masked_lms.rst
