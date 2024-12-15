.. Promptzl documentation master file, created by
   sphinx-quickstart on Sun Nov 17 23:34:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

.. While large generative models are often used and exhibit strong performance, they can be opaque and slow.
.. Promptzl works in batch mode, returns a softmax distribution, and is 100% transparent. All `causal <https://huggingface.co/models?pipeline_tag=text-generation>`_ 
.. and `masked <https://huggingface.co/models?pipeline_tag=fill-mask>`_ language models from the Hugging Face Hub are available in Promptzl.

.. .. note::
..    This project is in beta. Please report any issues on the
..    `GitHub issue tracker <https://github.com/LazerLambda/Promptzl/issues/new>`_.


How does it Work?
-----------------

Languages models predict a token given a certain context by calculating a distribution over the vocabulary. When classifying sentences,
only few tokens are relevant for the classification task. Extracting the tokens' logits and forming a distribution over them, allows turning
the LLM into a classifier. This is what Promptzl does. A simple example can be found here :ref:`tutorial_tldr`.


.. image:: _static/carbon_promptzl.png
   :align: center
   :scale: 50 %

This approach has been shown effective in mulitple research works (`Schick and Schütze, 2021 <https://aclanthology.org/2021.eacl-main.20/>`_; Schick et al., 2021, Hu et al., 2022).

.. This approach has been shown effective in multiple zero-shot settings (`Schick and Schütze, 2021 <https://aclanthology.org/2021.eacl-main.20/>`_; Schick et al., 2021, Hu et al., 2022). 
.. An overview of performance is provided here: TODO


.. Installation
.. ------------

.. Install promptzl via pip:

.. .. code-block:: bash

..    pip install -U promptzl

.. or through the GitHub repository:

.. .. code-block:: bash

..    pip install -U git+https://github.com/lazerlambda/promptzl.git


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
