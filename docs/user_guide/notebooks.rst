Notebooks
=========

The ``notebooks`` directory of the repository holds two `marimo <https://marimo.io>`_
notebooks, which run in the browser on `molab <https://molab.marimo.io>`_ straight from
GitHub:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Notebook
     - Contents
     - Run
   * - `lifted_convolutions.py <https://github.com/astro-informatics/s2conv/blob/main/notebooks/lifted_convolutions.py>`_
     - Each convolution of s2conv, checked numerically against the closed forms of the
       article, a literal evaluation on a sampling of the rotation group and the real-space
       definitions.
     - |molab-lifted|
   * - `real_fields.py <https://github.com/astro-informatics/s2conv/blob/main/notebooks/real_fields.py>`_
     - The lifted convolutions applied to real spin fields of the Earth and the sky.
     - |molab-real|

.. |molab-lifted| image:: https://marimo.io/molab-shield.svg
   :alt: Open in molab
   :target: https://molab.marimo.io/github/astro-informatics/s2conv/blob/main/notebooks/lifted_convolutions.py

.. |molab-real| image:: https://marimo.io/molab-shield.svg
   :alt: Open in molab
   :target: https://molab.marimo.io/github/astro-informatics/s2conv/blob/main/notebooks/real_fields.py

Each notebook lists its dependencies at the top of the file, as inline script metadata
(`PEP 723 <https://peps.python.org/pep-0723/>`_), which molab reads to install them.  When
the notebook runs without the repository, as in molab, ``real_fields.py`` downloads its
fields from the ``notebooks/data`` directory on GitHub.

To run a notebook locally, install s2conv with the notebook extras (see :doc:`install`) and
open it with marimo, from the repository root:

.. code-block:: bash

   pip install "s2conv[notebooks]"
   marimo edit notebooks/lifted_convolutions.py

Alternatively, marimo can install the dependencies listed in the notebook into an isolated
environment, with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   marimo edit --sandbox notebooks/lifted_convolutions.py
