Computation time
================

With the azimuthal band-limit :math:`N` of the fields and kernels on the rotation group fixed,
each of the four lifted convolutions costs :math:`\mathcal{O}(L^3)` operations at band-limit
:math:`L`, the cost of its forward and inverse transforms (Table III of the article).  The
script ``benchmarks/scaling.py`` checks this by timing one operator of each kind,
:math:`\mathcal{C}_{2\to1}`, :math:`\mathcal{C}_{2\to G}`, :math:`\mathcal{C}_{G\to2}` and
:math:`\mathcal{C}_{G\to G}`, with random kernels and :math:`N = 5`, at band-limits from 8 to
256.  As in the time column of the :doc:`equivariance benchmark <equivariance>`, each operator
runs end to end, with its transforms, compiled by JAX's JIT with compilation excluded, and its
time is the median of five runs.  Run it from the repository root:

.. code-block:: bash

    python benchmarks/scaling.py -o scaling.json --plot scaling.pdf > scaling.md

The script runs on a single CPU core, the fastest available, with the multithreading of XLA
off, so that the time counts the operations; ``--device default`` runs on JAX's default device
instead, such as a GPU, on which the time grows more slowly than :math:`L^3` until the
band-limit is large enough to occupy the hardware.  The script fits a power law to the times
from :math:`L = 32` up, prints a markdown table of the times and the fitted exponents, and
writes them as JSON and the plot as PDF, with the times on logarithmic axes against an
:math:`L^3` reference and the times divided by :math:`L^3`, which level off for cubic scaling.
Like the equivariance benchmark, it records the git commits of the code that ran.
``python benchmarks/scaling.py --help`` lists the options, including the band-limits, the
operators and the number of runs.
