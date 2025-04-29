.. title:: Fluctuation Analysis Tools documentation

.. module:: StatTools

========================================
Fluctuation Analysis Tools documentation
========================================

What is Fluctuation Analysis Tools?
===================================

Fluctuation Analysis Tools is a library for creation, processing and analysis of long-term dependent timeseries datasets.

Features
========

- Creation of long-term dependent timeseries datasets.
- Processing of timeseries datasets.
- Analysis of timeseries datasets.

Install
=======

.. tab-set::
    :class: sd-width-content-min

    .. tab-item:: pip

        .. code-block:: bash

            pip install FluctuationAnalysisTools

    .. tab-item:: Sources

        .. code-block:: bash

            git clone https://github.com/Digiratory/StatTools.git
            cd StatTools
            pip install .

Citation
========
If you use Fluctuation Analysis Tools in your research, please cite our publications as follows.

* For DFA (Detrended Fluctuation Analysis):
.. code-block:: bibtex

    @article{10.3389/fninf.2023.1101112,
        author     = {Bogachev, M. and Sinitca, A. M. and Grigarevichius, K. and Pyko, N. and Lyanova, A. and Tsygankova, M. and Davletshin, E. and Petrov, K. and Ageeva, T. and Pyko, S. and Kaplun, D. and Kayumov, A. and Mukhamedshina, Y.},
        title      = {Video-based marker-free tracking and multi-scale analysis of mouse locomotor activity and behavioral aspects in an open field arena: A perspective approach to the quantification of complex gait disturbances associated with Alzheimer's disease},
        journal    = {Frontiers in Neuroinformatics},
        year       = {2023},
        volume     = {17},
        doi        = {10.3389/fninf.2023.1101112},
        art_number = {1101112},
    }
* For DPCCA (Detrended Partial-Cross-Correlation Analysis):
.. code-block:: bibtex

    @article{10.1016/j.bspc.2022.104409,
        author     = {Bogachev, M.I. and Lyanova, A.I. and Sinitca, A. M. and Pyko, S.A. and Pyko, N.S. and Kuzmenko, A.V. and Romanov, S.A. and Brikova, O.I. and Tsygankova, M. and Ivkin, D.Y. and Okovityi, S.V. and Prikhodko, V.A. and Kaplun, D.I. and Sysoev, Y.I. and Kayumov, A.R.},
        title      = {Understanding the complex interplay of persistent and antipersistent regimes in animal movement trajectories as a prominent characteristic of their behavioral pattern profiles: Towards an automated and robust model based quantification of anxiety test data},
        journal    = {Biomedical Signal Processing and Control},
        year       = {2023},
        volume     = {81},
        doi        = {10.1016/j.bspc.2022.104409},
        art_number = {104409},
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
    :hidden:

    modules
    installing
    changelog
