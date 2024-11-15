Getting Started
===============

.. image:: https://img.shields.io/github/v/release/varchasgopalaswamy/AutoUncertainties?label=Current%20Release&color
   :target: https://pypi.org/project/auto-uncertainties/
   :alt: GitHub Release

.. image:: https://img.shields.io/badge/Python-3.11%20%7C%203.12-ffed57?logo=python&logoColor=white
   :target: https://www.python.org/downloads/
   :alt: python

* AutoUncertainties is easy to install via ``pip``:

.. code:: bash

    pip install auto_uncertainties

* The Pint extensions can be installed and enabled alongside AutoUncertainties using:

.. code:: bash

   pip install auto_uncertainties[pint]

* The integration with `pandas` (still WIP) can be enabled by installing `pandas`, either separately or via:

.. code:: bash

    pip install auto_uncertainties[pandas]

* The documentation can be built by installing `auto_uncertainties` with the `[docs]` extension:

.. code:: bash

   pip install auto_uncertainties[docs]
   sphinx-build docs/source docs/build