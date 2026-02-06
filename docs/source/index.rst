SuperDARN RST Documentation
============================

Welcome to the SuperDARN Radar Software Toolkit (RST) documentation. RST is a comprehensive toolkit for processing SuperDARN radar data with **CUDA GPU acceleration** for significant performance improvements.

.. image:: https://zenodo.org/badge/74060190.svg
   :target: https://zenodo.org/badge/latestdoi/74060190
   :alt: DOI

.. note::
   This version includes CUDA acceleration providing up to 16x speedup on GPU-enabled systems.

Quick Links
-----------

* :doc:`../tutorials/index` - Get started with RST
* :doc:`../user_guide/index` - Data processing workflows
* :doc:`../architecture/index` - Technical deep-dive
* :doc:`../guides/index` - Comprehensive references
* :doc:`../maintenance/index` - Deployment & operations

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   ../tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   ../user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   ../architecture/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   ../guides/index

.. toctree::
   :maxdepth: 2
   :caption: Operations

   ../maintenance/index

.. toctree::
   :maxdepth: 2
   :caption: Reports

   ../reports/index

Quick Start
-----------

**Docker (Recommended):**

.. code-block:: bash

   git clone https://github.com/SuperDARN/rst.git && cd rst
   docker build -t superdarn-rst .
   docker run --gpus all -it superdarn-rst

**Native Installation:**

.. code-block:: bash

   git clone https://github.com/SuperDARN/rst.git && cd rst
   source .profile.bash
   cd build && make

Key Features
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - CUDA Acceleration
     - Up to 16x speedup on GPU-enabled systems
   * - Complete Pipeline
     - Raw ACF → FITACF → Grid → Convection Maps
   * - Backward Compatible
     - Automatic CPU fallback, same API
   * - Python Bindings
     - Full Python interface with NumPy integration
   * - Docker Support
     - Easy deployment and reproducibility

Performance Highlights
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - Operation
     - CPU Time
     - CUDA Time
     - Speedup
   * - Data Processing
     - 45 ms
     - 2.8 ms
     - **16.1x**
   * - ACF Computation
     - 387 ms
     - 19 ms
     - **20.4x**
   * - Grid Operations
     - 234 ms
     - 32 ms
     - **7.3x**

Getting Help
------------

* **Issues**: `GitHub Issues <https://github.com/SuperDARN/rst/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/SuperDARN/rst/discussions>`_
* **DAWG**: `SuperDARN DAWG <https://superdarn.github.io/dawg/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
* Reporting issues
* Contributing code
* Testing CUDA optimizations

License and Citation
--------------------

The RST is open source software. When using SuperDARN data or the RST, please include the standard acknowledgement:

    The authors acknowledge the use of SuperDARN data. SuperDARN is a collection of radars funded by national scientific funding agencies of Australia, Canada, China, France, Italy, Japan, Norway, South Africa, United Kingdom and the United States of America.

For citation information, visit our `Zenodo page <https://doi.org/10.5281/zenodo.801458>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
