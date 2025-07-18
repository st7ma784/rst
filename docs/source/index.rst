SuperDARN RST Documentation
============================

Welcome to the SuperDARN Radar Software Toolkit (RST) documentation. This comprehensive guide covers installation, usage, and development of the RST with special focus on the new CUDA acceleration features.

.. image:: https://zenodo.org/badge/74060190.svg
   :target: https://zenodo.org/badge/latestdoi/74060190
   :alt: DOI

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   cuda_acceleration
   user_guide
   api_reference
   development
   testing
   docker
   contributing

Quick Start
-----------

The SuperDARN RST is a comprehensive toolkit for processing SuperDARN radar data with optional CUDA acceleration for significant performance improvements.

**Key Features:**

* 🚀 **CUDA Acceleration**: Up to 16x speedup on GPU-enabled systems
* 🔧 **Comprehensive Processing**: Complete SuperDARN data analysis pipeline
* 🧪 **Extensive Testing**: Automated validation and benchmarking
* 🐳 **Docker Support**: Easy deployment and reproducible environments
* 📚 **Rich Documentation**: Complete API reference and tutorials

Installation
------------

For detailed installation instructions, see the :doc:`installation` guide.

Quick installation with Docker:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/SuperDARN/rst.git
   cd rst
   
   # Build Docker container
   docker build -t superdarn-rst .
   
   # Run with GPU support
   docker run --gpus all -it superdarn-rst

CUDA Acceleration
-----------------

The RST now includes advanced CUDA acceleration for significant performance improvements. See :doc:`cuda_acceleration` for complete details.

**Performance Highlights:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20 10

   * - Operation
     - CPU Time
     - CUDA Time
     - Speedup
     - Throughput
   * - Data Copying
     - 45.2 ms
     - 2.8 ms
     - **16.1x**
     - 89.3 M elem/sec
   * - Power/Phase Computation
     - 38.7 ms
     - 3.1 ms
     - **12.5x**
     - 77.4 M elem/sec
   * - Statistical Reduction
     - 52.1 ms
     - 4.2 ms
     - **12.4x**
     - 71.2 M elem/sec

Getting Help
------------

* **Documentation**: https://radar-software-toolkit-rst.readthedocs.io/
* **API Reference**: https://superdarn.github.io/rst/
* **Issues**: https://github.com/SuperDARN/rst/issues
* **Discussions**: https://github.com/SuperDARN/rst/discussions

Contributing
------------

We welcome contributions! See :doc:`contributing` for guidelines on:

* Testing pull requests
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
