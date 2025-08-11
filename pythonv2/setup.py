"""
SuperDARN GPU-Accelerated Processing Package
============================================

A modern Python implementation of SuperDARN radar data processing
with CUDA/GPU acceleration using CUPy.
"""

from setuptools import setup, find_packages

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="superdarn-gpu",
    version="2.0.0",
    author="SuperDARN DAWG",
    author_email="darn-dawg@isee.nagoya-u.ac.jp",
    description="GPU-accelerated SuperDARN radar data processing",
    long_description=__doc__,
    long_description_content_type="text/plain",
    url="https://github.com/SuperDARN/rst",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "cupy>=12.0.0",
        "h5py>=3.6.0",
        "netCDF4>=1.5.0",
        "xarray>=0.20.0",
        "dask>=2022.1.0",
        "numba>=0.56.0",
        "cython>=0.29.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "pandas>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyterlab>=3.4.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "bokeh>=2.4.0",
            "cartopy>=0.20.0",
            "ipywidgets>=8.0.0",
        ],
        "ml": [
            "cuml>=22.0.0",
            "scikit-learn>=1.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "superdarn-process=superdarn_gpu.tools.cli:main",
            "superdarn-benchmark=superdarn_gpu.tools.benchmarks:main",
            "superdarn-validate=superdarn_gpu.tools.validation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "superdarn_gpu": [
            "data/*.yaml",
            "data/*.json",
            "kernels/*.cu",
            "kernels/*.cuh",
        ]
    },
)