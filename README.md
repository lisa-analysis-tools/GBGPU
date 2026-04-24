# GBGPU

[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://mikekatz04.github.io/GBGPU)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138723.svg)](https://doi.org/10.5281/zenodo.17138723)

`GBGPU` is a GPU-accelerated version of the `FastGB` waveform which has been developed by Neil Cornish, Tyson Littenberg, Travis Robson, and Stas Babak. It computes gravitational waveforms for Galactic binary systems observable by LISA using a fast/slow-type decomposition. For more details on the original construction of `FastGB` see [arXiv:0704.1808](https://arxiv.org/abs/0704.1808).

The current version of the code is very closely related to the implementation of `FastGB` in the LISA Data Challenges' Python code package. The waveform code is entirely Python-based. It is about 1/2 the speed of the full C version, but much simpler in Python for right now. There are also many additional functions including fast likelihood computations for individual Galactic binaries, as well as fast C-based methods to combine waveforms into global fitting templates. 

The code is CPU/GPU agnostic. CUDA and NVIDIA GPUs are required to run these codes for GPUs.

See the [documentation](https://mikekatz04.github.io/GBGPU/) for more details. This code was designed for [arXiv:2205.03461](https://arxiv.org/abs/2205.03461). If you use any part of this code, please cite [arXiv:2205.03461](https://arxiv.org/abs/2205.03461), its [Zenodo page](https://zenodo.org/records/16999246), [arXiv:0704.1808](https://arxiv.org/abs/0704.1808), and [arXiv:1806.00500](https://arxiv.org/abs/1806.00500). 

To install the latest version of `gbgpu` using `pip`, simply run:

```sh
# For CPU-only version
pip install gbgpu

# For GPU-enabled versions with CUDA 11.Y.Z
pip install gbgpu-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install gbgpu-cuda12x
```

To know your CUDA version, run the tool `nvidia-smi` in a terminal a check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

Now, in a python file or notebook:

```py3
import gbgpu
```

You may check the currently available backends:

```py3
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
...     print(f" - Backend '{backend}': {"available" if gbgpu.has_backend(backend) else "unavailable"}")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda': unavailable
 - Backend 'gpu': unavailable
```

Note that the `cuda` backend is an alias for either `cuda11x` or `cuda12x`. If any is available, then the `cuda` backend is available.
Similarly, the `gpu` backend is (for now) an alias for `cuda`.

If you expected a backend to be available but it is not, run the following command to obtain an error
message which can guide you to fix this issue:

```py3
>>> import gbgpu
>>> gbgpu.get_backend("cuda12x")
ModuleNotFoundError: No module named 'gbgpu_backend_cuda12x'

The above exception was the direct cause of the following exception:
...

gbgpu.cutils.BackendNotInstalled: The 'cuda12x' backend is not installed.

The above exception was the direct cause of the following exception:
...

gbgpu.cutils.MissingDependencies: GBGPU CUDA plugin is missing.
    If you are using gbgpu in an environment managed using pip, run:
        $ pip install gbgpu-cuda12x

The above exception was the direct cause of the following exception:
...

gbgpu.cutils.BackendAccessException: Backend 'cuda12x' is unavailable. See previous error messages.
```

Once GBGPU is working and the expected backends are selected, check out the [examples notebooks](https://github.com/mikekatz04/GBGPU/tree/master/examples/)
on how to start with this software.

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- A Python version supported by [scikit-build-core](https://github.com/scikit-build/scikit-build-core) (>=3.7 as of Jan. 2025)

If you want to enable GPU support in GBGPU, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`).


### Installation instructions using conda

We recommend to install GBGPU using conda in order to have the compilers all within an environment.
First clone the repo

```
git clone https://github.com/mikekatz04/GBGPU.git
cd GBGPU
```

Now create an environment (these instructions work for all platforms but some
adjustements can be needed, refer to the
[detailed installation documentation](https://mikekatz04.github.io/GBGPU) for more information):

```
conda create -n gbgpu_env -y -c conda-forge --override-channels |
    cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate gbgpu_env
```

Then we can install locally for development:
```
pip install -e '.[dev, testing]'
```

### Installation instructions using conda on GPUs and linux
Below is a quick set of instructions to install the GBGPU package on GPUs and linux.

```sh
conda create -n gbgpu_env -c conda-forge gbgpu-cuda12x python=3.12
conda activate gbgpu_env
```

Test the installation device by running python
```python
import gbgpu
gbgpu.get_backend("cuda12x")
```

### Running the installation

To start the from-source installation, ensure the pre-requisite are met, clone
the repository, and then simply run a `pip install` command:

```sh
# Clone the repository
git clone https://github.com/mikekatz04/GBGPU.git
cd GBGPU

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://mikekatz04.github.io/GBGPU). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/mikekatz04/GBGPU/issues)
or contact the developers through other means.



### Running the Tests

The tests require a few dependencies which are not installed by default. To install them, add the `[testing]` label to GBGPU package
name when installing it. E.g:

```sh
# For CPU-only version with testing enabled
pip install gbgpu[testing]

# For GPU version with CUDA 12.Y and testing enabled
pip install gbgpu-cuda12x[testing]

# For from-source install with testing enabled
git clone https://github.com/mikekatz04/GBGPU.git
cd GBGPU
pip install '.[testing]'
```

To run the tests, open a terminal in a directory containing the sources of GBGPU and then run the `unittest` module in `discover` mode:

```sh
$ git clone https://github.com/mikekatz04/GBGPU.git
$ cd GBGPU
$ python -m gbgpu.tests  # or "python -m unittest discover"
...
----------------------------------------------------------------------
Ran 20 tests in 71.514s
OK
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop GBGPU and produce documentation, install `gbgpu` from source with the `[dev]` label and in `editable` mode:

```
$ git clone https://github.com/mikekatz04/GBGPU.git
$ cd GBGPU
pip install -e '.[dev, testing]'
```

This will install necessary packages for building the documentation (`sphinx`, `pypandoc`, `sphinx_rtd_theme`, `nbsphinx`) and to run the tests.

The documentation source files are in `docs/source`. To compile the documentation locally, change to the `docs` directory and run `make html`.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/GBGPU/tags).

## Contributors

A (non-exhaustive) list of contributors to the GBGPU code can be found in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Citation

Please make sure to cite GBGPU papers and the GBGPU software on [Zenodo](https://zenodo.org/records/17138723).
We provide a set of prepared references in [PAPERS.bib](PAPERS.bib). There are other papers that require citation based on the classes used. For most classes this applies to, you can find these by checking the `citation` attribute for that class.  All references are detailed in the [CITATION.cff](CITATION.cff) file.

