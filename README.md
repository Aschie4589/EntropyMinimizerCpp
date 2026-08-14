# EntropyMinimizerCpp

This project provides a highly optimized implementation of an entropy minimization algorithm for the output of quantum channels.

## Installation

Here we provide instructions for running the algorithm. A C++ compiler and `cmake` need to be available on the system. On top of that, you will have to install the NVIDIA Toolkit and NVCC compiler. On Arch linux, they can be obtained with `sudo pacman -S gcc cmake cuda`. 

You will also need the CBLAS and Lapack libraries: `sudo pacman -S openblas lapack lapacke`. If you are compiling for MacOS, openblas and lapack are provided as part of Apple's Accelerate, and this project supports passing the flag BACKEND=accelerate to use these optimized libraries.

As a first step, clone this Git repository:
```bash
git clone https://github.com/Aschie4589/EntropyMinimizerCpp
cd EntropyMinimizerCpp
```

The project has the following 

Within the directory, make a new folder for building, and cd to it
```bash
mkdir build && cd build
```

Build the project using:
```bash
cmake ..
cmake --build .
```

This will create the `moe` executable in the `build` directory.

## Usage


## Folder structure:




```plaintext
quantum-additivity/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── .gitattributes
├── environment.yml              # Python env for post-processing
├── CMakeLists.txt               # root CMake build
│
├── cmake/                       # custom CMake modules, toolchains
│
├── src/                         # all C++ source code
│   ├── main/                    # entry points (executables)
│   │   ├── main_minimizer.cpp
│   │   ├── main_generate.cpp
│   │   └── main_test.cpp
│   │
│   ├── channel/                 # random channel generators, utilities
│   │   ├── channel_factory.cpp
│   │   ├── random_kraus.cpp
│   │   ├── perturbation.cpp
│   │   └── ...
│   │
│   ├── minimizer/               # minimization algorithms (still evolving)
│   │   ├── minimizer_base.cpp
│   │   ├── gradient_descent.cpp
│   │   ├── adam.cpp
│   │   ├── convergence.cpp
│   │   └── ...
│   │
│   ├── io/                      # input/output (JSON, HDF5, logging)
│   │   ├── io_hdf5.cpp
│   │   ├── io_json.cpp
│   │   ├── metadata_logger.cpp
│   │   └── run_folder.cpp
│   │
│   ├── utils/                   # misc utils, math helpers, rng, timers
│   │   ├── rng.cpp
│   │   ├── linear_algebra.cpp
│   │   ├── time_utils.cpp
│   │   └── ...
│   │
│   ├── version/                 # auto-generated build info
│   │   └── version_info.h.in
│   │
│   └── CMakeLists.txt
│
├── include/                     # public headers
│   ├── channel/
│   ├── minimizer/
│   ├── io/
│   ├── utils/
│   └── version/
│
├── build/                       # local build directory (git-ignored)
│
├── configs/                     # YAML/JSON experiment configs
│   ├── phase1.yml
│   ├── phase2.yml
│   ├── defaults.yml
│   └── test_small.yml
│
├── runs/                        # GPU run outputs, one folder per run
│   ├── phase1/
│   │   ├── 2025-10-24_N1024_d12_seed42/
│   │   │   ├── metadata.json
│   │   │   ├── results.h5
│   │   │   ├── log.txt
│   │   │   ├── checkpoints/
│   │   │   └── plots/
│   │   └── ...
│   ├── phase2/
│   └── phase3/
│
├── results/                     # aggregated results / derived data
│   ├── summaries/
│   │   ├── phase1_summary.parquet
│   │   ├── phase2_summary.parquet
│   │   └── ...
│   ├── figures/
│   └── reports/
│
├── scripts/                     # Python orchestration & analysis
│   ├── run_phase1.py            # launches C++ executable jobs
│   ├── analyze_phase1.py
│   ├── plot_gaps.py
│   ├── summarize_runs.py
│   └── utils/
│       ├── hdf5_utils.py
│       ├── plotting.py
│       └── metadata_tools.py
│
├── notebooks/                   # exploratory Jupyter notebooks
│   ├── inspect_best_channels.ipynb
│   ├── visualize_entropy.ipynb
│   └── ...
│
├── tests/                       # unit / regression tests for C++
│   ├── test_channel.cpp
│   ├── test_minimizer.cpp
│   └── test_io.cpp
│
├── data/                        # optional static data (constants, benchmarks)
│   └── sample_channels/
│
├── docs/                        # documentation, notes
│   ├── design.md
│   ├── roadmap.md
│   ├── experiments.md
│   └── references.bib
│
├── containers/                  # Docker or Singularity files for reproducibility
│   ├── Dockerfile
│   └── README.md
│
└── build-info/                  # automatically filled at compile time
    ├── build-info.txt
    ├── dependencies.txt
    └── system-info.txt
```