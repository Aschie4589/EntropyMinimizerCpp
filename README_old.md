# ENTROPY MINIMIZER

This is my attempt at porting the EntropyMinimizer written in Python to C++.
Ideally, it should make use of LAPACK and CBLAS to perform matrix operations fast (and on multiple CPU cores!)

## Known issues
So far, none! Wheev...

## How to compile
To compile, open a terminal in the main folder and execute:
```batch
make FLAGS=value
```
Available flags are:
- `PLATFORM`. Can be `apple` or `linux`. Determines the compiler used, and the architecture (`arm64` or `x86_64`).
- `LAPACK`. Can be `accelerate`, `openblas` or `mkl`. Will make use of the Apple Accelerate, OpenBLAS or IntelMKL versions of BLAS and LAPACK.

**Important:** In order to compile, please make sure that the location of the relevant LAPACK libraries is set correctly (check `makefile`).

The compiled binary will be called `moe`.

## Usage
[This part was generated automatically]
### General Syntax
```bash
moe <command> [options]
```

---
## Commands

### 1. `kraus`: Generate Kraus Operators
Generates Kraus operators for a quantum channel using different methods.

#### Subcommand: `haar`
Generates Kraus operators using Haar random unitaries.

**Options:**
- `-N <int>`: Dimension of the Hilbert space (**required**).
- `-d <int>`: Number of Kraus operators (**required**).
- `--output`, `-o <path>`: Path to save the Kraus operators (**required**).
- `--logging`, `-l`: Enable logging (optional; default: `false`).
- `--silent`, `-s`: Disable printing (optional; default: `false`).

**Example:**
```bash
moe kraus haar -N 4 -d 3 --output kraus_operators.txt
```

---

### 2. `singleshot`: Single-Shot Entropy Minimization
Performs single-shot entropy minimization for quantum channels.

**Description:**
With a randomly initialized vector, the algorithm runs until convergence.

#### Required Arguments:
- `-k`, `--kraus <path>`: Path to stored Kraus operators (**required**).

#### Other Arguments:
- `--save`, `-S`: Save the final vector (optional; default: `false`).
- `-i`, `--iters <int>`: Maximum number of iterations for the minimizer (optional).

#### Printing Arguments:
- `--logging`, `-l`: Enable logging (optional; default: `false`).
- `--silent`, `-s`: Disable printing (optional; default: `false`).

**Example:**
```bash
moe singleshot -k kraus_operators.txt -i 100 --save --logging
```

---

### 3. `multishot`: Multi-Shot Entropy Minimization
Performs multi-shot entropy minimization for quantum channels.

**Description:**
Run the algorithm multiple times with different starting vectors.

#### Required Arguments:
- `-k`, `--kraus <path>`: Path to stored Kraus operators (**required**).

#### Other Arguments:
- `--save`, `-S`: Save the final vector (optional; default: `false`).
- `-i`, `--iters <int>`: Number of iterations for the minimizer (optional).
- `-a`, `--atts <int>`: Number of minimization attempts (optional).

#### Printing Arguments:
- `--logging`, `-l`: Enable logging (optional; default: `false`).
- `--silent`, `-s`: Disable printing (optional; default: `false`).

**Example:**
```bash
moe multishot -k kraus_operators.txt -i 100 -a 10 --logging
```

---

## Notes
- The program automatically displays help messages for any command by using the `--help` flag. For example:
  ```bash
  moe kraus --help
  ```