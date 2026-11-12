# SPD Geometry Delivery Package

This package contains the reproducible notebooks, scripts, figures, and CSV
results for the SPD dimensionality-reduction experiments.

## Contents

- `SPD_integrated_results.ipynb`: main presentation notebook for the MLSP/COBRE
  SPD experiment, including the extracted adaptive multi-reference code.
- `scripts/compare_paper_spd_reductions.py`: 50/50 comparison on the paper
  synthetic SPD benchmark.
- `scripts/benchmark_notebook_lie_lle_synthetic.py`: comparison where the
  notebook friend's Lie-LLE workflow is used only for Lie-LLE, while the
  previous train-only 50/50 workflow is used for PCA and adaptive references.
- `scripts/run_eth80_notebook_benchmark.py`: reproduces the friend's ETH-80
  workflow: five training objects and five test objects per category, `k=6`,
  one-nearest-neighbour classification, and Lie-LLE fitted on train plus test.
- `scripts/test_eth80_spd.py`: stricter ETH-80 SPD benchmark with GPU support
  when CUDA is available.
- `spd_reduction/multi_reference.py`: reusable adaptive multi-reference tangent
  representation.
- `figures/`: comparison and geometry figures.
- `results/`: CSV outputs from the verified runs.

## Installation

From this directory on Windows:

```powershell
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

The scripts use NumPy, SciPy, scikit-learn, pandas, matplotlib, Pillow, and
PyTorch. CUDA is optional; `test_eth80_spd.py` uses CUDA automatically when
PyTorch detects an NVIDIA GPU.

## ETH-80 data

The archive intentionally does not include the raw ETH-80 image directory.
Download it separately from:

`https://github.com/chenchkx/ETH-80`

Place the extracted directory beside this package as:

```text
SPD_geometry_delivery/
ETH-80/
  1/
  2/
  ...
  8/
```

The ETH-80 workflow groups 41 views into each object, resizes images to 20x20,
and creates a 401x401 Gaussian SPD descriptor with covariance regularization.
The minimum eigenvalue is checked before evaluation.

## Complete testing protocol

### ETH-80 SPD construction

ETH-80 contains 80 objects from 8 categories, with 10 objects and 41 views
per category. Each view is converted to grayscale and resized to 20x20,
giving a 400-dimensional image vector. For each object, the 41 vectors are
converted into a 401x401 Gaussian SPD descriptor:

1. compute the image-set mean \(m\);
2. compute the covariance \(C\);
3. regularize it with
   \[
   C \leftarrow C + 10^{-3}\operatorname{tr}(C)I;
   \]
4. construct
   \[
   M =
   \begin{bmatrix}
   C + mm^T & m\\
   m^T & 1
   \end{bmatrix};
   \]
5. apply the determinant normalization used by the friend's notebook;
6. verify that every eigenvalue of \(M\) is strictly positive.

The strict GPU script reports a minimum eigenvalue of approximately
`4.118e-05`.

### ETH-80 friend's notebook workflow

For each of 10 repetitions, the 10 objects of every category are split into
5 training objects and 5 test objects. Thus each split contains 40 training
SPD matrices and 40 test SPD matrices.

The friend's Lie-LLE workflow uses:

- `k=6` nearest neighbours;
- target dimension `d=30`;
- log-Euclidean distances;
- local Gram-matrix reconstruction weights;
- the \(N \times N\) LLE eigenproblem;
- 1-nearest-neighbour classification using log-Euclidean distance;
- Lie-LLE fitted on train plus test, exactly as in the supplied notebook.

Fitting Lie-LLE on train plus test is transductive: test samples participate
in the neighbourhood graph and embedding. This reproduces the notebook but
is not a strict inductive machine-learning evaluation.

The configuration cell in the notebook defines `K_NEIGH=6`, `TARGET_D=30`,
`N_REPEATS=10`, and `SEED=0`. The reproduced run below uses the same number
of repetitions and protocol; individual random split values can differ if
the random-number implementation or seed handling differs.

### Strict ETH-80 comparison

The strict comparison fits the transformation on training objects only and
uses 5-NN after 2-D reduction. It is not directly comparable with the
transductive notebook workflow.

| Method | Mean accuracy | Std | Neighbour preservation |
|---|---:|---:|---:|
| Classic PCA | 47.25% | 3.05% | 80.58% |
| Tangent PCA | 45.25% | 6.66% | 57.71% |
| Adaptive references | 65.25% | 6.37% | 51.46% |
| Lie-LLE | 49.75% | 7.02% | 51.25% |

### ETH-80 notebook-workflow comparison

The following table uses the friend's workflow for the comparison, including
5 train and 5 test objects per category, target dimension 30, and 1-NN:

| Method | Mean accuracy | Std |
|---|---:|---:|
| Lie-LLE notebook | 91.00% | 4.44% |
| Tangent PCA | 89.25% | 3.55% |
| Adaptive references | 77.00% | 5.75% |
| Classic PCA | 62.50% | 7.26% |

The original attached notebook displays a Lie-LLE mean of 93.75% ± 2.56%.
Small differences are expected when the exact split order, seed handling, or
floating-point implementation differs. A single split can reach 97.5%; that
is not the 10-repetition mean.

### Synthetic paper data with overlap

For each synthetic sample, a random matrix \(A\) is symmetrized and
exponentiated:
\[
S=\exp\left(\frac{A+A^T}{2}\right).
\]
The result is SPD because the exponential of a real symmetric matrix has
strictly positive eigenvalues. Each case contains 1000 matrices per class,
2000 total, with a 50/50 train/test split and 10 repetitions.

The distant case is excluded from the final comparison. The retained cases
are moderate overlap and strong overlap:

| Case | Classic PCA | Tangent PCA | Adaptive references | Lie-LLE |
|---|---:|---:|---:|---:|
| Moderate overlap | 98.22% | 97.00% | 97.13% | 93.46% |
| Strong overlap | 84.27% | 80.03% | 83.29% | 63.71% |

Neighbour preservation for the same 50/50 benchmark, measured over the six
nearest neighbours, is:

| Case | Classic PCA | Tangent PCA | Adaptive references | Lie-LLE |
|---|---:|---:|---:|---:|
| Moderate overlap | 21.32% | 5.40% | 23.21% | 3.89% |
| Strong overlap | 16.75% | 4.33% | 18.77% | 3.83% |

### Spherical log-SPD data

For the spherical experiment, the symmetric logarithm is normalized before
the exponential:
\[
L_{\mathrm{sphere}}=\frac{L}{\|L\|_F},
\qquad
S=\exp(L_{\mathrm{sphere}}).
\]
Therefore every logarithmic descriptor has Frobenius norm one. Two classes
are generated around different directions on this sphere, with moderate or
strong overlap, then mapped to SPD matrices.

The spherical benchmark uses 1000 matrices per class, 50/50 train/test,
10 repetitions, and `k=6`:

| Spherical case | Classic PCA | Tangent PCA | Adaptive references | Lie-LLE notebook |
|---|---:|---:|---:|---:|
| Moderate overlap | 97.86% | 95.81% | 95.08% | 97.75% |
| Strong overlap | 97.38% | 94.27% | 94.29% | 97.97% |

Neighbour preservation over six neighbours is:

| Spherical case | Classic PCA | Tangent PCA | Adaptive references |
|---|---:|---:|---:|
| Moderate overlap | 4.27% | 4.71% | 5.05% |
| Strong overlap | 4.54% | 4.97% | 4.91% |

The spherical results should be interpreted separately from the paper
interval-based data: they use a different data-generation geometry and are
not expected to have the same accuracy profile.

## Reproduce

Run from the package parent directory:

```powershell
$python = "C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe"
& $python -m scripts.benchmark_notebook_lie_lle_synthetic
& $python -m scripts.run_eth80_notebook_benchmark
& $python -m scripts.test_eth80_spd
```

The main notebook can be opened in Jupyter or VS Code. The extracted
multi-reference implementation is available as:

```python
from spd_reduction.multi_reference import fit_multi_reference_tangent
```

## Interpretation

- Classic PCA optimizes global Euclidean variance.
- Tangent PCA linearizes the SPD data around one global reference.
- Adaptive references use several local SPD tangent charts and align them to a
  common tangent space.
- Lie-LLE preserves local reconstruction structure in the log-Euclidean space.

Accuracy and neighbour preservation measure different properties. A method can
classify well while distorting local neighbourhoods, or preserve neighbours
without maximizing class separation.

### Terminal output

The benchmark scripts print percentages directly. `Accuracy (%)` is the mean
classification accuracy, `Accuracy std (%)` is its standard deviation, and
`Neighbours preserved (%)` is the mean percentage of the six original nearest
neighbours retained after reduction. The same columns are saved in the CSV
files.
