# Reduction-des-matrices-SPD

## Geomstats connectome data

The files in `train/` are the official connectome training data distributed
with [geomstats](https://github.com/geomstats/geomstats). Each row contains
378 upper-triangular entries, which represent a 28 x 28 connectome.

The raw connectomes are correlation matrices and are not automatically SPD.
`prepare_spd_data.py` constructs a symmetric zero-diagonal matrix from each
row and applies the matrix exponential. Since the exponential of a real
symmetric matrix is strictly positive definite, the generated
`train/train_spd.npy` array contains SPD matrices.

Install the runtime dependencies and generate the matrices with:

```text
python -m pip install numpy pandas scipy
python prepare_spd_data.py
```

The geomstats repository provides the training files, but it does not include
the 119,748-row Kaggle test file expected by `final.ipynb`. That test file must
be obtained separately; it should not be fabricated from the training data.

The final notebook also contains a subject-level comparison using the 86
training subjects:

- Euclidean PCA on flattened SPD matrices
- Tangent PCA using the matrix logarithm
- The existing autoencoder on tangent features
- The existing MIAF model on tangent features

Each method produces a 2-D embedding and is evaluated with 10 stratified
80/20 splits and 5-nearest-neighbour classification. Results are saved to
`spd_subject_reduction_results.csv`.

## Modular project structure

The reusable implementation is now separated from the presentation notebook:

```text
spd_reduction/
  data.py          # CSV loading, symmetric reconstruction, exp/log SPD maps
  pca_methods.py   # Euclidean PCA and tangent PCA
  autoencoder.py   # autoencoder and MIAF models/training
  evaluation.py    # repeated stratified kNN evaluation
scripts/
  run_spd_experiments.py
```

Run the complete modular experiment from the repository root with:

```text
python -m pip install -r requirements.txt
python -m scripts.run_spd_experiments
```

On Windows, if `python` resolves to the Microsoft Store alias, use the
installed interpreter explicitly:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.run_spd_experiments
```

The original `final.ipynb` is retained as an exploratory record. New
experiments should use the modules and runner above so that PCA, tangent PCA,
autoencoder, and MIAF share the same data preparation and evaluation code.

The runner also generates presentation-ready figures in `figures/`:

```text
figures/
  01_pipeline.png             # data -> SPD -> reduction -> evaluation
  02_spd_eigenvalues.png      # positive eigenvalue verification
  03_subject_embeddings.png   # four 2-D method embeddings
  04_accuracy_comparison.png  # mean accuracy +/- standard deviation
  05_full_results_summary.png # data size, protocol, SPD check, and results
```

The current run produced these repeated 80/20 5-NN results:

| Method | Mean accuracy | Standard deviation |
|---|---:|---:|
| PCA on SPD | 0.5611 | 0.1302 |
| Tangent PCA | 0.6889 | 0.0831 |
| SPD Autoencoder | 0.4944 | 0.1203 |
| SPD MIAF | 0.4611 | 0.0705 |

To compare the modular autoencoder with the matching `AE_baseline_4L`
configuration from `final.ipynb` on the same raw FNC features and the same
64/22 stratified split, run:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.compare_autoencoder_implementations
```

The comparison is saved to `autoencoder_comparison.csv`. The historical
notebook result for `AE_baseline_4L` was 0.7273 with kNN on that split.
The comparison figure is saved to `figures/06_autoencoder_comparison.png`.

To compare PCA and tangent PCA directly on the MLSP/COBRE data, run:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.compare_pca_mlsp
```

This evaluates both methods with kNN, logistic regression, RBF-SVM, and random
forest over 10 stratified 80/20 splits. It saves
`mlsp_pca_tangent_pca_results.csv` and
`figures/07_mlsp_pca_vs_tangent_pca.png`.
