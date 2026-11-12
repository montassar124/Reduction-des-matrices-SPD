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

This evaluates both methods with kNN and RBF-SVM over 10 stratified 80/20
splits. It saves
`mlsp_pca_tangent_pca_results.csv` and
`figures/07_mlsp_pca_vs_tangent_pca.png`.

## Local geometry with multiple tangent references

The local-geometry experiment adapts the number of references to the data. It
tests one or more possible numbers of reference points and keeps multiple
references only when the Riemannian silhouette is positive and the within-group
dispersion decreases sufficiently. Otherwise, it keeps one global reference.
It computes one SPD center per retained group and assigns each subject to its
nearest center using the affine-invariant Riemannian distance. Each local tangent vector is then
parallel-transported to the global reference tangent space, and the
transported offset of its local center is added before classification. This
alignment is necessary because tangent coordinates from different references
cannot be compared directly.

Run it from the repository root:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.run_local_geometry
```

The experiment saves:

- `local_geometry_centers.csv`: size and distance statistics for each local
  reference;
- `local_geometry_center_distances.csv`: pairwise Riemannian distances
  between the local references;
- `local_geometry_subject_assignments.csv`: subject-to-reference assignments;
- `local_geometry_accuracy.csv`: classification comparison with global tangent
  PCA;
- `figures/08_local_geometry_multi_reference.png`: distance and local-chart
  visualizations.
- `figures/09_local_geometry_accuracy.png`: accuracy comparison between the
  global and multi-reference tangent representations.

To test the method in controlled situations, generate synthetic SPD data with
different levels of separation and noise:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.generate_geometry_comparison
```

This creates `synthetic_geometry_cases.npz` containing the generated SPD
matrices and labels, `synthetic_geometry_comparison.csv`, and
`figures/10_synthetic_geometry_comparison.png`. The cases are: one compact
region, two moderately separated regions, two strongly separated regions, and
two regions with high noise. It also includes a fourth-region stress test,
where the adaptive method must detect four references. Each case contains
1000 SPD matrices and compares classic PCA, global tangent PCA, and adaptive
references using ten stratified 50/50 train/test splits. All preprocessing
and reference selection are fitted on the training half only.

The paper-based comparison uses the saved matrices from the other paper
protocol in `paper_synthetic_spd_cases.npz` when that file is available. Run:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.compare_paper_spd_reductions
```

It evaluates classic PCA, tangent PCA, adaptive references, and LLE with ten
stratified 50/50 train/test splits and writes
`paper_spd_reductions_comparison.csv`.

To visualize the spherical log-SPD data and the 3-D geometry produced by each
method, including a transparent triangular mesh, run:

```text
C:\Users\Globaltech\AppData\Local\Programs\Python\Python312\python.exe -m scripts.plot_spherical_geometries_3d
```

The figures are saved as `figures/14_spherical_geometries_modere.png` and
`figures/14_spherical_geometries_fort.png`.

Benchmark terminal output now reports accuracy and six-neighbour preservation
as percentages. The CSV files contain `Accuracy (%)`, `Accuracy std (%)`, and
`Neighbours preserved (%)` for direct comparison.
