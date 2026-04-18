# Emission Validation Outputs

This folder contains two synthetic validation cases:

- `case1_uniform_source/`: uniform-source baseline
- `case2_analytic_loop_top/`: analytic loop-top morphology case

Each case includes:

- `morphology_*.png`: theory vs reconstructed maps at selected frequencies
- `spectrum_comparison.png`: integrated spectrum comparison
- `validation_product.h5`: theory and reconstruction cubes plus parameter maps and component-aware references
- `summary.json`: compact numerical summary with parameter, morphology, and spectral metrics

Metric groups:

- `parameter_metrics`: MAE/RMSE and relative errors for nonthermal density and power-law index
- `morphology_metrics`: per-frequency image relative error, brightness centroid offset, peak-value error, and plane-integrated flux error
- `spectrum_band_metrics`: low/mid/high band RMS spectral error and integrated-flux error

Component-aware reference groups inside `validation_product.h5`:

- `references/truth` and `references/reconstruction`: thermal-only, low-density no-Razin proxy, optically thin proxy, and background-only curves
- `diagnostics/truth` and `diagnostics/reconstruction`: absorption impact and Razin impact curves

Case summaries:

- `case1_uniform_source`: RMS spectrum error = 2.1283e-01, accepted local power-law fits = 384
- `case2_analytic_loop_top`: RMS spectrum error = 3.1006e-01, accepted local power-law fits = 1034
