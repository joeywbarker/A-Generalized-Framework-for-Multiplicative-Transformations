# Multiplication as Reflection, Rotation and Scaling

[![DOI](https://zenodo.org/badge/928180086.svg)](https://doi.org/10.5281/zenodo.24409524)

A decomposition framework from the reals to matrices.

Multiplication by a fixed factor is split into a **magnitude** part (expansion, reduction, isometry, nullification) and an **orientation** part (identity, reflection, rotation). The note proves this for the real numbers (`R* ≅ {±1} × (0,∞)`), the complex numbers (polar form) and matrices (polar decomposition and SVD).

The underlying mathematics is classical; the contribution is one consistent classification with complete proofs, aimed at teaching and at relating the elementary and advanced cases.

## Verify

Every numerical claim and worked example in the paper is checked by a script (requires Python 3 and `numpy`):

```
python3 verify_claims.py
```

## Build

```
latexmk -pdf main.tex
```

## Cite

See `CITATION.cff`. Licensed under CC BY 4.0 (see `LICENSE`).
