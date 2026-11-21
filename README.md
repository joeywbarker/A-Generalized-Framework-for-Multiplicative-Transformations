# Size and Direction: A Simple Classification of Multiplication by Any Real Number

[![DOI](https://zenodo.org/badge/928180086.svg)](https://doi.org/10.5281/zenodo.24409524)

Multiplying by a real number does two separate things to the number line: it changes **size** by the factor |b|, and it keeps or flips **direction** according to the sign of b. From this one picture the note derives, with elementary proofs, the sign rule, how dividing undoes multiplying, why multiplication by zero cannot be undone, and exactly when "multiplying makes a number bigger."

The mathematics is elementary and well known; the contribution is the organization and its use for teaching. Teaching ideas are stated as **untested hypotheses**.

## Contents

| File | What it is |
|---|---|
| `main.tex`, `main.pdf` | The short paper (5 pages, real numbers only) |
| `extended/main.tex`, `extended/main.pdf` | Extended version: complex numbers, matrices (polar decomposition, SVD), and orientation/topology |
| `verify_claims.py` | Checks every numerical claim and example in both papers |
| `references.bib` | Bibliography |

## Status and honesty notes

- Both documents were prepared with substantial AI assistance (Claude, Anthropic). See the acknowledgement in each paper.
- The original idea (separating size from direction) is the author's.
- The extended version's sections on complex numbers, matrices and topology state classical results. They are checked numerically but **have not been reviewed by a professional mathematician**.
- Not peer reviewed.

## Verify and build

```
python3 verify_claims.py        # requires Python 3 and numpy
latexmk -pdf main.tex           # short paper
cd extended && latexmk -pdf main.tex
```

## Cite

See `CITATION.cff`. Licensed under CC BY 4.0 (see `LICENSE`).
