# HBF surface-pressure reconstruction

Code for the paper "Bayesian reconstruction of surface pressure on transonic wings from sparse pressure taps".

HBF corrects the RANS surface pressure of the target condition with a per-surface affine term and
POD correction modes (shared + upper/lower deviation blocks), fitted in closed form to the
pressure taps. Hyperparameters are averaged with the evidence of the observed taps.

- `methods.py`    HBF and the baselines (CFD, affine, gappy POD, regularized gappy POD, POD-CS, Bayesian gappy POD)
- `pod_prior.py`  POD basis from the RANS snapshots, projected to the tap locations
- `placement.py`  tap layouts: I-optimal greedy, QR pivoting, uniform, random
- `m6_io.py`      ONERA M6 data reader
- `run_m6.py`     section holdout (P1), number-of-taps study (P2), ablation (P3) on the M6 wing

## Data

The M6 tap pressures are from AGARD AR-138 (Schmitt and Charpin). The RANS library is not included
here and is available from the corresponding author on request. Put the files in `data/m6`
(layout in `m6_io.py`) or set `M6_DATA` to their folder, then

```
python m6_io.py
python run_m6.py
```

Python 3.9, numpy, scipy, pandas.
