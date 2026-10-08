"""
ONERA M6 runs:
  P1  section holdout: taps of the other stations observed, held-out station reconstructed
  P2  N taps chosen by each layout strategy, error on the remaining taps
  P3  model ablation at N = 16
results -> results/{p1_section, p2_taps, p3_ablation}.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import m6_io
import methods
import placement
from pod_prior import PodPrior

OUT = Path(__file__).resolve().parent / 'results'
N_TAPS = [8, 12, 16, 24, 32, 48, 64]
N_RANDOM = 8


def build_problems():
    ds = m6_io.build_dataset()
    wt, cfd = ds['wt'], ds['cfd']
    prior_c = PodPrior(cfd, center=True)
    prior_nc = PodPrior(cfd, center=False)
    problems, cache = {}, {}
    for c in m6_io.conditions(wt):
        w = m6_io.subset(wt, ma=c[0], alpha=c[1])
        pts, sf = w[:, 2:4], w[:, 5]
        key = (pts.round(6).tobytes(), sf.tobytes())
        if key not in cache:
            xbar, U, C = prior_c.basis_at(pts, sf)
            _, Unc, _ = prior_nc.basis_at(pts, sf)
            cache[key] = (xbar, U, C, Unc)
        xbar, U, C, Unc = cache[key]
        j = prior_c.cond_index(c)
        problems[c] = dict(y=w[:, 4], cfd=C[:, j], xbar=xbar, U=U,
                           lam=prior_c.lam, surf=sf, U_nc=Unc, xy=pts,
                           eta=w[:, 3], xc=w[:, 2], ma=c[0], alpha=c[1])
    return problems


def evaluate(res, y, V):
    err = res['mean'][V] - y[V]
    out = {'rmse': float(np.sqrt(np.mean(err ** 2))),
           'mae': float(np.mean(np.abs(err))),
           'cov1': np.nan, 'cov2': np.nan}
    if res.get('std') is not None:
        z = err / np.maximum(res['std'][V], 1e-12)
        out['cov1'] = float(np.mean(np.abs(z) <= 1.0))
        out['cov2'] = float(np.mean(np.abs(z) <= 2.0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problems = build_problems()
    print(f'{len(problems)} conditions')

    rows2, rows3 = [], []
    for c, prob in problems.items():
        t0 = time.perf_counter()
        n = len(prob['y'])
        sets = placement.make_sets(prob, N_TAPS, n_random=N_RANDOM)
        for (strat, N), S in sets.items():
            V = np.setdiff1d(np.arange(n), S)
            for mname, fn in methods.METHODS.items():
                rows2.append({'ma': c[0], 'alpha': c[1], 'N': N, 'strat': strat,
                              'method': mname, **evaluate(fn(prob, S), prob['y'], V)})
            if N == 16:
                for mname, fn in methods.ABLATION.items():
                    rows3.append({'ma': c[0], 'alpha': c[1], 'N': N, 'strat': strat,
                                  'method': mname, **evaluate(fn(prob, S), prob['y'], V)})
        print(f'  P2/P3 {c}: {time.perf_counter() - t0:.1f}s')
    pd.DataFrame(rows2).to_csv(OUT / 'p2_taps.csv', index=False)
    pd.DataFrame(rows3).to_csv(OUT / 'p3_ablation.csv', index=False)

    rows1 = []
    for c, prob in problems.items():
        for e in sorted(set(np.round(prob['eta'], 6))):
            V = np.where(np.abs(prob['eta'] - e) < 1e-6)[0]
            S = np.setdiff1d(np.arange(len(prob['y'])), V)
            for mname, fn in methods.METHODS.items():
                rows1.append({'ma': c[0], 'alpha': c[1], 'eta': e, 'method': mname,
                              **evaluate(fn(prob, S), prob['y'], V)})
    pd.DataFrame(rows1).to_csv(OUT / 'p1_section.csv', index=False)

    df2, df1, df3 = pd.DataFrame(rows2), pd.DataFrame(rows1), pd.DataFrame(rows3)
    pd.set_option('display.width', 200)
    print('\nP2, N=16, I-optimal layouts')
    print(df2[(df2.N == 16) & (df2.strat == 'IOPT')]
          .groupby('method')[['rmse', 'mae', 'cov1', 'cov2']].mean().round(4))
    h = df2[df2.method == 'HBF'].copy()
    h['strat'] = h['strat'].str.replace(r'RAND\d+', 'RAND', regex=True)
    print('\nP2, HBF RMSE, layout x N')
    print(h.groupby(['strat', 'N'])['rmse'].mean().unstack().round(4))
    print('\nP1')
    print(df1.groupby('method')[['rmse', 'mae', 'cov1', 'cov2']].mean().round(4))
    a = df3.copy()
    a['strat'] = a['strat'].str.replace(r'RAND\d+', 'RAND', regex=True)
    print('\nP3, N=16')
    print(a.groupby(['method', 'strat'])['rmse'].mean().unstack().round(4))


if __name__ == '__main__':
    main()
