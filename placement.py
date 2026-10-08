"""
Tap layouts. Only the design matrix / coordinates are used, never the measured Cp.

fn(problem, N) -> N tap indices. Greedy and QR sequences are nested.
"""
import numpy as np
from scipy.linalg import qr

import methods

# design-stage affine prior (sigma_a=0.5, sigma_b=0.1)
DESIGN_A_VAR = 0.25
DESIGN_B_VAR = 0.01


def random_set(problem, N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(len(problem['y']), N, replace=False)


def uniform_set(problem, N):
    """quota per (surface, station) proportional to its tap count, equal chordwise spacing"""
    surf, eta, x = problem['surf'], problem['eta'], problem['xc']
    groups = []
    for sf in (1.0, -1.0):
        for e in sorted(set(np.round(eta[surf == sf], 6))):
            idx = np.where((surf == sf) & (np.abs(eta - e) < 1e-6))[0]
            groups.append(idx[np.argsort(x[idx])])
    quota = np.array([len(g) for g in groups], float)
    quota = quota / quota.sum() * N
    take = np.floor(quota).astype(int)
    rem = N - take.sum()
    order = np.argsort(-(quota - take))
    take[order[:rem]] += 1
    sel = []
    for g, t in zip(groups, take):
        if t <= 0:
            continue
        pos = np.linspace(0, len(g) - 1, t).round().astype(int)
        sel += list(g[np.unique(pos)])
    sel = np.array(sel)
    if len(sel) > N:
        sel = sel[:N]
    return sel


def qr_rank(problem):
    """QR with column pivoting on U^T (Manohar et al.), full permutation"""
    _, _, piv = qr(problem['U'].T, pivoting=True)
    return piv


def ioptimal_rank(problem, sigma=0.02, tau=1.0, dev_ratio=0.1, n_max=None):
    """greedy I-optimal: each step takes the tap with the largest reduction of the mean
    posterior predictive variance over all candidate taps (HBF design matrix, rank-1 updates)"""
    D, pv_rel, _, dev_cols, pod, lab = methods.build_design(problem, 'hier')
    pv = pv_rel.copy()
    pv[pod] *= tau ** 2
    pv[dev_cols] *= dev_ratio
    pv[[l.startswith('da') for l in lab]] = DESIGN_A_VAR
    pv[[l.startswith('b_') for l in lab]] = DESIGN_B_VAR
    Sig = np.diag(pv)
    n = len(D)
    n_max = n_max or n
    chosen, avail = [], list(range(n))
    for _ in range(min(n_max, n)):
        SD = Sig @ D.T
        TD = D @ SD
        diag = np.einsum('ij,ji->i', D, SD)
        denom = sigma ** 2 + diag
        red = np.mean(TD ** 2, axis=0) / denom
        red[chosen] = -np.inf
        best = int(np.argmax(red))
        d = D[best]
        Sd = Sig @ d
        Sig = Sig - np.outer(Sd, Sd) / (sigma ** 2 + d @ Sd)
        chosen.append(best)
    return np.array(chosen)


def make_sets(problem, n_taps, n_random=8, seed0=100):
    """layouts of every strategy for each N in n_taps"""
    qr_seq = qr_rank(problem)
    io_seq = ioptimal_rank(problem, n_max=max(n_taps))
    sets = {}
    for N in n_taps:
        sets[('QR', N)] = qr_seq[:N]
        sets[('IOPT', N)] = io_seq[:N]
        sets[('UNIF', N)] = uniform_set(problem, N)
        for r in range(n_random):
            sets[(f'RAND{r}', N)] = random_set(problem, N, seed=seed0 + r)
    return sets
