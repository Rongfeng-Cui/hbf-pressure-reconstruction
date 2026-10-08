"""
Reconstruction methods for one target condition (taps + RANS).

problem dict:
  y     [n]    measured Cp at all taps of the condition
  cfd   [n]    RANS Cp of the condition at the taps
  xbar  [n]    snapshot mean at the taps
  U     [n,K]  centered POD modes at the taps
  lam   [K]    mode variances
  surf  [n]    +1 upper / -1 lower
  U_nc  [n,K'] uncentered modes (POD-CS only)
  psp   [n]    optional PSP Cp at the taps

All methods: fn(problem, S) -> dict(mean, std or None, info), S = observed taps.
"""
import numpy as np

# affine prior: gain deviation sigma_a = 0.1, offset sigma_b = 0.05
A_VAR = 0.01
B_VAR = 0.0025
PSP_TRUST_VAR = 0.25   # sigma_r = 0.5
SIGMA_GRID = (0.005, 0.01, 0.02, 0.04, 0.08)
TAU_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
DEV_GRID = (0.05, 0.2)
# 'mixture': evidence-weighted average over the hyperparameter grid; 'plugin': max evidence
PREDICTIVE = 'mixture'
MIN_WEIGHT = 1e-4


def build_design(problem, mode):
    """
    y - CFD = da*CFD + b (per surface) + POD terms

    mode  'affine'  : affine only
          'shared'  : affine + one POD block for both surfaces
          'surface' : affine + separate upper/lower POD blocks
          'hier'    : affine + shared block + upper/lower deviation blocks  (HBF)
          'bgpod'   : no affine term, shared POD block

    With PSP in the problem a term a_r*(PSP - CFD) is added per surface.
    Returns (D, prior_rel, offset, dev_cols, pod_cols, labels).
    """
    n = len(problem['y'])
    U, lam, surf, cfd = problem['U'], problem['lam'], problem['surf'], problem['cfd']
    cols, pv, dev, pod, lab = [], [], [], [], []
    offset = cfd.copy()
    if mode != 'bgpod':
        for sf in (1.0, -1.0):
            m = surf == sf
            cols += [np.where(m, cfd, 0.0), np.where(m, 1.0, 0.0)]
            pv += [A_VAR, B_VAR]
            dev += [False, False]
            pod += [False, False]
            lab += [f'da_{sf:+.0f}', f'b_{sf:+.0f}']
        psp = problem.get('psp')
        if psp is not None:
            for sf in (1.0, -1.0):
                m = surf == sf
                cols.append(np.where(m, psp - cfd, 0.0))
                pv.append(PSP_TRUST_VAR)
                dev.append(False)
                pod.append(False)
                lab.append(f'ar_{sf:+.0f}')
    if mode in ('shared', 'hier', 'bgpod'):
        cols += list(U.T)
        pv += list(lam)
        dev += [False] * U.shape[1]
        pod += [True] * U.shape[1]
        lab += [f'c{k}' for k in range(U.shape[1])]
    if mode == 'surface':
        for sf in (1.0, -1.0):
            m = (surf == sf).astype(float)
            cols += list((U * m[:, None]).T)
            pv += list(lam)
            dev += [False] * U.shape[1]
            pod += [True] * U.shape[1]
            lab += [f's{sf:+.0f}_{k}' for k in range(U.shape[1])]
    if mode == 'hier':
        for sf in (1.0, -1.0):
            m = (surf == sf).astype(float)
            cols += list((U * m[:, None]).T)
            pv += list(lam)
            dev += [True] * U.shape[1]
            pod += [True] * U.shape[1]
            lab += [f'd{sf:+.0f}_{k}' for k in range(U.shape[1])]
    D = np.column_stack(cols)
    return (D, np.array(pv), offset, np.array(dev, bool),
            np.array(pod, bool), lab)


def _evidence_mu(M, b0, yy, nS, prior_var, sigma):
    """posterior mean and log evidence from the Gram quantities (Cholesky)"""
    from scipy.linalg import solve_triangular
    G = M / sigma ** 2 + np.diag(1.0 / prior_var)
    Lc = np.linalg.cholesky(G)
    b = b0 / sigma ** 2
    z = solve_triangular(Lc, b, lower=True)
    mu = solve_triangular(Lc.T, z, lower=False)
    logdetG = 2 * np.sum(np.log(np.diag(Lc)))
    logdetCov = nS * np.log(sigma ** 2) + logdetG + np.sum(np.log(prior_var))
    quad = yy / sigma ** 2 - b @ mu          # y' Cov^-1 y via Woodbury
    logev = -0.5 * (quad + logdetCov + nS * np.log(2 * np.pi))
    return mu, Lc, logev


def _mixture_predict(components, D, offset):
    """mean and std of the Gaussian mixture, components = [(w, mu, Lc, sigma)]"""
    from scipy.linalg import solve_triangular
    m_mix = np.zeros(D.shape[0])
    s_mix = np.zeros(D.shape[0])
    for w, mu, Lc, sigma in components:
        mean_j = offset + D @ mu
        W = solve_triangular(Lc, D.T, lower=True)
        var_j = np.sum(W ** 2, axis=0) + sigma ** 2
        m_mix += w * mean_j
        s_mix += w * (var_j + mean_j ** 2)
    var = s_mix - m_mix ** 2
    return m_mix, np.sqrt(np.maximum(var, 0.0))


def bayes_fuse(problem, S, mode='hier',
               sigma_grid=SIGMA_GRID, tau_grid=TAU_GRID, dev_grid=DEV_GRID,
               predictive=None, affine_var=None):
    """Closed-form linear-Gaussian fusion. (sigma, tau, rho) are handled with the
    evidence of the observed taps only. affine_var=(a_var, b_var) overrides A_VAR/B_VAR."""
    predictive = predictive or PREDICTIVE
    key = ('_design', mode)
    if key not in problem:
        problem[key] = build_design(problem, mode)
    D, pv_rel, offset, dev_cols, pod, lab = problem[key]
    if affine_var is not None:
        pv_rel = pv_rel.copy()
        pv_rel[[l.startswith('da') for l in lab]] = affine_var[0]
        pv_rel[[l.startswith('b_') for l in lab]] = affine_var[1]
    yS = (problem['y'] - offset)[S]
    DS = D[S]
    M, b0, yy, nS = DS.T @ DS, DS.T @ yS, yS @ yS, len(yS)
    devs = dev_grid if mode == 'hier' else (1.0,)
    fits = []
    for sigma in sigma_grid:
        for tau in tau_grid:
            for dr in devs:
                pv = pv_rel.copy()
                pv[pod] *= tau ** 2
                pv[dev_cols] *= dr
                mu, Lc, ev = _evidence_mu(M, b0, yy, nS, pv, sigma)
                fits.append((ev, sigma, tau, dr, mu, Lc))
    evs = np.array([f[0] for f in fits])
    ev, sigma, tau, dr, mu, Lc = fits[int(np.argmax(evs))]
    if predictive == 'mixture':
        w = np.exp(evs - evs.max())
        keep = w >= MIN_WEIGHT
        w = w[keep] / np.sum(w[keep])
        components = [(float(wj), f[4], f[5], f[1])
                      for wj, f in zip(w, [fits[i] for i in np.where(keep)[0]])]
        mu_bar = sum(wj * c_mu for wj, c_mu, _, _ in components)
        n_eff = float(1.0 / np.sum(w ** 2))
    else:
        components = [(1.0, mu, Lc, sigma)]
        mu_bar = mu
        n_eff = 1.0
    mean, std = _mixture_predict(components, D, offset)
    coef = {l: float(m_) for l, m_ in zip(lab, mu_bar)
            if l.startswith(('da', 'b_', 'ar'))}
    return {'mean': mean, 'std': std,
            'info': {'sigma': sigma, 'tau': tau, 'dev_ratio': dr,
                     'logev': ev, 'coef': coef, 'n_eff': n_eff,
                     'predictive': predictive},
            'model': {'mu': mu, 'Lc': Lc, 'sigma': sigma, 'mode': mode,
                      'components': components}}


def bayes_predict_at(fit, target_problem):
    """predict a fitted bayes_fuse model at another point set (same K / surfaces / psp)"""
    model = fit['model']
    D, pv_rel, offset, dev, pod, lab = build_design(target_problem,
                                                    model['mode'])
    components = model.get('components') or [(1.0, model['mu'], model['Lc'], model['sigma'])]
    return _mixture_predict(components, D, offset)


# ---- baselines
def cfd_only(problem, S):
    return {'mean': problem['cfd'].copy(), 'std': None, 'info': {}}


def affine_calib(problem, S):
    """least-squares a*CFD + b per surface"""
    y, cfd, surf = problem['y'], problem['cfd'], problem['surf']
    mean = cfd.copy()
    for sf in (1.0, -1.0):
        m = surf == sf
        sel = np.intersect1d(S, np.where(m)[0])
        if len(sel) >= 2:
            A = np.column_stack([cfd[sel], np.ones(len(sel))])
            c, *_ = np.linalg.lstsq(A, y[sel], rcond=None)
            mean[m] = c[0] * cfd[m] + c[1]
    return {'mean': mean, 'std': None, 'info': {}}


def _energy_K(lam, thresh=0.99):
    e = np.cumsum(lam) / np.sum(lam)
    return int(np.searchsorted(e, thresh) + 1)


def gpod(problem, S, ridge=False):
    """gappy POD with the 99% energy modes; ridge=True picks the penalty by PRESS"""
    K = min(_energy_K(problem['lam']), len(S))
    U = problem['U'][:, :K]
    A = U[S]
    b = problem['y'][S] - problem['xbar'][S]
    if not ridge:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        info = {'K': K}
    else:
        Ua, sa, Vta = np.linalg.svd(A, full_matrices=False)
        best = None
        for alpha in 10.0 ** np.arange(-8, 1):
            f = sa / (sa ** 2 + alpha)
            coef = Vta.T @ (f * (Ua.T @ b))
            H = (Ua * (sa * f)) @ Ua.T
            r = (b - A @ coef) / np.maximum(1 - np.diag(H), 1e-6)
            press = np.mean(r ** 2)
            if best is None or press < best[0]:
                best = (press, alpha, coef)
        _, alpha, coef = best
        info = {'K': K, 'alpha': alpha}
    mean = problem['xbar'] + U @ coef
    return {'mean': mean, 'std': None, 'info': info}


def _lasso_admm(A, b, alpha_w, rho=1.0, iters=400, tol=1e-9):
    """min 0.5||Ax-b||^2 + sum_j alpha_w[j]|x_j|"""
    from scipy.linalg import solve_triangular
    p = A.shape[1]
    L = np.linalg.cholesky(A.T @ A + rho * np.eye(p))
    Atb = A.T @ b
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    for _ in range(iters):
        q = Atb + rho * (z - u)
        x = solve_triangular(L.T, solve_triangular(L, q, lower=True), lower=False)
        z_old = z
        v = x + u
        z = np.sign(v) * np.maximum(np.abs(v) - alpha_w / rho, 0.0)
        u += x - z
        if max(np.max(np.abs(x - z)), rho * np.max(np.abs(z - z_old))) < tol:
            break
    return z


def _irl1(A, b, alpha, n_irl1=5, eps_w=1e-3):
    """iteratively reweighted L1, threshold n*alpha*w_j (sklearn scaling)"""
    n = len(b)
    w = np.ones(A.shape[1])
    coef = np.zeros(A.shape[1])
    for _ in range(n_irl1):
        new = _lasso_admm(A, b, n * alpha * w)
        if np.max(np.abs(new - coef)) < 1e-6:
            coef = new
            break
        coef = new
        w = 1.0 / (np.abs(coef) + eps_w)
    return coef


def pod_cs_irl1(problem, S, K_grid=(6, 10, 20), alpha_grid=(1e-4, 1e-3, 1e-2),
                n_irl1=5, eps_w=1e-3, cv_folds=3):
    """POD-CS on uncentered modes, (K, alpha) by 3-fold CV on the observed taps"""
    U_full = problem['U_nc']
    Kmax = U_full.shape[1]
    Ks = sorted({min(k, Kmax) for k in K_grid})
    A_all, b_all = U_full[S], problem['y'][S]
    n = len(b_all)
    if n >= 6:
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        folds = np.array_split(perm, cv_folds)
        best = None
        for K in Ks:
            for al in alpha_grid:
                err = 0.0
                for f in folds:
                    tr = np.setdiff1d(perm, f)
                    coef = _irl1(A_all[tr][:, :K], b_all[tr], al, n_irl1, eps_w)
                    r = A_all[f][:, :K] @ coef - b_all[f]
                    err += float(np.sum(r ** 2))
                if best is None or err < best[0]:
                    best = (err, K, al)
        _, K, al = best
    else:
        K, al = min(10, Kmax), 1e-3
    coef = _irl1(A_all[:, :K], b_all, al, n_irl1, eps_w)
    return {'mean': U_full[:, :K] @ coef, 'std': None,
            'info': {'K': K, 'alpha': al}}


def bgpod(problem, S):
    """Bayesian gappy POD: y = CFD + U c, c ~ N(0, tau^2 lam)"""
    return bayes_fuse(problem, S, mode='bgpod')


METHODS = {
    'CFD': cfd_only,
    'Affine': affine_calib,
    'GPOD': lambda p, S: gpod(p, S, ridge=False),
    'RGPOD': lambda p, S: gpod(p, S, ridge=True),
    'POD-CS': pod_cs_irl1,
    'BGPOD': bgpod,
    'HBF': lambda p, S: bayes_fuse(p, S, mode='hier'),
}

ABLATION = {
    'A0-Affine': lambda p, S: bayes_fuse(p, S, mode='affine'),
    'A1-Shared': lambda p, S: bayes_fuse(p, S, mode='shared'),
    'A2-Surface': lambda p, S: bayes_fuse(p, S, mode='surface'),
    'A3-Hier': lambda p, S: bayes_fuse(p, S, mode='hier'),
    'A3-Wide': lambda p, S: bayes_fuse(p, S, mode='hier', affine_var=(0.25, 0.01)),
    'BGPOD': bgpod,
}
