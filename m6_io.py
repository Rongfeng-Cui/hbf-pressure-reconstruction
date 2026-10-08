"""
ONERA M6 data: AR-138 tap pressures + RANS surface-pressure library.

Files expected under M6_DATA (default ./data/m6):
  WT_UP/wt_7.dat, wt_84.dat, wt_88.dat     taps, upper surface, csv  [Ma, alpha, x/c, eta, Cp]
  WT_LOW/wt_7.dat, wt_84.dat, wt_88.dat    taps, lower surface
  WT_UP/cfd_data_up.dat                    RANS, upper surface, whitespace [Ma, alpha, x/c, eta, Cp]
  WT_LOW/cfd_data_low.dat                  RANS, lower surface

Conditions are keyed by (Ma, alpha label) as in the files; the AR-138 angle of
attack of each key is hbf_tools.M6_ALPHA_TRUE.
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('M6_DATA', HERE / 'data' / 'm6'))
CACHE = HERE / 'data' / 'm6_dataset.npz'
CACHE_VERSION = 4

WT_FILES = {'0.7': 'wt_7.dat', '0.84': 'wt_84.dat', '0.88': 'wt_88.dat'}
SURF_DIR = {1.0: 'WT_UP', -1.0: 'WT_LOW'}
CFD_FILE = {1.0: 'cfd_data_up.dat', -1.0: 'cfd_data_low.dat'}

COLS = ['ma', 'alpha', 'x', 'eta', 'cp', 'surf']


def _correct_known_wt_metadata(a, surf, filename):
    # three mistyped x/c values (Ma .70, upper) and one station label (Ma .88 alpha 0, lower);
    # Cp values are not touched
    if surf == 1.0 and filename == WT_FILES['0.7']:
        x_fixes = ((0.20, 0.00366, 0.00866),
                   (0.20, 0.20744, 0.20044),
                   (0.44, 0.01030, 0.00108))
        for station, source_x, corrected_x in x_fixes:
            mistyped_x = (np.isclose(a[:, 0], 0.70) & np.isclose(a[:, 1], 1.25)
                          & np.isclose(a[:, 2], source_x, rtol=0.0, atol=1e-10)
                          & np.isclose(a[:, 3], station))
            if mistyped_x.sum() != 1:
                raise ValueError('unexpected M6 Ma=0.70, alpha=1.25 upper-surface x pattern')
            a[mistyped_x, 2] = corrected_x

    if surf == -1.0 and filename == WT_FILES['0.88']:
        duplicated = (np.isclose(a[:, 0], 0.88) & np.isclose(a[:, 1], 0.0)
                      & np.isclose(a[:, 3], 0.90))
        reference = (np.isclose(a[:, 0], 0.88) & np.isclose(a[:, 1], 0.25)
                     & np.isclose(a[:, 3], 0.95))
        reference_x = a[reference, 2]
        same_fingerprint = np.any(
            np.isclose(a[:, 2, None], reference_x[None, :], rtol=0.0, atol=1e-10),
            axis=1,
        )
        corrected = duplicated & same_fingerprint
        if duplicated.sum() != 28 or reference.sum() != 14 or corrected.sum() != 14:
            raise ValueError('unexpected M6 Ma=0.88, alpha=0 lower-surface station pattern')
        # this 14-tap x family belongs to eta=0.95 (AR-138, all other incidences)
        a[corrected, 3] = 0.95
    return a


def _load_wt_surface(surf):
    rows = []
    d = DATA_ROOT / SURF_DIR[surf]
    for f in WT_FILES.values():
        a = np.loadtxt(d / f, delimiter=',')
        a = _correct_known_wt_metadata(a, surf, f)
        rows.append(np.column_stack([a, np.full(len(a), surf)]))
    return np.vstack(rows)


def _load_cfd_surface(surf):
    a = np.loadtxt(DATA_ROOT / SURF_DIR[surf] / CFD_FILE[surf])
    return np.column_stack([a, np.full(len(a), surf)])


def build_dataset(force=False):
    """{'wt': (N,6), 'cfd': (M,6)} long tables, columns COLS. Cached as npz."""
    if CACHE.exists() and not force:
        with np.load(CACHE) as z:
            current = ('cache_version' in z.files
                       and int(z['cache_version'].item()) == CACHE_VERSION)
            if current:
                return {'wt': z['wt'].copy(), 'cfd': z['cfd'].copy()}
    wt = np.vstack([_load_wt_surface(+1.0), _load_wt_surface(-1.0)])
    cfd = np.vstack([_load_cfd_surface(+1.0), _load_cfd_surface(-1.0)])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, wt=wt, cfd=cfd,
                        cache_version=np.asarray(CACHE_VERSION, dtype=np.int64))
    return {'wt': wt, 'cfd': cfd}


def conditions(tab):
    return sorted({(r[0], r[1]) for r in tab[:, :2]})


def subset(tab, ma=None, alpha=None, surf=None):
    m = np.ones(len(tab), bool)
    if ma is not None:
        m &= np.isclose(tab[:, 0], ma)
    if alpha is not None:
        m &= np.isclose(tab[:, 1], alpha)
    if surf is not None:
        m &= tab[:, 5] == surf
    return tab[m]


def _grid_key(tab):
    xy = np.round(tab[:, 2:4].astype(float), 8)
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    return xy[order]


def tap_coords(wt, surf):
    """unique tap coordinates (x, eta) of one surface + flag: same set at every condition"""
    conds = conditions(subset(wt, surf=surf))
    keys = [_grid_key(subset(wt, ma=c[0], alpha=c[1], surf=surf)) for c in conds]
    n0 = len(keys[0])
    same = all(len(k) == n0 and np.allclose(k, keys[0]) for k in keys)
    return keys[0], same


def cfd_snapshots(cfd, surf):
    """RANS snapshot matrix of one surface -> (xy [n,2], S [n, n_cond], conds)"""
    conds = conditions(subset(cfd, surf=surf))
    cols, xy0 = [], None
    for c in conds:
        t = subset(cfd, ma=c[0], alpha=c[1], surf=surf)
        xy = np.round(t[:, 2:4], 8)
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        t = t[order]
        if xy0 is None:
            xy0 = t[:, 2:4].copy()
        else:
            assert len(t) == len(xy0) and np.allclose(t[:, 2:4], xy0), \
                f'RANS surface grid differs at {c}'
        cols.append(t[:, 4])
    return xy0, np.column_stack(cols), conds


def interp_cfd_to_points(cfd, surf, pts):
    """linear interpolation of every RANS condition to pts [n,2], nearest outside the hull"""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    xy, S, conds = cfd_snapshots(cfd, surf)
    out = np.empty((len(pts), len(conds)))
    for j in range(len(conds)):
        lin = LinearNDInterpolator(xy, S[:, j])
        v = lin(pts)
        bad = np.isnan(v)
        if bad.any():
            near = NearestNDInterpolator(xy, S[:, j])
            v[bad] = near(pts[bad])
        out[:, j] = v
    return out, conds


if __name__ == '__main__':
    ds = build_dataset(force=True)
    wt, cfd = ds['wt'], ds['cfd']
    print('taps rows:', len(wt), ' RANS rows:', len(cfd))
    for surf, name in ((1.0, 'upper'), (-1.0, 'lower')):
        pts, same = tap_coords(wt, surf)
        xy, S, conds = cfd_snapshots(cfd, surf)
        print(f'{name}: {len(pts)} taps (same at all conditions: {same}), '
              f'{len(conditions(subset(wt, surf=surf)))} tap conditions, '
              f'RANS grid {len(xy)} points x {S.shape[1]} conditions')
