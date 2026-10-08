"""
POD prior from RANS snapshots, projected consistently to an arbitrary point set.

Linear interpolation commutes with linear combination, so the modes at the target
points are U_t = (L X - mean) V / s, in the same coefficient space as the grid modes.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay


def _cond_eq(a, b):
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


class PodBasis:
    """xy: 2-D surface parameterization, surf: +1 upper / -1 lower, X: snapshots [n, m].
    center=False keeps the uncentered basis (used by POD-CS)."""

    def __init__(self, xy, surf, X, center=True):
        self.xy = np.asarray(xy, float)
        self.surf = np.asarray(surf, float)
        self.X = np.asarray(X, float)
        self.center = center
        m = self.X.shape[1]
        self.xbar = self.X.mean(axis=1) if center else np.zeros(len(self.X))
        Xc = self.X - self.xbar[:, None]
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        if center:                       # last mode is null after centering
            U, s, Vt = U[:, :m - 1], s[:m - 1], Vt[:m - 1]
        self.U, self.s, self.V = U, s, Vt.T
        self.lam = s ** 2 / max(m - 1, 1)
        self._tri = {sf: Delaunay(self.xy[self.surf == sf])
                     for sf in np.unique(self.surf)}

    @property
    def K(self):
        return self.U.shape[1]

    def project_snapshots(self, pts, surf_flags):
        """all snapshots at pts (per surface, nearest neighbour outside the hull) -> [n, m]"""
        pts = np.asarray(pts, float)
        surf_flags = np.asarray(surf_flags, float)
        C = np.empty((len(pts), self.X.shape[1]))
        for sf, tri in self._tri.items():
            src = self.surf == sf
            dst = surf_flags == sf
            if not dst.any():
                continue
            P = pts[dst]
            for j in range(self.X.shape[1]):
                lin = LinearNDInterpolator(tri, self.X[src][:, j])
                v = lin(P)
                bad = np.isnan(v)
                if bad.any():
                    near = NearestNDInterpolator(self.xy[src], self.X[src][:, j])
                    v[bad] = near(P[bad])
                C[dst, j] = v
        return C

    def basis_at(self, pts, surf_flags):
        """(xbar_t, U_t, C_t) at the target points"""
        C = self.project_snapshots(pts, surf_flags)
        xbar_t = C.mean(axis=1) if self.center else np.zeros(len(C))
        U_t = (C - xbar_t[:, None]) @ self.V / self.s
        return xbar_t, U_t, C


class PodPrior(PodBasis):
    """M6: joint upper/lower basis from the m6_io long table"""

    def __init__(self, cfd, drop_conds=(), center=True):
        import m6_io
        xyu, Su, cu = m6_io.cfd_snapshots(cfd, 1.0)
        xyl, Sl, cl = m6_io.cfd_snapshots(cfd, -1.0)
        assert cu == cl, 'upper/lower RANS conditions differ'
        conds = list(cu)
        X = np.vstack([Su, Sl])
        keep = [i for i, c in enumerate(conds)
                if not any(_cond_eq(c, d) for d in drop_conds)]
        xy = np.vstack([xyu, xyl])
        surf = np.concatenate([np.ones(len(xyu)), -np.ones(len(xyl))])
        super().__init__(xy, surf, X[:, keep], center=center)
        self.conds = [conds[i] for i in keep]

    def cond_index(self, cond):
        for i, c in enumerate(self.conds):
            if _cond_eq(c, cond):
                return i
        return None
