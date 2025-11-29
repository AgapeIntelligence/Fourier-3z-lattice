# torus_3d_3zlattice.py
import numpy as np
from scipy.linalg import lstsq
from lattice_3z_nD import build_3z_lattice

indices = build_3z_lattice(3)      # (343, 3)
K = indices.shape[0]

N = 25
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
grids = np.meshgrid(theta, theta, theta, indexing='ij')
coords = np.column_stack([g.ravel() for g in grids])

A = np.exp(1j * coords @ indices.T)

np.random.seed(42)
c_true = np.zeros(K, complex)
active = np.random.choice(K, 15, replace=False)
c_true[active] = np.random.randn(15) + 1j*np.random.randn(15)

field = A @ c_true

frac = 0.08
idx = np.random.choice(A.shape[0], int(frac*A.shape[0]), replace=False)
c_rec, _, _, _ = lstsq(A[idx], field[idx], lapack_driver="gelsy")
recovered = A @ c_rec

error = np.linalg.norm(recovered - field) / np.linalg.norm(field)
print(f"[3D Torus] Measurements {100*frac:.1f}% → Relative error = {error:.2e}")
