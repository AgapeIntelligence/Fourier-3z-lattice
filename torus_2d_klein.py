# torus_2d_klein.py
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def enforce_klein_symmetry(f_grid):
    M1, M2 = f_grid.shape
    rolled = np.roll(np.flip(f_grid, axis=0), shift=M1//2, axis=0)
    return 0.5 * (f_grid + rolled)

n1 = np.array([3, 6, 9])
n2 = np.arange(-12, 13)
indices = np.cartesian_product(n1, n2)
K = indices.shape[0]

M1, M2 = 25, 33
t1 = np.linspace(0, 2*np.pi, M1, endpoint=False)
t2 = np.linspace(0, 2*np.pi, M2, endpoint=False)
T1, T2 = np.meshgrid(t1, t2, indexing='ij')
coords = np.column_stack([T1.ravel(), T2.ravel()])

A = np.exp(1j * coords @ indices.T)

np.random.seed(0)
c_true = np.zeros(K, complex)
active = np.random.choice(K, 6, replace=False)
c_true[active] = np.random.randn(6) + 1j*np.random.randn(6)

field = (A @ c_true).reshape(M1, M2)
field = enforce_klein_symmetry(field.real) + 1j*enforce_klein_symmetry(field.imag)

frac = 0.25
idx = np.random.choice(M1*M2, int(frac*M1*M2), replace=False)
c_rec, _, _, _ = lstsq(A[idx], field.ravel()[idx], lapack_driver="gelsy")
recovered = (A @ c_rec).reshape(M1, M2)
recovered = enforce_klein_symmetry(recovered.real) + 1j*enforce_klein_symmetry(recovered.imag)

print(f"[2D Klein] Error = {np.linalg.norm(recovered-field)/np.linalg.norm(field):.2e}")
