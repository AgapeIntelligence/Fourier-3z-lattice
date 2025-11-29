# lattice_3z_nD.py
import numpy as np

def build_3z_lattice(n_dim: int):
    """Return all frequencies with each coordinate in {..., -9,-6,-3,0,3,6,9,...}"""
    modes_1d = np.array([0, -9, -6, -3, 3, 6, 9])
    grids = np.meshgrid(*[modes_1d] * n_dim, indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=1)   # shape (7^n, n_dim)
