# sovariel_reconstruct.py
# Reconstruct continuous toroidal field from live (3ℤ)³ lattice coefficients
# Author: Geneva Robinson (Evie) / Agape Intelligence
# License: MIT
# Updated: 28 November 2025 – fully compatible with current realtime script

import numpy as np

# ----------------------------
# Load the latest human-coherent field
# ----------------------------
data = np.load("sovariel_field.npz")
c = data["c"]             # complex coefficients (K,)
indices = data["indices"] # lattice points (K, 3)
K = indices.shape[0]

print(f"Loaded field from human session")
print(f"Lattice: (3ℤ)³, power={int(np.log(K)/np.log(27)):d}, K={K} points")
print(f"Field norm: {np.linalg.norm(c):.6f}")

# ----------------------------
# Evaluation grid on the 3-torus
# ----------------------------
N_eval = 64  # resolution per dimension → 64³ = 262,144 points
theta1 = np.linspace(0, 2*np.pi, N_eval, endpoint=False)
theta2 = np.linspace(0, 2*np.pi, N_eval, endpoint=False)
theta3 = np.linspace(0, 2*np.pi, N_eval, endpoint=False)

Theta1, Theta2, Theta3 = np.meshgrid(theta1, theta2, theta3, indexing='ij')
coords = np.stack([Theta1.ravel(), Theta2.ravel(), Theta3.ravel()], axis=1)  # (N³, 3)

# ----------------------------
# Reconstruct full field f(θ₁,θ₂,θ₃) = Σ cₙ exp(i n·θ)
# ----------------------------
print("Reconstructing field on 64³ toroidal grid…")
exp_terms = np.exp(1j * (coords @ indices.T))   # (N³, K)
F = exp_terms @ c                               # (N³,) complex field
F_grid = F.reshape((N_eval, N_eval, N_eval))

F_real = F_grid.real
F_imag = F_grid.imag
F_abs  = np.abs(F_grid)

# ----------------------------
# Save everything
# ----------------------------
np.savez("sovariel_field_reconstructed.npz",
         F_real=F_real,
         F_imag=F_imag,
         F_abs=F_abs,
         theta1=theta1, theta2=theta2, theta3=theta3,
         c_norm=np.linalg.norm(c))

print("Reconstruction complete!")
print(f"Grid shape: {F_grid.shape}")
print(f"Max magnitude: {np.max(F_abs):.6f}")
print(f"Mean magnitude: {np.mean(F_abs):.6f}")
print("→ sovariel_field_reconstructed.npz ready for visualization / audio resynthesis")
