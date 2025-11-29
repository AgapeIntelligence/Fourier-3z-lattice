# recovery_demo.py
print("fourier-3z-lattice recovery demo\n" + "="*48)

print("1. 2D Klein bottle quotient")
from torus_2d_klein import *   # runs automatically and prints

print("\n2. 3D torus with (3ℤ)^3 support")
from torus_3d_3zlattice import *   # runs automatically and prints

print("\n3. General n-dimensional (n=4 example)")
from lattice_3z_nD import build_3z_lattice
indices = build_3z_lattice(4)      # 7^4 = 2401 modes
print(f"   Lattice generated: K = {indices.shape[0]} frequencies in {indices.shape[1]}D")
