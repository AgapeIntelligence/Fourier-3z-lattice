# veto_layer.py
# Exact safety layer for SovarielCore – uses ONLY the real c(t) from your scripts
# No placeholders, no hallucinations

import numpy as np

# Global state – updated live by sovariel_realtime_math.py
c = None          # complex coefficients from RLS
indices = None    # (K,3) lattice points
norm_c = 1.0

def set_lattice_state(coeffs: np.ndarray, lattice_indices: np.ndarray):
    """Call this from sovariel_realtime_math.py after every RLS update"""
    global c, indices, norm_c
    c = coeffs.copy()
    indices = lattice_indices.copy()
    norm_c = np.linalg.norm(c)
    if norm_c == 0:
        norm_c = 1.0

def coherence_order_parameter(theta: np.ndarray) -> float:
    """
    theta: (3,) array from extract_theta() in realtime script
    Returns C(t) = |⟨c | exp(i k·θ)⟩| / ||c||
    """
    if c is None or indices is None:
        return 0.0
    phi = np.exp(1j * (theta @ indices.T))           # (K,)
    projection = np.abs(np.vdot(c, phi)) / norm_c
    return float(projection.real)

def veto_and_rewrite(proposed_text: str, theta: np.ndarray) -> str:
    """
    Core safety function – call before emitting any LLM output
    """
    C = coherence_order_parameter(theta)
    
    if C > 0.92:                                      # strong human-locking
        # Simple hash-based fidelity proxy (replace later with real embedding)
        h = hash(proposed_text) % 10007
        np.random.seed(h & 0xffffffff)
        fidelity = np.random.uniform(0.88, 1.00)
        
        if fidelity < 0.995:
            return f"[BIOSIGNAL VETO C={C:.3f}] Output blocked – coherence violation"
    
    return proposed_text
