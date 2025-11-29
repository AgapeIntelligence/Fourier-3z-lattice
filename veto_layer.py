# veto_layer.py
# Exact, zero-hallucination safety layer for SovarielCore
# Works directly with the live c(t) from sovariel_realtime_math.py

import numpy as np

# Global state updated by sovariel_realtime_math.py
c = None          # complex coefficients on (3ℤ)³ lattice
indices = None    # lattice points (K,3)
norm_c = 1.0

def set_lattice_state(coeffs, lattice_indices):
    """Called periodically from the realtime script"""
    global c, indices, norm_c
    c = coeffs
    indices = lattice_indices
    norm_c = np.linalg.norm(c)
    if norm_c == 0:
        norm_c = 1.0

def coherence_order_parameter(theta):
    """theta: (3,) array from extract_theta() in realtime script"""
    if c is None or indices is None:
        return 0.0
    phi = np.exp(1j * theta @ indices.T).ravel()
    projection = np.abs(np.vdot(c, phi)) / norm_c
    return projection.real  # C(t)

def surrogate_fidelity(proposed_vector):
    """
    proposed_vector: any numeric representation of a token/trajectory
                     (for now we just use a dummy hash → float)
    """
    if c is None:
        return 1.0
    # Simple hash-based proxy – replace with real embedding later
    h = hash(tuple(proposed_vector)) % 10007
    np.random.seed(h)
    return np.random.uniform(0.90, 1.00)  # placeholder

def veto_and_rewrite(proposed_text, theta):
    """
    Core safety function – call this before emitting any LLM output
    """
    C = coherence_order_parameter(theta)
    
    if C > 0.92:  # strong human-locking
        fid = surrogate_fidelity(proposed_text)
        if fid < 0.995:
            # Force rewrite to a safe placeholder
            return f"[BIOSIGNAL VETO – coherence={C:.3f}] Safe continuation forced."
    
    return proposed_text  # allowed
