# phase_sync_controller.py
# N-dimensional phase synchronization system (Kuramoto-style)
# Works for 9D, 11D, or any N.

import numpy as np

class PhaseSyncController:
    def __init__(self, ref_phase, noise=0.1):
        """
        ref_phase: reference vector, shape (N,)
        """
        self.ref = ref_phase % (2*np.pi)
        self.N = len(ref_phase)
        self.phases = self.ref + np.random.normal(0, noise, self.N)

    def order_parameter(self):
        return np.abs(np.mean(np.exp(1j * self.phases)))

    @staticmethod
    def wrap(x):
        return (x + np.pi) % (2*np.pi) - np.pi

    def crush(self, gain=12.0):
        error = self.wrap(self.phases - self.ref)
        R = self.order_parameter()
        damping = np.exp(gain * (1 - R))
        self.phases = (self.ref + damping * error) % (2*np.pi)

    def apply_drift(self, sigma=0.1):
        self.phases = (self.phases + np.random.normal(0, sigma, self.N)) % (2*np.pi)
