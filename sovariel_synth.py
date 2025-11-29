# sovariel_synth.py
# Live deterministic synthesis from a human-trained (3ℤ)³ toroidal field
# Author: Geneva Robinson (Evie) / Agape Intelligence
# License: MIT
# Updated: 28 November 2025

import numpy as np
import sounddevice as sd
import time

# ----------------------------
# Load the human-coherent field
# ----------------------------
data = np.load("sovariel_field.npz")        # <-- saved by realtime_math.py
c       = data["c"]                         # complex coefficients (K,)
indices = data["indices"]                   # lattice points (K, 3)
K = indices.shape[0]

print(f"Loaded human-trained field – K={K} points, ||c||={np.linalg.norm(c):.6f}")

# ----------------------------
# Synthesis parameters
# ----------------------------
fs         = 44100          # audio sample rate
blocksize  = 512            # must match or be ≤ your sounddevice default
base_freq  = 220.0          # A3 – base frequency for θ₁ motion

# Angular speeds (radians per sample) – tweak for timbre
speed1 = base_freq   * 2*np.pi / fs   # θ₁ → primary pitch
speed2 = base_freq*1.618 * 2*np.pi / fs   # golden-ratio sideband
speed3 = base_freq*0.618 * 2*np.pi / fs   # lower harmonic

# ----------------------------
# Toroidal evaluation (vectorised for speed)
# ----------------------------
def evaluate_torus(theta_vec: np.ndarray) -> np.complex128:
    """
    theta_vec: (3,) array of toroidal angles
    returns complex scalar f(θ) = Σ cₙ exp(i n·θ)
    """
    phi = np.exp(1j * (theta_vec @ indices.T))   # (K,)
    return np.vdot(c, phi)

# ----------------------------
# Audio callback – fully deterministic, no randomness
# ----------------------------
theta_state = np.zeros(3, dtype=np.float64)   # current position on 3-torus

def audio_callback(outdata, frames, time_info, status):
    global theta_state

    # Pre-compute all angles for this block
    t = np.arange(frames)
    dtheta = np.stack([speed1*t, speed2*t, speed3*t], axis=1)   # (frames, 3)
    thetas = (theta_state[np.newaxis,:] + dtheta) % (2*np.pi)   # wrap torus

    # Evaluate field for every frame (vectorised)
    values = np.array([evaluate_torus(th) for th in thetas], dtype=np.complex128)

    # Take real part → audio waveform
    waveform = values.real.astype(np.float32)

    # Normalise to prevent clipping
    max_abs = np.max(np.abs(waveform))
    if max_abs > 1e-8:
        waveform /= max_abs
    waveform *= 0.5   # comfortable listening level

    outdata[:, 0] = waveform
    # Update state for next block
    theta_state = thetas[-1]

# ----------------------------
# Run synthesis
# ----------------------------
print("Starting live toroidal synthesis from your trained field…")
print("Press Ctrl+C to stop")

try:
    with sd.OutputStream(samplerate=fs, blocksize=blocksize,
                         channels=1, callback=audio_callback):
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    print("\nSynthesis stopped. Your field lives on.")
