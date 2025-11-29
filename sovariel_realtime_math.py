# sovariel_realtime_math.py
# Real-time adaptive toroidal waveform sampling on (3ℤ)³ lattice
# Author: Geneva Robinson (Evie) / Agape Intelligence
# License: MIT
# Updated: 28 November 2025 – now syncs live c(t) to veto_layer

import numpy as np
import sounddevice as sd
import time

# Your lattice builder – must be in the same directory or on PYTHONPATH
from lattice_3z_nD import build_3z_lattice

# NEW: Sync live coefficients with the veto layer
from veto_layer import set_lattice_state

# ===============================
# CONFIGURATION
# ===============================
d = 3
lattice_power = 3           # gives 27 points on (3ℤ)³
sample_rate = 44100
blocksize = 512
forgetting = 0.998          # RLS forgetting factor
measurement_function = lambda x: np.sqrt(np.mean(x**2))  # RMS as scalar

# ===============================
# LATTICE SETUP
# ===============================
indices = build_3z_lattice(lattice_power, d=d)  # shape (K,3)
K = indices.shape[0]

# RLS state
c = np.zeros(K, dtype=np.complex128)
P = np.eye(K) * 1e4  # initial inverse covariance

# ===============================
# TOROIDAL FEATURE MAPPING
# ===============================
def extract_theta(audio_block):
    fft = np.abs(np.fft.rfft(audio_block))
    freqs = np.fft.rfftfreq(len(audio_block), d=1/sample_rate)
    
    low  = np.sum(fft[5:20])  + 1e-8
    high = np.sum(fft[40:100]) + 1e-8
    theta1 = 4.0 * np.log(low/high) % (2*np.pi)
    
    centroid = np.sum(fft * freqs) / (np.sum(fft) + 1e-8)
    theta2 = (centroid / 2000.0) * 2*np.pi % (2*np.pi)
    
    theta3 = 10.0 * np.log10(np.sqrt(np.mean(audio_block**2)) + 1e-8) * 2*np.pi % (2*np.pi)
    
    return np.array([theta1, theta2, theta3])

# ===============================
# AUDIO CALLBACK
# ===============================
def audio_callback(indata, frames, time_info, status):
    global c, P
    audio = indata[:,0]
    
    theta = extract_theta(audio)
    y = np.array([measurement_function(audio)])
    phi = np.exp(1j * theta @ indices.T).ravel()
    
    # RLS update
    denom = 1 + forgetting * np.real(phi.conj() @ P @ phi)
    k = (P @ phi.conj()) / denom
    c = c + k * (y - np.real(phi.conj() @ c))
    P = (P - np.outer(k, phi.conj() @ P)) / forgetting
    
    # NEW: Keep the veto layer perfectly in sync with the human-driven field
    set_lattice_state(c.copy(), indices)

# ===============================
# START STREAM
# ===============================
print("SovarielCore real-time lattice → starting (3ℤ)³ adaptive tracking + veto sync…")

stream = sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=sample_rate,
                        blocksize=blocksize)

stream.start()

try:
    start_time = time.time()
    while True:
        time.sleep(1)
        elapsed = int(time.time() - start_time)
        print(f"t={elapsed}s | ||c||={np.linalg.norm(c):.4f} | C(t) live → veto_layer")
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    stream.stop()
    stream.close()
    np.savez("sovariel_field.npz", c=c, indices=indices)
    print("Final human-coherent field saved → sovariel_field.npz")
