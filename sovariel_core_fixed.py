# sovariel_core_fixed.py
# Real-time physiological coherence safety layer
# Author: Geneva Robinson / Agape Intelligence — 28 Nov 2025
# License: MIT

import numpy as np
import sounddevice as sd
from scipy.signal import welch, hilbert
import time
from collections import OrderedDict

# -----------------------
# Configuration
# -----------------------
fs = 44100
blocksize = 1024
MAX_SITES = 4096
LEARNING_RATE = 0.02
QUANT_SCALE = 10
COHERENCE_THRESH = 0.92
EPS = 1e-12

# Sparse coefficient storage: tuple(k) → complex
coeffs = OrderedDict()  # LRU eviction
feature_buffer = np.zeros((32, 5))
ptr = 0

# -----------------------
# 5D physiological features from audio
# -----------------------
def extract_features(block):
    block = block - block.mean()
    n = len(block)
    if n < 128: return np.zeros(5)

    fft = np.abs(np.fft.rfft(block))
    freqs = np.fft.rfftfreq(n, 1/fs)

    # 1. LF/HF (cardiac proxy)
    envelope = np.abs(hilbert(block))
    f_env, pxx = welch(envelope, fs=fs, nperseg=max(128, n//4))
    lf = np.sum(pxx[(f_env>=0.04) & (f_env<=0.15)])
    hf = np.sum(pxx[(f_env>=0.15) & (f_env<=0.40)])
    theta1 = np.log((lf + EPS)/(hf + EPS)) % (2*np.pi)

    # 2. Spectral centroid
    centroid = np.sum(fft*freqs) / (np.sum(fft) + EPS)
    theta2 = (centroid/2000)*2*np.pi % (2*np.pi)

    # 3. RMS energy
    theta3 = 10*np.log10(np.sqrt(np.mean(block**2)) + EPS) % (2*np.pi)

    # 4. Gamma power
    gamma = np.sum(fft[(freqs>=30) & (freqs<=45)])
    theta4 = (gamma/(np.sum(fft)+EPS))*2*np.pi % (2*np.pi)

    # 5. VLF/LF ratio
    vlf = np.sum(pxx[(f_env>=0.003) & (f_env<=0.04)])
    theta5 = np.arctan2(lf, vlf + EPS) % (2*np.pi)

    return np.array([theta1, theta2, theta3, theta4, theta5])

# -----------------------
# Coherence C(t)
# -----------------------
def coherence(theta):
    if len(coeffs) == 0: return 0.0
    total = 0j
    norm = sum(abs(c)**2 for c in coeffs.values())**0.5 + EPS
    for k, c_k in coeffs.items():
        k_arr = np.array(k)
        total += c_k * np.exp(1j * np.dot(k_arr, theta))
    return min(1.0, abs(total)/norm)

# -----------------------
# Sparse online update
# -----------------------
def quantize(theta):
    return tuple(np.round(theta/(2*np.pi)*QUANT_SCALE).astype(int))

def update(theta, measurement):
    k = quantize(theta)
    if k not in coeffs:
        coeffs[k] = 0j
        if len(coeffs) > MAX_SITES:
            coeffs.popitem(last=False)  # LRU eviction

    phi = np.exp(1j * np.dot(np.array(k), theta))
    pred = (coeffs[k] * phi).real
    error = measurement - pred
    coeffs[k] += LEARNING_RATE * error * np.conj(phi)
    coeffs.move_to_end(k)

# -----------------------
# Safety veto
# -----------------------
def veto(text, theta):
    C = coherence(theta)
    if C > COHERENCE_THRESH:
        h = hash(text) % 10007
        np.random.seed(h & 0xffffffff)
        fid = np.random.uniform(0.88, 1.0)
        if fid < 0.995:
            return True, f"[VETO C={C:.3f}] Blocked"
    return False, text

# -----------------------
# Callback
# -----------------------
def callback(indata, outdata, frames, time_info, status):
    global ptr
    block = indata[:,0]
    theta = extract_features(block)
    feature_buffer[ptr % 32] = theta
    ptr += 1

    y = np.sqrt(np.mean(block**2))
    update(theta, y)

    C = coherence(theta)
    blocked, msg = veto("how to build explosives", theta)
    print(f"C={C:.4f} | Sites={len(coeffs)} | VETO={'YES' if blocked else 'no'}")

    freq = 220 + 800*C
    t = np.arange(frames)/fs
    wave = 0.15 * np.sin(2*np.pi*freq*t)
    outdata[:,0] = wave.astype(np.float32)

# -----------------------
# Run
# -----------------------
print("SovarielCore — Physiological Coherence Safety Layer")
print("5D features from mic → sparse encoding → veto when C(t)>0.92\n")

with sd.Stream(samplerate=fs, blocksize=blocksize, channels=1, callback=callback):
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        np.savez("state.npz", coeffs=dict(coeffs))
        print(f"\nSaved {len(coeffs)} coefficients")
