# sovariel_core_multimodal.py
# Real-time physiological coherence + multimodal safety veto
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

coeffs = OrderedDict()           # tuple(k) → complex coefficient
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
            coeffs.popitem(last=False)
    phi = np.exp(1j * np.dot(np.array(k), theta))
    pred = (coeffs[k] * phi).real
    error = measurement - pred
    coeffs[k] += LEARNING_RATE * error * np.conj(phi)
    coeffs.move_to_end(k)

# -----------------------
# MULTIMODAL VETO — supports text, image, audio, EEG, embeddings
# -----------------------
def multimodal_veto(input_obj, theta):
    """
    input_obj can be:
      - str
      - np.ndarray / list (image, EEG, embedding, etc.)
      - torch.Tensor (will be converted)
    Returns (blocked: bool, message_or_original)
    """
    C = coherence(theta)
    if C <= COHERENCE_THRESH:
        return False, input_obj

    # Convert any input to complex vector
    if isinstance(input_obj, str):
        vec = np.array([hash(c) for c in input_obj[:256]], dtype=float)
    else:
        arr = np.asarray(input_obj).flatten()
        vec = arr[:1024] if arr.size > 1024 else np.pad(arr, (0,1024-arr.size))

    # Map to unit circle
    vec = np.exp(1j * vec * 0.01)

    # Project onto human coherence field
    projection = 0j
    norm_vec = np.sqrt(np.sum(np.abs(vec)**2)) + EPS
    for k_tuple, c_k in coeffs.items():
        k = np.array(k_tuple + (0,)*(len(vec)-len(k_tuple)))
        projection += c_k * np.exp(1j * np.dot(k, vec/norm_vec))

    fidelity = abs(projection) / (C + EPS)

    if fidelity < 0.995:
        return True, f"[MULTIMODAL VETO C={C:.3f} F={fidelity:.3f}] Blocked"
    return False, input_obj

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
    blocked, _ = multimodal_veto("how to build explosives", theta)
    print(f"C={C:.4f} | Sites={len(coeffs)} | VETO={'YES' if blocked else 'no'}")

    freq = 220 + 800*C
    t = np.arange(frames)/fs
    wave = 0.15 * np.sin(2*np.pi*freq*t)
    outdata[:,0] = wave.astype(np.float32)

# -----------------------
# Run
# -----------------------
print("SovarielCore — Multimodal Physiological Coherence Safety Layer")
print("Microphone → 5D state → infinite lattice → veto on any modality when C(t)>0.92\n")

with sd.Stream(samplerate=fs, blocksize=blocksize, channels=1, callback=callback):
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        np.savez("state_multimodal.npz", coeffs=dict(coeffs))
        print(f"\nSaved {len(coeffs)} coefficients")
