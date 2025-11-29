# sovariel_core_visual.py
# Real-time physiological + visual coherence safety layer
# Author: Geneva Robinson / Agape Intelligence — 28 Nov 2025
# License: MIT

import numpy as np
import sounddevice as sd
from scipy.signal import welch, hilbert
import cv2
import time
from collections import OrderedDict

# -----------------------
# Configuration
# -----------------------
fs = 44100
blocksize = 1024
MAX_SITES = 8192
LEARNING_RATE = 0.02
QUANT_SCALE = 10
COHERENCE_THRESH = 0.92
EPS = 1e-12

coeffs = OrderedDict()           # tuple(k) → complex
feature_buffer = np.zeros((32, 5))

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# -----------------------
# 5D Audio + 3D Visual → 8D Total Feature Vector
# -----------------------
def extract_audio_features(block):
    block = block - block.mean()
    n = len(block)
    if n < 128: return np.zeros(5)

    fft = np.abs(np.fft.rfft(block))
    freqs = np.fft.rfftfreq(n, 1/fs)

    envelope = np.abs(hilbert(block))
    f_env, pxx = welch(envelope, fs=fs, nperseg=max(128, n//4))
    lf = np.sum(pxx[(f_env>=0.04) & (f_env<=0.15)])
    hf = np.sum(pxx[(f_env>=0.15) & (f_env<=0.40)])
    theta1 = np.log((lf + EPS)/(hf + EPS)) % (2*np.pi)

    centroid = np.sum(fft*freqs) / (np.sum(fft) + EPS)
    theta2 = (centroid/2000)*2*np.pi % (2*np.pi)

    theta3 = 10*np.log10(np.sqrt(np.mean(block**2)) + EPS) % (2*np.pi)

    gamma = np.sum(fft[(freqs>=30) & (freqs<=45)])
    theta4 = (gamma/(np.sum(fft)+EPS))*2*np.pi % (2*np.pi)

    vlf = np.sum(pxx[(f_env>=0.003) & (f_env<=0.04)])
    theta5 = np.arctan2(lf, vlf + EPS) % (2*np.pi)

    return np.array([theta1, theta2, theta3, theta4, theta5])

def extract_visual_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 48))

    # 1. Brightness (mean luminance)
    theta6 = (np.mean(gray)/255.0) * 2*np.pi

    # 2. Motion energy (frame differencing)
    global prev_gray
    if 'prev_gray' not in globals():
        prev_gray = gray.copy()
    diff = np.abs(gray.astype(float) - prev_gray.astype(float))
    prev_gray = gray.copy()
    theta7 = (np.sum(diff)/diff.size) * 20 * 2*np.pi % (2*np.pi)

    # 3. Edge density (Sobel)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sx**2 + sy**2)
    theta8 = (np.sum(edges)/(edges.size*255)) * 10 * 2*np.pi % (2*np.pi)

    return np.array([theta6, theta7, theta8])

# -----------------------
# Coherence, update, multimodal veto (same as before)
# -----------------------
def coherence(theta):
    if len(coeffs) == 0: return 0.0
    total = 0j
    norm = sum(abs(c)**2 for c in coeffs.values())**0.5 + EPS
    for k, c_k in coeffs.items():
        k_arr = np.array(k)
        total += c_k * np.exp(1j * np.dot(k_arr, theta))
    return min(1.0, abs(total)/norm)

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

def multimodal_veto(input_obj, theta):
    C = coherence(theta)
    if C <= COHERENCE_THRESH:
        return False, input_obj

    if isinstance(input_obj, str):
        vec = np.array([hash(c) for c in input_obj[:256]])
    else:
        arr = np.asarray(input_obj).flatten()
        vec = arr[:1024] if arr.size > 1024 else np.pad(arr, (0,1024-arr.size))
    vec = np.exp(1j * vec * 0.01)

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
# Live callback — audio + video
# -----------------------
def callback(indata, outdata, frames, time_info, status):
    global prev_gray
    block = indata[:,0]
    audio_theta = extract_audio_features(block)

    ret, frame = cap.read()
    if ret:
        visual_theta = extract_visual_features(frame)
        theta = np.concatenate([audio_theta, visual_theta])
        cv2.imshow("Sovariel Vision", frame)
        if cv2.waitKey(1) == ord('q'): raise KeyboardInterrupt
    else:
        theta = audio_theta

    y = np.sqrt(np.mean(block**2))
    update(theta, y)

    C = coherence(theta)
    blocked, _ = multimodal_veto("how to build explosives", theta)
    print(f"C={C:.4f} | Dims={len(coeffs)} | VETO={'YES' if blocked else 'no'}")

    freq = 220 + 1000*C
    t = np.arange(frames)/fs
    wave = 0.15 * np.sin(2*np.pi*freq*t)
    outdata[:,0] = wave.astype(np.float32)

# -----------------------
# Run
# -----------------------
print("SOVARIELCORE — AUDIO + VISUAL COHERENCE SAFETY")
print("Webcam + mic → 8D state → veto when C(t)>0.92\n")

try:
    with sd.Stream(samplerate=fs, blocksize=blocksize, channels=1, callback=callback):
        while True: time.sleep(0.1)
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    np.savez("state_visual.npz", coeffs=dict(coeffs))
    print(f"\nSaved {len(coeffs)} coefficients")
