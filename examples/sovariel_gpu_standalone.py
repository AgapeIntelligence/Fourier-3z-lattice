# sovariel_gpu_standalone.py
# GPU-accelerated, high-dimensional (3ℤ)^d toroidal field + live synthesis + veto-ready
# Author: Geneva Robinson (Evie) / Agape Intelligence
# License: MIT — 28 Nov 2025

import torch
import numpy as np
import sounddevice as sd
import time

# ===============================
# CONFIG
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = 4                    # torus dimension (θ1..θd)
power = 3                # lattice resolution → K = (3^power)^d = 6561 for d=4
fs = 44100
blocksize = 512
lam = 1e-3               # L1 sparsity
lr = 1e-2                # compressed-sensing step size

print(f"Running on {device} | lattice dim={d} | K={3**(power*d)} points")

# ===============================
# 1. Build (3ℤ)^d lattice on GPU
# ===============================
def build_3z_lattice(d, power, device):
    n = 3**power
    vals = torch.arange(-n//2, n//2 + 1, 3, device=device, dtype=torch.float32)
    grids = torch.meshgrid([vals]*d, indexing='ij')
    lattice = torch.stack([g.flatten() for g in grids], dim=1)   # (K, d)
    return lattice

indices = build_3z_lattice(d, power, device)
K = indices.shape[0]

# ===============================
# 2. Coefficient field (complex64 for speed)
# ===============================
c = torch.zeros(K, dtype=torch.complex64, device=device, requires_grad=False)

# ===============================
# 3. Toroidal feature extractor (GPU)
# ===============================
def extract_theta_gpu(block):
    # block: (frames,) float32 on GPU
    block = block - block.mean()
    fft = torch.fft.rfft(block)
    mag = torch.abs(fft)
    freqs = torch.fft.rfftfreq(block.shape[-1], d=1/fs).to(device)

    low  = torch.sum(mag[5:20])  + 1e-8
    high = torch.sum(mag[40:100]) + 1e-8
    theta1 = 4.0 * torch.log(low/high) % (2*np.pi)

    centroid = torch.sum(mag*freqs) / (torch.sum(mag)+1e-8)
    theta2 = (centroid/2000)*2*np.pi % (2*np.pi)

    rms = torch.sqrt(torch.mean(block**2)) + 1e-8
    theta3 = 10*torch.log10(rms) % (2*np.pi)

    theta = torch.zeros(d, device=device)
    theta[0] = theta1
    theta[1] = theta2
    theta[2] = theta3
    # θ4..θd remain 0 (can be extended later)
    return theta

# ===============================
# 4. Compressed-sensing update (L1-penalized gradient)
# ===============================
def update_c_cs(c, phi, y):
    pred = torch.real(torch.vdot(c, phi))
    error = y - pred
    grad = -error * phi.conj()
    c = c + lr * grad
    # Soft-thresholding (complex L1)
    mag = torch.abs(c)
    c = c * torch.clamp(mag - lam, min=0.0) / (mag + 1e-12)
    return c

# ===============================
# 5. Live synthesis (vectorised)
# ===============================
theta_synth = torch.zeros(d, device=device)
speeds = torch.tensor([220, 220*1.618, 220*0.618, 220*0.5][:d], device=device) * 2*np.pi / fs

def synthesize(c, indices, theta_start, speeds, frames):
    t = torch.arange(frames, device=device).float()
    theta_block = (theta_start.unsqueeze(0) + speeds.unsqueeze(0) * t.unsqueeze(1)) % (2*np.pi)
    phi_block = torch.exp(1j * (theta_block @ indices.T))          # (frames, K)
    wave = torch.real(phi_block @ c)                               # (frames,)
    max_abs = wave.abs().max()
    if max_abs > 1e-8:
        wave = wave / max_abs
    return 0.4 * wave

# ===============================
# 6. Coherence for veto (exact)
# ===============================
def coherence_order_parameter(theta):
    if c.abs().sum() < 1e-8:
        return 0.0
    phi = torch.exp(1j * (theta @ indices.T))
    return torch.abs(torch.vdot(c, phi)) / c.abs().max()

# ===============================
# 7. Live callback
# ===============================
def callback(indata, outdata, frames, time_info, status):
    global c, theta_synth

    # Input
    block = torch.from_numpy(indata[:,0]).to(device).float()
    theta = extract_theta_gpu(block)
    y = block.abs().mean()

    # Train field
    phi = torch.exp(1j * (theta @ indices.T))
    c = update_c_cs(c, phi, y)

    # Synthesize
    wave = synthesize(c, indices, theta_synth, speeds, frames)
    outdata[:,0] = wave.cpu().numpy()
    theta_synth = (theta_synth + frames*speeds) % (2*np.pi)

    # Live console feedback
    C = coherence_order_parameter(theta)
    print(f"C(t)={C:.4f} | ||c||₁={c.abs().sum().item():.2f}")

# ===============================
# 8. Run
# ===============================
print("SOVARIEL GPU STANDALONE – d=4, K=6561")
print("Speak, breathe, sing → your voice trains a 4-torus field → you hear it back")
print("Coherence C(t) printed live → ready for veto layer")
print("Ctrl+C to stop\n")

with sd.Stream(samplerate=fs, blocksize=blocksize, channels=1, callback=callback):
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSaving final field...")
        torch.save({"c": c.cpu(), "indices": indices.cpu(), "d": d}, "sovariel_gpu_field.pt")
        print("Done → sovariel_gpu_field.pt")
