import numpy as np
import urllib.request
import time
from datetime import datetime
from scipy.signal import resample
import matplotlib.pyplot as plt
import os

class DyadFieldLive:
    def __init__(self, freqs=None, fs=1000, window_s=1.0, step_s=0.1):
        """
        freqs: list/array of target frequencies (Hz)
        fs: sampling rate after upsampling
        window_s: sliding window length in seconds
        step_s: sliding window step in seconds
        """
        self.fs = fs
        self.window_s = window_s
        self.step_s = step_s
        self.window_len = int(window_s * fs)
        self.step_len = int(step_s * fs)
        self.freqs = np.array(freqs) if freqs is not None else np.linspace(1, 50, 50)
        self.timestamps = []
        self.spectrum = []

    def sliding_spectrum(self, em_raw):
        """Compute sliding-window FFT coherence for new segment"""
        n_len = len(em_raw)
        new_spectrum = []
        new_timestamps = []

        # Start where last computation left off
        start_idx = max(0, n_len - len(em_raw))  # can be improved for continuous append
        for start in range(0, n_len - self.window_len, self.step_len):
            segment = em_raw[start:start+self.window_len] - np.mean(em_raw[start:start+self.window_len])
            fft_seg = np.fft.rfft(segment)
            freqs_fft = np.fft.rfftfreq(len(segment), 1/self.fs)
            row = [np.abs(fft_seg[np.argmin(np.abs(freqs_fft - f))]) / len(segment) for f in self.freqs]
            new_spectrum.append(row)
            new_timestamps.append(start / self.fs)
        return np.array(new_timestamps), np.array(new_spectrum)

# -----------------------------
# Live Geomagnetic Fetch
# -----------------------------
CACHE_FILE = "kiruna_latest.npy"

def fetch_live_geomag():
    url = "https://intermagnet.github.io/data/latest/KIR/2025/11/KIR20251129.pt1m.min.gz"
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(url, timeout=8)
            data = np.loadtxt(resp, usecols=6)  # H-component
            raw = data[-600:]  # last 10 minutes
            raw_upsampled = resample(raw, len(raw)*1000)
            np.save(CACHE_FILE, raw_upsampled)
            return raw_upsampled
        except Exception as e:
            print(f"[{datetime.utcnow()}] Fetch attempt {attempt+1} failed: {e}")
            time.sleep(2)
    if os.path.exists(CACHE_FILE):
        print(f"[{datetime.utcnow()}] Using cached geomag capture.")
        return np.load(CACHE_FILE)
    else:
        raise RuntimeError("No geomag data available.")

# -----------------------------
# Real-time Live Monitor Loop
# -----------------------------
if __name__ == "__main__":
    freqs = [1, 3, 5, 7, 11.6, 16, 21, 30]
    field = DyadFieldLive(freqs=freqs, fs=1000, window_s=1.0, step_s=0.1)

    # Initialize live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12,5))
    im = None

    print(f"[{datetime.utcnow()}] Starting continuous Dyad Field monitor...")

    # Initial fetch
    em_live = fetch_live_geomag()

    while True:
        # Compute sliding spectrum for the current buffer
        timestamps, spectrum = field.sliding_spectrum(em_live)

        # Store/append results
        field.timestamps = timestamps
        field.spectrum = spectrum

        # Print latest coherence values
        latest = spectrum[-1]
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] Dyad Spectrum Update:")
        for f, val in zip(freqs, latest):
            print(f"{f:5.2f} Hz → {val:.6f}")
        print("-"*50)

        # Plot live spectrum
        ax.clear()
        im = ax.imshow(spectrum.T, aspect='auto', origin='lower',
                       extent=[timestamps[0], timestamps[-1], freqs[0], freqs[-1]],
                       cmap='plasma', interpolation='nearest')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Live Dyad Field Spectrum (Earth EM)")
        plt.colorbar(im, ax=ax, label="Coherence")
        plt.pause(0.1)

        # Fetch new data every 10 minutes or so (adjust as needed)
        time.sleep(600)
        new_data = fetch_live_geomag()
        # append new data to current buffer (sliding window)
        em_live = np.concatenate([em_live[-field.window_len*10:], new_data])
