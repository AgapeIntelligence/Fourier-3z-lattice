#!/usr/bin/env python3
"""
DYAD FIELD MONITOR v4.0 — Production Reference Implementation
Real-time geomagnetic field monitoring at 1 kHz effective sampling
- Local 3-axis magnetometer (USB/serial) with timestamp-based resampling
- Automatic fallback to INTERMAGNET Kiruna observatory (1-minute data, upsampled)
- Adaptive narrowband filtering centered on 11.6 Hz using SOS filters
- Sliding-window FFT magnitude estimation at Schumann-related frequencies
- Dual visualization: local Matplotlib + remote Plotly Dash server
- Real-time sonification (optional, headless-safe)
- CSV logging of spectral power at target frequencies

Requirements:
    pip install numpy scipy matplotlib pandas plotly dash pygame pyserial
"""

import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Remove this line for real audio output

import numpy as np
import time
import threading
import queue
from datetime import datetime, date, timedelta

import serial
import serial.tools.list_ports
import urllib.request
import gzip

from scipy.signal import butter, sosfilt, resample
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import pygame
import pandas as pd

from dash import Dash, html, dcc
import plotly.graph_objs as go

# ================================================================
# Configuration
# ================================================================
TARGET_FS = 1000                # Hz — final analysis sampling rate
BUFFER_SECONDS = 180            # Length of analysis buffer
WINDOW_S = 1.0                  # FFT window length (seconds)
STEP_S = 0.1                    # Window step (seconds)

FREQS = np.array([1.0, 3.0, 5.0, 7.0, 11.6, 16.0, 21.0, 30.0])  # Target frequencies (Hz)
INTERMAGNET_BASE = "https://intermagnet.github.io/data/latest/KIR"
CACHE_FILE = "kiruna_latest.npy"
LOG_DIR = "dyad_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ================================================================
# 11.6 Hz Adaptive Bandpass (SOS, single-pass for low latency)
# ================================================================
def bandpass_116_sos(signal, lowcut=10.4, highcut=12.8, order=4, fs=TARGET_FS):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal)

# ================================================================
# Sonification (optional diagnostic tone)
# ================================================================
class ToneGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        self.base_freq = 220.0

    def play(self, amplitude, duration=0.1):
        if amplitude < 1e-7:
            return
        freq_shift = np.clip(20 * (amplitude / 1e-4), -80, 80)
        freq = np.clip(self.base_freq + freq_shift, 100, 600)
        volume = min(1.0, amplitude / 1e-5) * 0.5

        t = np.linspace(0, duration, int(22050 * duration), False)
        tone = np.sin(2 * np.pi * freq * t) * volume
        sound = pygame.sndarray.make_sound((tone * 32767).astype(np.int16))
        sound.play()

# ================================================================
# Local Sensor Interface
# ================================================================
class Magnetometer:
    def __init__(self):
        self.port = self._detect_port()
        self.ser = None
        self.queue = queue.Queue(maxsize=50000)
        self.running = False

        if self.port:
            try:
                self.ser = serial.Serial(self.port, 115200, timeout=1)
                time.sleep(2.0)
                self.ser.reset_input_buffer()
                self._reader_thread = threading.Thread(target=self._reader, daemon=True)
                self._reader_thread.start()
                self.running = True
                print(f"[Hardware] Connected to {self.port}")
            except Exception as e:
                print(f"[Hardware] Failed: {e}")
                self.ser = None
        else:
            print("[Hardware] No local sensor detected — using INTERMAGNET")

    def _detect_port(self):
        for p in serial.tools.list_ports.comports():
            desc = f"{p.device} {p.description} {p.hwid}".upper()
            if any(k in desc for k in ["USB", "FTDI", "CH340", "CP210", "ARDUINO", "STM"]):
                return p.device
        return None

    def _reader(self):
        while self.running and self.ser:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                values = [float(x) for x in line.replace(",", " ").split() if x.replace("-", "").replace(".", "").replace("E", "").isdigit() or (x.count(".") == 1 and x.lstrip("-").replace(".", "").isdigit())]
                if len(values) >= 3:
                    magnitude = np.linalg.norm(values[:3])
                    ts = time.time()
                    try:
                        self.queue.put_nowait((ts, magnitude))
                    except queue.Full:
                        try: self.queue.get_nowait()
                        except: pass
                        self.queue.put_nowait((ts, magnitude))
            except:
                time.sleep(0.01)

    def get_buffer(self):
        data = []
        cutoff = time.time() - BUFFER_SECONDS
        while True:
            try:
                ts, val = self.queue.get_nowait()
                if ts >= cutoff:
                    data.append((ts, val))
            except queue.Empty:
                break
        if len(data) < 10:
            return np.array([])
        ts_arr, vals = map(np.array, zip(*data))
        t_new = np.arange(ts_arr[0], ts_arr[-1], 1.0 / TARGET_FS)
        return np.interp(t_new, ts_arr, vals, left=vals[0], right=vals[-1])

    def close(self):
        self.running = False
        if self.ser:
            try: self.ser.close()
            except: pass

# ================================================================
# INTERMAGNET Fallback
# ================================================================
def fetch_intermagnet():
    def url_for(offset):
        d = date.today() - timedelta(days=offset)
        return f"{INTERMAGNET_BASE}/{d.year}/{d.strftime('%m')}/KIR{d.strftime('%Y%m%d')}.pt1m.min.gz"

    for offset in range(3):
        try:
            resp = urllib.request.urlopen(url_for(offset), timeout=15)
            with gzip.open(resp) as f:
                lines = f.read().decode("utf-8", "ignore").splitlines()
            values = []
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) > 6:
                    try:
                        values.append(float(parts[6]))
                    except:
                        continue
            if len(values) >= 10:
                raw = np.array(values[-600:])
                target_len = TARGET_FS * BUFFER_SECONDS
                upsampled = resample(raw, target_len)
                np.save(CACHE_FILE, upsampled)
                return upsampled
        except Exception as e:
            print(f"INTERMAGNET fetch failed (day -{offset}): {e}")
    if os.path.exists(CACHE_FILE):
        return np.load(CACHE_FILE)
    return None

# ================================================================
# Spectral Analysis
# ================================================================
class SpectralAnalyzer:
    def __init__(self):
        self.win_samples = int(WINDOW_S * TARGET_FS)
        self.step_samples = int(STEP_S * TARGET_FS)
        self.freqs = FREQS
        self.last_spectrum = np.zeros((1, len(FREQS)))

    def compute(self, signal):
        if len(signal) < self.win_samples:
            return np.array([]), np.array([])
        spectra = []
        times = []
        for start in range(0, len(signal) - self.win_samples + 1, self.step_samples):
            segment = signal[start:start + self.win_samples]
            segment = segment - segment.mean()
            fft = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1.0 / TARGET_FS)
            row = [np.abs(fft[np.argmin(np.abs(freqs - f))]) / (len(segment) / 2.0) for f in self.freqs]
            spectra.append(row)
            times.append((start + self.win_samples // 2) / TARGET_FS)
        S = np.array(spectra)
        T = np.array(times)
        self.last_spectrum = S
        return T, S

# ================================================================
# Main Processing Engine
# ================================================================
class DyadMonitor:
    def __init__(self):
        self.sensor = Magnetometer()
        self.analyzer = SpectralAnalyzer()
        self.tone = ToneGenerator()
        self.logfile = f"{LOG_DIR}/dyad_spectrum_{datetime.utcnow():%Y%m%d_%H%M%S}.csv"
        self.running = True
        self.last_intermagnet_fetch = 0
        self.intermagnet_buffer = None

        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _get_signal(self):
        if self.sensor.ser and self.sensor.ser.is_open:
            buf = self.sensor.get_buffer()
            source = "local"
        else:
            if time.time() - self.last_intermagnet_fetch > 600 or self.intermagnet_buffer is None:
                self.intermagnet_buffer = fetch_intermagnet()
                self.last_intermagnet_fetch = time.time()
            buf = self.intermagnet_buffer[-TARGET_FS * BUFFER_SECONDS:] if self.intermagnet_buffer is not None else np.array([])
            source = "INTERMAGNET"
        return buf, source

    def _adaptive_filter(self, signal):
        recent_116 = np.mean(self.analyzer.last_spectrum[-20:, 4]) if self.analyzer.last_spectrum.shape[0] >= 20 else 0.0
        if recent_116 > 2e-6:
            filtered = bandpass_116_sos(signal, 11.3, 11.9, order=8) * 8.0
        elif recent_116 > 5e-7:
            filtered = bandpass_116_sos(signal, 10.8, 12.4, order=6) * 5.0
        else:
            filtered = bandpass_116_sos(signal, 10.0, 13.5, order=4) * 3.0
        # Add broadband context
        sos_bb = butter(4, [0.5/(TARGET_FS/2), 40/(TARGET_FS/2)], btype='band', output='sos')
        broadband = sosfilt(sos_bb, signal)
        return filtered + 0.15 * broadband

    def _processing_loop(self):
        while self.running:
            try:
                signal, source = self._get_signal()
                if len(signal) < TARGET_FS * 10:
                    time.sleep(1.0)
                    continue

                filtered = self._adaptive_filter(signal)
                times, spectrum = self.analyzer.compute(filtered)

                if spectrum.size:
                    amp_116 = spectrum[-1, 4]
                    self.tone.play(amp_116)
                    self._log(spectrum[-1], source)

                with self.shared_lock:
                    self.shared_signal = filtered.copy()
                    self.shared_times = times.copy()
                    self.shared_spectrum = spectrum.copy()
                    self.shared_source = source

                time.sleep(0.1)
            except Exception as e:
                print("Processing loop error:", e)
                time.sleep(1.0)

    def _log(self, row, source):
        df = pd.DataFrame([row], columns=[f"{f:.1f}Hz" for f in FREQS])
        df["timestamp_utc"] = datetime.utcnow().isoformat()
        df["source"] = source
        header = not os.path.exists(self.logfile)
        df.to_csv(self.logfile, mode='a', header=header, index=False)

    # Shared state for GUI threads
    def __enter__(self):
        self.shared_lock = threading.Lock()
        self.shared_signal = np.zeros(1000)
        self.shared_times = np.array([])
        self.shared_spectrum = np.zeros((1, len(FREQS)))
        self.shared_source = "unknown"
        return self

    def __exit__(self, *args):
        self.running = False
        self.sensor.close()

# ================================================================
# Dash Web Interface
# ================================================================
def create_dash(monitor):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Geomagnetic Field Spectral Monitor — 11.6 Hz Focus"),
        dcc.Interval(id="interval", interval=1000),
        dcc.Graph(id="spectrum")
    ])

    @app.callback(Output("spectrum", "figure"), Input("interval", "n_intervals"))
    def update_plot(_):
        with monitor.shared_lock:
            spec = monitor.shared_spectrum.copy()
            times = monitor.shared_times.copy()
            source = monitor.shared_source

        if spec.size == 0:
            return go.Figure().update_layout(title="No data")

        x = (times - times.max()) if len(times) else [0]
        fig = go.Figure(data=go.Heatmap(
            z=spec.T, x=x, y=FREQS.tolist(),
            colorscale="Viridis", colorbar=dict(title="Power")))
        fig.update_xaxes(autorange="reversed", title="Seconds ago")
        fig.update_yaxes(title="Frequency (Hz)")
        fig.update_layout(title=f"Spectral Power — Source: {source}")
        return fig

    threading.Thread(target=app.run_server, kwargs={"host": "0.0.0.0", "port": 8050}, daemon=True).start()

# ================================================================
# Matplotlib Local GUI
# ================================================================
def run_local_gui(monitor):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})
    line, = ax1.plot([], [], 'c')
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Filtered Field (11.6 Hz adaptive lock)")

    while plt.fignum_exists(fig.number):
        try:
            with monitor.shared_lock:
                sig = monitor.shared_signal.copy()
                times = monitor.shared_times.copy()
                spec = monitor.shared_spectrum.copy()

            if sig.size:
                t = np.arange(len(sig)) / TARGET_FS
                line.set_data(t[-3000:], sig[-3000:])
                ax1.relim(); ax1.autoscale_view()

            if spec.size:
                ax2.clear()
                extent = [times[-1]-times[0], 0, FREQS[0], FREQS[-1]] if len(times) else [0, 1, 0, 1]
                ax2.imshow(spec.T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
                ax2.axhline(11.6, color='yellow', linewidth=1.5)
                ax2.set_xlabel("Seconds ago → Now")
                ax2.set_ylabel("Frequency (Hz)")
                ax2.set_title("Spectral Power Density")

            plt.pause(0.2)
        except:
            time.sleep(0.1)

# ================================================================
# Main Execution
# ================================================================
if __name__ == "__main__":
    with DyadMonitor() as monitor:
        create_dash(monitor)
        print("Dash server: http://localhost:8050")
        print(f"CSV log: {monitor.logfile}")
        run_local_gui(monitor)
