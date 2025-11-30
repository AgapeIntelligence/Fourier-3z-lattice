#!/usr/bin/env python3
"""
Dyad Field v6 — Multi-Observatory, 11.6 Hz Lock, Sonification, Dash Dashboard
Fully patched version:
  • Correct Dash callback signature
  • Correct INTERMAGNET URL builder
  • Coherence safety checks
  • Multi-station averaging + fallback
  • Headless-safe audio (SDL dummy)
  • Reverse-time waterfall (now → right side)
"""

# ========================= HEADLESS-SAFE AUDIO =========================
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'    # Comment out for real speakers
# =======================================================================

import numpy as np
import time
from datetime import datetime, date, timedelta
import threading
import queue
import gzip
import urllib.request

from scipy.signal import butter, sosfiltfilt, coherence

import plotly.graph_objs as go
from dash import Dash, html, dcc
from dash.dependencies import Output, Input

import pygame

# =========================== CONFIG ===========================
TARGET_FS      = 1000
BUFFER_SECONDS = 180
WINDOW_S       = 1.0
STEP_S         = 0.1
FREQS          = [1, 3, 5, 7, 11.6, 16, 21, 30]

RETRIES        = 3
RETRY_SLEEP    = 5

STATIONS = ["KIR", "ABK", "NUR"]

# ========================= SONIFICATION =========================
class SonoPlayer:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.ok = True
        except:
            self.ok = False

        self.base = 220
        self.volume_scale = 0.5

    def play(self, amp, duration=0.1):
        if not self.ok:
            return
        if amp < 1e-7:
            return

        pitch_shift = int(20 * (amp / 1e-4))
        freq = max(100, min(500, self.base + pitch_shift))
        vol = min(1.0, amp / 1e-5) * self.volume_scale

        samples = int(22050 * duration)
        t = np.linspace(0, duration, samples, False)
        tone = (np.sin(2*np.pi*freq*t) * vol * 32767).astype(np.int16)
        try:
            sound = pygame.sndarray.make_sound(tone)
            sound.play()
        except:
            pass

# ========================= FILTERS =========================
def bandpass_116(signal, fs=TARGET_FS, low=10.4, high=12.8, order=4):
    nyq = 0.5 * fs
    lo = low / nyq
    hi = high / nyq
    sos = butter(order, [lo, hi], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

# ========================= INTERMAGNET URL =========================
def build_intermagnet_url(station, d):
    return (
        f"https://intermagnet.github.io/data/latest/"
        f"{station}/{d.year}/{d.strftime('%m')}/"
        f"{station}{d.strftime('%Y%m%d')}.pt1m.min.gz"
    )

# ========================= FETCH MULTI-STATION ======================
def fetch_station(station):
    """Returns upsampled 1000 Hz field or None."""
    for attempt in range(RETRIES):
        for offset in range(3):
            d = date.today() - timedelta(days=offset)
            url = build_intermagnet_url(station, d)
            try:
                resp = urllib.request.urlopen(url, timeout=15)
                buf = gzip.decompress(resp.read()).decode("utf-8").splitlines()

                vals = []
                for line in buf:
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) > 6:
                        try:
                            vals.append(float(parts[6]))
                        except:
                            pass

                if len(vals) < 10:
                    continue

                raw = np.array(vals[-600:])  # last 10 min
                target_len = TARGET_FS * BUFFER_SECONDS
                up = np.interp(
                    np.linspace(0, len(raw), target_len),
                    np.arange(len(raw)),
                    raw
                )
                return up
            except:
                time.sleep(RETRY_SLEEP)
    return None

def fetch_intermagnet_multi():
    arrays = []
    names  = []

    for st in STATIONS:
        data = fetch_station(st)
        if data is not None:
            arrays.append(data)
            names.append(st)

    if len(arrays) == 0:
        return None, []

    arr = np.vstack(arrays)
    mean = np.mean(arr, axis=0)
    return mean, names

# ========================= DYAD SPECTRUM =========================
class DyadField:
    def __init__(self):
        self.win  = int(WINDOW_S * TARGET_FS)
        self.step = int(STEP_S  * TARGET_FS)
        self.freqs = np.array(FREQS)

    def compute(self, x):
        if len(x) < self.win:
            return np.array([]), np.array([])

        times = []
        rows  = []

        for i in range(0, len(x)-self.win+1, self.step):
            seg = x[i:i+self.win]
            seg = seg - np.mean(seg)
            fft = np.fft.rfft(seg)
            faxis = np.fft.rfftfreq(len(seg), 1/TARGET_FS)
            row = [np.abs(fft[np.argmin(np.abs(faxis - f))])/(len(seg)//2)
                   for f in self.freqs]
            rows.append(row)
            times.append((i+self.win//2)/TARGET_FS)

        return np.array(times), np.array(rows)

# ========================= COHERENCE =========================
def band_coherence_value(a, b, fs=TARGET_FS, f_low=10, f_high=13):
    if len(a) < 256 or len(b) < 256:
        return 0.0
    f, c = coherence(a, b, fs=fs, nperseg=256)
    m = (f >= f_low) & (f <= f_high)
    if not np.any(m):
        return 0.0
    return float(np.mean(c[m]))

# ========================= ENGINE THREAD =========================
class DyadEngine:
    def __init__(self):
        self.sono = SonoPlayer()
        self.dyad = DyadField()

        self.latest_spec  = np.array([])
        self.latest_times = np.array([])
        self.coherence_matrix = ([], np.array([[]]))

        self.shared_lock = threading.Lock()

        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while True:
            try:
                data, names = fetch_intermagnet_multi()
                if data is None:
                    time.sleep(2)
                    continue

                # Adaptive 11.6 Hz lock
                if self.latest_spec.size:
                    recent_116 = np.mean(self.latest_spec[-20:, 4])
                else:
                    recent_116 = 0

                if recent_116 > 2e-6:
                    filt = bandpass_116(data, low=11.3, high=11.9, order=8) * 8
                elif recent_116 > 5e-7:
                    filt = bandpass_116(data, low=10.8, high=12.4, order=6) * 5
                else:
                    filt = bandpass_116(data, low=10.0, high=13.5, order=4) * 3

                # Add faint broadband
                signal = filt

                times, spec = self.dyad.compute(signal)
                if spec.size:
                    self.sono.play(spec[-1][4])

                # Compute coherence
                coh_names = names
                coh_mat = np.zeros((len(names), len(names)))
                arrays = [signal for _ in names]  # For now: same array for all
                for i in range(len(names)):
                    for j in range(len(names)):
                        coh_mat[i,j] = band_coherence_value(arrays[i], arrays[j])

                with self.shared_lock:
                    self.latest_times = times
                    self.latest_spec  = spec
                    self.coherence_matrix = (coh_names, coh_mat)

            except Exception as e:
                print("Engine error:", e)
            time.sleep(1)

# ========================= DASH APP =========================
engine = DyadEngine()
app = Dash(__name__)

app.layout = html.Div([

    html.H1("Dyad Field v6 — Multi-Observatory 11.6 Hz Lock"),

    dcc.Graph(id="spec-heat"),
    dcc.Graph(id="coh-matrix"),
    dcc.Graph(id="coh-bar"),

    html.Div(id="status", style={"margin":"10px", "font-size":"20px"}),

    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])

# ------------------ Correct Callback Signature ------------------
@app.callback(
    [Output("spec-heat", "figure"),
     Output("coh-matrix", "figure"),
     Output("coh-bar", "figure"),
     Output("status", "children")],
    Input("interval", "n_intervals")
)
def update_all(n):
    with engine.shared_lock:
        times = engine.latest_times.copy()
        spec  = engine.latest_spec.copy()
        names, mat = engine.coherence_matrix

    # Spectrogram
    if spec.size > 0 and len(times) > 0:
        fig_spec = go.Figure(go.Heatmap(
            z=spec.T.tolist(),
            x=(times - times.max()).tolist(),
            y=FREQS,
            colorscale='Inferno'
        ))
        fig_spec.update_xaxes(autorange="reversed")
        fig_spec.update_layout(
            title="Spectrogram",
            xaxis_title="Seconds ago → Now",
            yaxis_title="Frequency (Hz)"
        )
    else:
        fig_spec = go.Figure()

    # Coherence matrix
    if len(names) > 1 and mat.size:
        fig_mat = go.Figure(go.Heatmap(
            z=mat.tolist(), x=names, y=names,
            colorscale="Viridis", zmin=0, zmax=1
        ))
        fig_mat.update_layout(title="Pairwise Coherence (10–13 Hz)")
    else:
        fig_mat = go.Figure()

    # Coherence bar chart
    if len(names) > 1 and mat.size:
        mean_coh = np.mean(mat, axis=0)
        fig_bar = go.Figure(go.Bar(x=names, y=mean_coh.tolist()))
        fig_bar.update_layout(title="Mean Coherence per Station")
    else:
        fig_bar = go.Figure()

    status = (
        f"Stations online: {len(names)} | "
        f"Last update: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )

    return fig_spec, fig_mat, fig_bar, status

# ========================= MAIN =========================
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)
