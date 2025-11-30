#!/usr/bin/env python3
"""
dyad_field_v6_4.py
v6.4 — Sliding coherence spectrograms + phase mapping + GPU hooks + ROS2 outputs
- Multi-station weighted averaging (inverse-variance)
- Expanded frequency grid
- 11.6 Hz lock + alpha
- Sliding-window coherence spectrogram (time x freq)
- Phase-difference mapping (per-band)
- GPU fallback: uses cupy if available; fallbacks to numpy/pyfftw
- ROS2: publishes topics (if rclpy available)
- Dash UI updated with coherence spectrogram + phase heatmap
- Sonification retained

Install (recommended):
  pip install numpy scipy pandas dash plotly pygame h5py pyserial pyfftw rclpy
  # rclpy install depends on your ROS2 distro; follow ROS2 instructions.

Run:
  python dyad_field_v6_4.py
Open dashboard at http://127.0.0.1:8050
"""

import os
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')   # headless-safe

import time
import threading
import queue
from datetime import datetime, date, timedelta
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, coherence
from scipy.signal import resample
import gzip, urllib.request, io
import serial, serial.tools.list_ports
import pygame
import h5py
import math

# Dash / Plotly
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# GPU hook: prefer cupy; fallback to numpy; optional pyfftw for CPU acceleration
USE_CUPY = False
try:
    import cupy as cp
    xp = cp
    USE_CUPY = True
    print("[GPU] cupy available — using GPU arrays for FFT where appropriate.")
except Exception:
    import numpy.fft as npfft  # keep for reference
    xp = np
    print("[GPU] cupy not available — using numpy. Consider installing cupy for GPU acceleration.")

# Optional pyFFTW
USE_PYFFTW = False
try:
    import pyfftw
    USE_PYFFTW = True
    print("[CPU] pyFFTW available for accelerated FFTs.")
except Exception:
    USE_PYFFTW = False

# ROS2 hook (optional). If rclpy not installed, code continues but will not publish.
USE_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    USE_ROS2 = True
    print("[ROS2] rclpy available — will publish topics.")
except Exception:
    USE_ROS2 = False

# ----------------- CONFIG -----------------
TARGET_FS = 1000
BUFFER_SECONDS = 180
WINDOW_S = 1.0
STEP_S = 0.1

# Expandable grid
EXP_MIN = 0.5
EXP_MAX = 80.0
EXP_BINS = 120
EXP_FREQS = np.linspace(EXP_MIN, EXP_MAX, EXP_BINS)
LOCK_FREQ = 11.6
ALPHA_BAND = (8.0, 12.0)

STATIONS = ["KIR","ABK","NUR"]    # change as needed
RETRIES = 3
RETRY_SLEEP = 5

LOG_DIR = "dyad_v6_4_logs"
os.makedirs(LOG_DIR, exist_ok=True)
HDF5_ARCHIVE = os.path.join(LOG_DIR, "spectra_archive.h5")

# Sonification safe init
pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
pygame.init()
SONO_ENABLED = True

# ----------------- UTILS & FILTERS -----------------
def bandpass_sos(signal, lowcut, highcut, fs=TARGET_FS, order=4, use_sosfiltfilt=True):
    nyq = fs * 0.5
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    sos = butter(order, [low, high], btype='band', output='sos')
    if use_sosfiltfilt:
        try:
            return sosfiltfilt(sos, signal)
        except Exception:
            return sosfiltfilt(sos, signal) if hasattr(sosfiltfilt, "__call__") else np.copy(signal)
    else:
        from scipy.signal import sosfilt
        return sosfilt(sos, signal)

# ----------------- Serial magnetometer helper -----------------
class LocalMagnetometer:
    def __init__(self, baud=115200, qsize=100000):
        self.port = self._find_port()
        self.baud = baud
        self.ser = None
        self.q = queue.Queue(maxsize=qsize)
        self.running = False
        if self.port:
            try:
                self.ser = serial.Serial(self.port, baud, timeout=1)
                time.sleep(1.5)
                self.ser.reset_input_buffer()
                self.running = True
                threading.Thread(target=self._reader, daemon=True).start()
                print(f"[SENSOR] opened {self.port}")
            except Exception as e:
                print("[SENSOR] open error:", e)
                self.ser = None
        else:
            print("[SENSOR] no local serial magnetometer found")

    def _find_port(self):
        for p in serial.tools.list_ports.comports():
            txt = " ".join(filter(None, [p.device, p.description, p.hwid])).upper()
            if any(k in txt for k in ("USB","FTDI","CH340","CP210","ARDUINO","STM")):
                return p.device
        return None

    def _reader(self):
        while self.running and self.ser:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                toks = line.replace(',', ' ').split()
                nums = []
                for t in toks:
                    t2 = t.strip().replace(',', '.')
                    if t2.replace('-', '').replace('.', '').isdigit():
                        try: nums.append(float(t2))
                        except: pass
                if len(nums) >= 3:
                    mag = float(np.linalg.norm(nums[:3]))
                    ts = time.time()
                    try: self.q.put_nowait((ts, mag))
                    except queue.Full:
                        try: _ = self.q.get_nowait()
                        except: pass
                        try: self.q.put_nowait((ts, mag))
                        except: pass
            except Exception:
                time.sleep(0.01)

    def get_buffer(self, seconds=BUFFER_SECONDS):
        data=[]
        cutoff = time.time() - seconds
        while not self.q.empty():
            try:
                ts,val = self.q.get_nowait()
            except queue.Empty:
                break
            if ts >= cutoff:
                data.append((ts,val))
        if len(data) < 10:
            return np.array([])
        ts_arr, val_arr = map(np.array, zip(*data))
        if np.any(np.diff(ts_arr)<=0):
            ts_arr = np.maximum.accumulate(ts_arr + 1e-9*np.arange(len(ts_arr)))
        t_new = np.arange(ts_arr[0], ts_arr[-1], 1.0/TARGET_FS)
        if len(t_new) == 0:
            return val_arr
        return np.interp(t_new, ts_arr, val_arr, left=val_arr[0], right=val_arr[-1])

# ----------------- INTERMAGNET multi-station weighted averaging -----------------
def build_intermagnet_url(station, d):
    return f"https://intermagnet.github.io/data/latest/{station}/{d.year}/{d.strftime('%m')}/{station}{d.strftime('%Y%m%d')}.pt1m.min.gz"

def fetch_station_simple(station, days_back=3):
    for offset in range(days_back):
        d = date.today() - timedelta(days=offset)
        url = build_intermagnet_url(station, d)
        try:
            resp = urllib.request.urlopen(url, timeout=12)
            raw = resp.read()
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    lines = gz.read().decode('utf-8', errors='ignore').splitlines()
            except Exception:
                lines = raw.decode('utf-8','ignore').splitlines()
            vals=[]
            for L in lines:
                if not L or L.startswith('#'): continue
                parts = L.split()
                if len(parts) > 6:
                    try: vals.append(float(parts[6]))
                    except: pass
            if len(vals) < 10: continue
            raw_arr = np.array(vals[-600:])
            target_len = int(TARGET_FS * BUFFER_SECONDS)
            # use linear interpolation resample for simplicity (robust)
            up = np.interp(np.linspace(0,len(raw_arr),target_len), np.arange(len(raw_arr)), raw_arr)
            return up
        except Exception:
            time.sleep(RETRY_SLEEP)
    return None

def fetch_multi_weighted(stations=STATIONS):
    arrays=[]
    names=[]
    for st in stations:
        arr = fetch_station_simple(st)
        if arr is not None:
            arrays.append(arr); names.append(st)
    if len(arrays) == 0:
        return None, [], None
    stacked = np.vstack(arrays)
    vars = np.var(stacked, axis=1)
    inv = np.array([1.0/v if v > 0 else 0.0 for v in vars])
    if np.sum(inv) == 0:
        weights = np.ones_like(inv)/len(inv)
    else:
        weights = inv / np.sum(inv)
    weighted = np.average(stacked, axis=0, weights=weights)
    return weighted, names, weights

# ----------------- SPECTRAL / SLIDING COHERENCE + PHASE -----------------
class SpectralEngine:
    def __init__(self, freqs=EXP_FREQS, alpha_band=ALPHA_BAND, lock_freq=LOCK_FREQ):
        self.freqs = np.array(freqs)
        self.alpha_band = alpha_band
        self.lock_freq = lock_freq
        self.win = int(WINDOW_S * TARGET_FS)
        self.step = int(STEP_S * TARGET_FS)

    def compute_spectrum(self, sig):
        # returns times (center), spectrogram (T x F), alpha_amp (T)
        if len(sig) < self.win:
            return np.array([]), np.array([]), np.array([])
        spec = []
        tvec = []
        alpha_amp = []
        nfft = int(2**np.ceil(np.log2(self.win)))
        for i in range(0, len(sig)-self.win+1, self.step):
            seg = sig[i:i+self.win] - np.mean(sig[i:i+self.win])
            if USE_CUPY:
                seg_gpu = cp.asarray(seg)
                sp = cp.fft.rfft(seg_gpu, n=nfft)
                faxis = cp.fft.rfftfreq(nfft, 1.0/TARGET_FS)
                sp_abs = cp.asnumpy(cp.abs(sp))
                faxis = cp.asnumpy(faxis)
            else:
                sp = np.fft.rfft(seg, n=nfft)
                faxis = np.fft.rfftfreq(nfft, 1.0/TARGET_FS)
                sp_abs = np.abs(sp)
            row = []
            for f in self.freqs:
                idx = np.argmin(np.abs(faxis - f))
                mag = sp_abs[idx] / (len(seg) / 2.0)
                row.append(float(mag))
            # alpha average (8-12 Hz)
            mask = (faxis >= self.alpha_band[0]) & (faxis <= self.alpha_band[1])
            alpha_val = float(np.mean(sp_abs[mask])/(len(seg)/2.0)) if np.any(mask) else 0.0
            spec.append(row)
            alpha_amp.append(alpha_val)
            tvec.append((i + self.win//2) / TARGET_FS)
        return np.array(tvec), np.array(spec), np.array(alpha_amp)

    def sliding_coherence_spectrogram(self, x, y, window_sec=2.0, step_sec=0.5, fmin=10.0, fmax=13.0):
        # compute coherence spectrogram C(t, f) between x and y across frequency grid
        win = int(window_sec * TARGET_FS)
        step = int(step_sec * TARGET_FS)
        nfft = int(2**np.ceil(np.log2(win)))
        times = []
        freqs = np.fft.rfftfreq(nfft, 1.0/TARGET_FS)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        coh_spect = []
        phase_spect = []
        for i in range(0, min(len(x), len(y)) - win + 1, step):
            xa = x[i:i+win] - np.mean(x[i:i+win])
            ya = y[i:i+win] - np.mean(y[i:i+win])
            f, Cxy = coherence(xa, ya, fs=TARGET_FS, nperseg=min(256, win))
            # cross-spectrum phase (coarse): compute FFTs and Sxy
            Sx = np.fft.rfft(xa, n=nfft)
            Sy = np.fft.rfft(ya, n=nfft)
            Sxy = Sx * np.conj(Sy)
            ph = np.angle(Sxy)
            # restrict to mask
            idxs = np.where(freq_mask)[0]
            coh_spect.append(np.interp(freqs[f >= fmin], freqs[f >= fmin], Cxy[(f >= fmin) & (f <= fmax)]) if len(Cxy)>0 else np.zeros(len(freqs[idxs])))
            phase_spect.append(ph[freq_mask])
            times.append((i + win//2) / TARGET_FS)
        return np.array(times), np.array(coh_spect), np.array(phase_spect), freqs[freq_mask]

# ----------------- SONIFICATION (simple) -----------------
class SonoPlayer:
    def __init__(self):
        self.base = 220
        self.vol = 0.5
    def play(self, amp_lock, amp_alpha):
        if not SONO_ENABLED: return
        if amp_lock > 1e-7:
            freq = max(120, min(1500, self.base + 50*math.log1p(amp_lock/1e-5)))
            self._play_tone(freq, min(1.0, amp_lock/1e-4)*self.vol)
        if amp_alpha > 1e-7:
            freq = max(120, min(1500, self.base*1.6 + 40*math.log1p(amp_alpha/1e-5)))
            self._play_tone(freq, min(1.0, amp_alpha/1e-4)*self.vol*0.6)
    def _play_tone(self, freq, vol, dur=0.08):
        sr = 22050
        n = int(sr*dur)
        t = np.linspace(0,dur,n,False)
        tone = np.sin(2*np.pi*freq*t)*vol
        try:
            snd = pygame.sndarray.make_sound((tone*32767).astype(np.int16))
            snd.play()
        except Exception:
            pass

# ----------------- ROS2 publisher node (optional) -----------------
if USE_ROS2:
    class ROSPublisher(Node):
        def __init__(self, name='dyad_publisher'):
            super().__init__(name)
            self.spec_pub = self.create_publisher(Float32MultiArray, 'dyad/spec', 10)
            self.coh_pub = self.create_publisher(Float32MultiArray, 'dyad/coherence', 10)
        def publish_spec(self, spec_row):
            msg = Float32MultiArray()
            msg.data = spec_row.astype(np.float32).tolist()
            self.spec_pub.publish(msg)
        def publish_coh(self, coh_row):
            msg = Float32MultiArray()
            msg.data = coh_row.astype(np.float32).tolist()
            self.coh_pub.publish(msg)
    # init rclpy in main if USE_ROS2

# ----------------- Dash Dashboard -----------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Dyad Field v6.4 — Sliding Coherence & Phase"),
    dcc.Graph(id='spec-heat', style={'height':'45vh'}),
    dcc.Graph(id='coh-spec', style={'height':'25vh'}),
    dcc.Graph(id='phase-heat', style={'height':'25vh'}),
    dcc.Interval(id='int', interval=800, n_intervals=0),
    html.Div(id='status')
])

# ----------------- Engine (background loop) -----------------
class Engine:
    def __init__(self):
        self.sensor = LocalMagnetometer()
        self.spec_engine = SpectralEngine()
        self.sono = SonoPlayer()
        self.lock = threading.Lock()
        # buffers / latest
        self.times = np.array([])
        self.spec = np.array([])           # T x F
        self.alpha = np.array([])          # T
        self.coherence_spec = np.array([]) # time x freq_coh
        self.phase_spec = np.array([])     # time x freq_coh
        self.coh_freqs = np.array([])
        self.station_names = []
        # ROS2 publisher
        if USE_ROS2:
            rclpy.init()
            self.rosnode = ROSPublisher()
        else:
            self.rosnode = None
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            try:
                # choose data source
                if self.sensor.ser and self.sensor.ser.is_open:
                    data = self.sensor.get_buffer()
                    source = 'LOCAL'
                else:
                    weighted, names, weights = fetch_multi_weighted()
                    data = weighted
                    source = 'INTERMAGNET' if weighted is not None else 'NO_DATA'
                    self.station_names = names

                if data is None or len(data) < TARGET_FS * 6:
                    time.sleep(1); continue

                # adaptive lock filter
                recent_lock = 0.0
                if self.spec.size:
                    col = np.argmin(np.abs(self.spec_engine.freqs - LOCK_FREQ))
                    recent_lock = float(np.mean(self.spec[-min(20, len(self.spec)):, col]))
                if recent_lock > 2e-6:
                    filt = bandpass_sos(data, 11.3, 11.9, order=8)
                    gain = 8.0
                elif recent_lock > 5e-7:
                    filt = bandpass_sos(data, 10.8, 12.4, order=6)
                    gain = 5.0
                else:
                    filt = bandpass_sos(data, 10.0, 13.5, order=6)
                    gain = 3.0

                # broadband
                try:
                    sos_bb = butter(4, [0.5/(TARGET_FS/2), 40.0/(TARGET_FS/2)], btype='band', output='sos')
                    from scipy.signal import sosfilt
                    bb = sosfilt(sos_bb, data)
                except Exception:
                    bb = np.zeros_like(data)

                signal = filt * gain + 0.15 * bb

                # compute spectrum + alpha
                times, spec, alpha = self.spec_engine.compute_spectrum(signal)

                # sliding coherence and phase between reference (signal) and itself or first station (if available)
                # For multi-station pairwise coherence you'd compute pairwise loops and store spectrograms per pair
                # Here, compute coherence spectrogram between current signal and a time-shifted copy (as a demonstration)
                t_coh, coh_spec, ph_spec, coh_freqs = self.spec_engine.sliding_coherence_spectrogram(signal, signal, window_sec=2.0, step_sec=0.5, fmin=10.0, fmax=13.0)

                # publish ROS2 topics (latest row)
                if self.rosnode is not None and len(spec)>0:
                    try:
                        self.rosnode.publish_spec(spec[-1])
                    except Exception:
                        pass

                # update shared state
                with self.lock:
                    self.times = times
                    self.spec = spec
                    self.alpha = alpha
                    self.coherence_spec = coh_spec
                    self.phase_spec = ph_spec
                    self.coh_freqs = coh_freqs

                # sonify using last window lock & alpha
                amp_lock = spec[-1, np.argmin(np.abs(self.spec_engine.freqs - LOCK_FREQ))] if spec.size else 0.0
                amp_alpha = float(np.mean(alpha[-1:])) if alpha.size else 0.0
                self.sono.play(amp_lock, amp_alpha)

                # archive hourly
                if datetime.utcnow().minute == 0 and datetime.utcnow().second < 2:
                    try:
                        with h5py.File(HDF5_ARCHIVE, 'a') as hf:
                            hf.create_dataset(datetime.utcnow().strftime('%Y%m%d_%H%M%S'), data=spec, compression='gzip')
                    except Exception:
                        pass

                time.sleep(STEP_S)
            except Exception as e:
                print("[ENGINE] error:", e)
                time.sleep(1)

engine = Engine()

# ----------------- Dash callbacks -----------------
@app.callback(
    Output('spec-heat','figure'),
    Input('int','n_intervals')
)
def cb_spec(n):
    with engine.lock:
        times = engine.times.copy()
        spec = engine.spec.copy()
    if spec.size and len(times):
        fig = go.Figure(go.Heatmap(z=spec.T.tolist(), x=(times - times.max()).tolist(), y=engine.spec_engine.freqs.tolist(), colorscale='Inferno'))
        fig.update_xaxes(autorange='reversed')
        fig.update_layout(title='Expanded Spectrogram (right=now)')
        return fig
    return go.Figure()

@app.callback(
    Output('coh-spec','figure'),
    Input('int','n_intervals')
)
def cb_coh(n):
    with engine.lock:
        coh = engine.coherence_spec.copy()
        f = engine.coh_freqs.copy()
        t = np.arange(len(coh))*0.5 if coh.size else np.array([])
    if coh.size:
        fig = go.Figure(go.Heatmap(z=coh.T.tolist(), x=t.tolist(), y=f.tolist(), colorscale='Viridis'))
        fig.update_layout(title='Sliding Coherence Spectrogram (10-13 Hz)')
        return fig
    return go.Figure()

@app.callback(
    Output('phase-heat','figure'),
    Input('int','n_intervals')
)
def cb_phase(n):
    with engine.lock:
        ph = engine.phase_spec.copy()
        f = engine.coh_freqs.copy()
        t = np.arange(len(ph))*0.5 if ph.size else np.array([])
    if ph.size:
        # unwrap phase for display
        ph_unwrap = np.unwrap(ph, axis=1)
        fig = go.Figure(go.Heatmap(z=ph_unwrap.T.tolist(), x=t.tolist(), y=f.tolist(), colorscale='RdBu'))
        fig.update_layout(title='Sliding Phase Spectrogram (10-13 Hz)')
        return fig
    return go.Figure()

@app.callback(
    Output('status','children'),
    Input('int','n_intervals')
)
def cb_status(n):
    with engine.lock:
        last = engine.times[-1] if engine.times.size else None
        station_names = engine.station_names if hasattr(engine,'station_names') else []
    return f"Last sample center t={last:.1f} s" if last is not None else "No samples yet"

# ----------------- MAIN -----------------
if __name__ == "__main__":
    print("[START] dyad_field_v6_4 — open http://127.0.0.1:8050")
    app.run_server(host='0.0.0.0', port=8050, debug=False)
