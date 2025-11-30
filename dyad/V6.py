# dyad_field_v6_2_geom_alpha.py
# v6.2 – 11.6 Hz Lock + Alpha (8-12 Hz) + Sonification + Multi-Observatory Averaging + GIC weighting
# Requires: numpy, scipy, matplotlib, pandas, pygame, dash, plotly

import os
import time
import threading
import queue
from datetime import datetime, date, timedelta
import numpy as np
from scipy.signal import butter, sosfilt, resample
import serial
import serial.tools.list_ports
import urllib.request, gzip
import pygame
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# =================== CONFIG ===================
TARGET_FS = 1000
WINDOW_S = 1.0
STEP_S = 0.1
FREQS = [1, 3, 5, 7, 11.6, 16, 21, 30]
ALPHA_FREQS = np.linspace(8, 12, 5)  # secondary alpha tracking
BUFFER_SECONDS = 180
SERIAL_BAUD = 115200
CACHE_FILE = "kiruna_latest.npy"
INTERMAGNET_STATIONS = ["KIR", "HON", "CLF"]  # multi-observatory
SONO_BASE_FREQ = 220
SONO_ENABLED = True

# =================== HEADLESS AUDIO SUPPORT ===================
os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

# =================== 11.6 Hz SOS LOCK FILTER ===================
def bandpass_sos(signal, lowcut, highcut, order=4, fs=TARGET_FS):
    nyq = 0.5 * fs
    low = max(lowcut/nyq, 0.001)
    high = min(highcut/nyq, 0.999)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal)

# =================== SONIFICATION ===================
class SonoPlayer:
    def __init__(self):
        self.base_note = SONO_BASE_FREQ
        self.volume_scale = 0.5

    def play_amp(self, amp_116, amp_alpha=None, duration=0.1):
        if not SONO_ENABLED or (amp_116 < 1e-7 and (amp_alpha is None or amp_alpha < 1e-7)):
            return
        # 11.6 Hz tone
        if amp_116 >= 1e-7:
            freq = max(100, min(500, self.base_note + int(20*(amp_116/1e-4))))
            vol = min(1.0, amp_116/1e-5)*self.volume_scale
            self._play_tone(freq, vol, duration)
        # Alpha tone (optional secondary channel)
        if amp_alpha is not None and amp_alpha >= 1e-7:
            freq = max(100, min(500, self.base_note + 100 + int(20*(amp_alpha/1e-4))))
            vol = min(1.0, amp_alpha/1e-5)*self.volume_scale
            self._play_tone(freq, vol, duration)

    def _play_tone(self, freq, vol, duration):
        samples = int(TARGET_FS*duration)
        t = np.linspace(0, duration, samples, False)
        tone = np.sin(2*np.pi*freq*t) * vol
        sound = pygame.sndarray.make_sound((tone*32767).astype(np.int16))
        sound.play()

# =================== LOCAL MAGNETOMETER ===================
class LocalMagnetometer:
    def __init__(self):
        self.port = self.find_device()
        self.ser = None
        self.q = queue.Queue(maxsize=30000)
        self.running = False
        self.thread = None
        if self.port:
            try:
                self.ser = serial.Serial(self.port, SERIAL_BAUD, timeout=1)
                time.sleep(2)
                self.ser.reset_input_buffer()
                self.start()
                print(f"Local sensor connected: {self.port}")
            except Exception as e:
                print(f"Failed to open local sensor ({e})")
                self.ser = None

    def find_device(self):
        for p in serial.tools.list_ports.comports():
            desc = f"{p.device} {p.description} {p.hwid}"
            if any(k in desc.upper() for k in ["USB","FTDI","CH340","CP210","STM","ARDUINO"]):
                return p.device
        return None

    def reader(self):
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                nums = []
                for token in line.replace(',', ' ').split():
                    token = token.strip().replace(',', '.')
                    if token.replace('-', '').replace('.', '').isdigit():
                        try: nums.append(float(token))
                        except: pass
                if len(nums)>=3:
                    mag = np.linalg.norm(nums[:3])
                    ts = time.time()
                    try: self.q.put_nowait((ts, mag))
                    except queue.Full:
                        try: self.q.get_nowait()
                        except: pass
                        self.q.put_nowait((ts, mag))
            except: time.sleep(0.01)

    def start(self):
        if self.ser:
            self.running = True
            self.thread = threading.Thread(target=self.reader, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try: self.ser.close()
            except: pass

    def get_buffer(self):
        data = []
        cutoff = time.time() - BUFFER_SECONDS
        while True:
            try:
                ts,val = self.q.get_nowait()
                if ts>=cutoff: data.append((ts,val))
            except queue.Empty:
                break
        if len(data)<20:
            return np.array([])
        ts,vals = zip(*data)
        ts = np.array(ts)
        vals = np.array(vals)
        t_new = np.arange(ts[0], ts[-1]+1/TARGET_FS, 1/TARGET_FS)
        vals_interp = np.interp(t_new, ts, vals, left=vals[0], right=vals[-1])
        return vals_interp

# =================== INTERMAGNET MULTI-STATION FETCH ===================
def build_intermagnet_url(station, d):
    return f"https://intermagnet.github.io/data/latest/{station}/{d.year}/{d.strftime('%m')}/{station}{d.strftime('%Y%m%d')}.pt1m.min.gz"

def fetch_intermagnet_multi(stations=INTERMAGNET_STATIONS, retries=3):
    arrays = []
    names = []
    for s in stations:
        for offset in range(3):
            try:
                d = date.today() - timedelta(days=offset)
                url = build_intermagnet_url(s,d)
                resp = urllib.request.urlopen(url, timeout=15)
                with gzip.open(resp) as f:
                    vals = []
                    for line in f.read().decode('utf-8','ignore').splitlines():
                        if line.startswith('#') or not line.strip(): continue
                        parts = line.split()
                        if len(parts)>6:
                            try: vals.append(float(parts[6]))
                            except: continue
                    if len(vals)<10: continue
                    raw = np.array(vals[-600:])
                    up = resample(raw, TARGET_FS*BUFFER_SECONDS)
                    arrays.append(up)
                    names.append(s)
                    break
            except Exception as e:
                print(f"{s} day-{offset} failed: {e}")
                time.sleep(5)
    return names, arrays

# =================== DYAD FIELD CORE ===================
class DyadField:
    def __init__(self, freqs=FREQS, alpha_freqs=ALPHA_FREQS):
        self.win = int(WINDOW_S*TARGET_FS)
        self.step = int(STEP_S*TARGET_FS)
        self.freqs = np.array(freqs)
        self.alpha_freqs = np.array(alpha_freqs)

    def compute(self, signal):
        if len(signal)<self.win: return np.array([]), np.array([]), np.array([])
        spec, alpha_spec = [], []
        tvec=[]
        for i in range(0,len(signal)-self.win+1,self.step):
            seg = signal[i:i+self.win]-np.mean(signal[i:i+self.win])
            fft = np.fft.rfft(seg)
            faxis = np.fft.rfftfreq(len(seg),1/TARGET_FS)
            row = [np.abs(fft[np.argmin(np.abs(faxis-f))])/(len(seg)//2) for f in self.freqs]
            arow = [np.abs(fft[np.argmin(np.abs(faxis-f))])/(len(seg)//2) for f in self.alpha_freqs]
            spec.append(row)
            alpha_spec.append(arow)
            tvec.append((i+self.win//2)/TARGET_FS)
        return np.array(tvec), np.array(spec), np.array(alpha_spec)

# =================== MAIN DASH APP ===================
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Dyad Field v6.2 — 11.6 Hz Lock + Alpha"),
    html.Div(id='status'),
    dcc.Graph(id='spec-heat'),
    dcc.Graph(id='coh-matrix'),
    dcc.Graph(id='coh-bar'),
    dcc.Interval(id='interval', interval=1000, n_intervals=0)
])

class Engine:
    def __init__(self):
        self.sensor = LocalMagnetometer()
        self.dyad = DyadField()
        self.latest_times = np.array([])
        self.latest_spec = np.array([])
        self.latest_alpha = np.array([])
        self.shared_lock = threading.Lock()
        self.coherence_matrix = ([], np.array([[]]))

engine = Engine()
sono = SonoPlayer()

def compute_coherence_matrix(arrays):
    N = len(arrays)
    mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if len(arrays[i])<256 or len(arrays[j])<256: v=0.0
            else:
                c = np.corrcoef(arrays[i],arrays[j])[0,1]
                v = max(0.0,min(1.0,c))
            mat[i,j]=v
    return mat

@app.callback(
    [Output('spec-heat','figure'),
     Output('coh-matrix','figure'),
     Output('coh-bar','figure'),
     Output('status','children')],
    Input('interval','n_intervals')
)
def update_all(n):
    with engine.shared_lock:
        times = engine.latest_times.copy()
        spec = engine.latest_spec.copy()
        alpha = engine.latest_alpha.copy()
        names, mat = engine.coherence_matrix if hasattr(engine,'coherence_matrix') else ([],np.array([[]]))
        source = "LOCAL" if (engine.sensor.ser and engine.sensor.ser.is_open) else "INTERMAGNET_MULTI"

    # Spectrogram
    if spec.size>0 and len(times)>0:
        fig_spec=go.Figure(go.Heatmap(
            z=spec.T.tolist(),
            x=(times-times.max()).tolist(),
            y=FREQS,
            colorscale='Inferno'
        ))
        fig_spec.update_xaxes(autorange="reversed")
        fig_spec.update_layout(title=f"Spectrogram — {source}", xaxis_title="Seconds ago → Now")
    else: fig_spec=go.Figure()

    # Coherence matrix
    if len(names)>1 and mat.size:
        fig_mat=go.Figure(go.Heatmap(z=mat.tolist(), x=names, y=names, colorscale='Viridis', zmin=0, zmax=1))
        fig_mat.update_layout(title="Pairwise Coherence (10–13 Hz)")
        mean_coh = np.mean(mat,axis=0)
        fig_bar=go.Figure(go.Bar(x=names, y=mean_coh.tolist(), name="Mean coherence"))
        fig_bar.update_layout(title="Mean Coherence per Station")
    else:
        fig_mat=go.Figure()
        fig_bar=go.Figure()

    status=f"Source: {source} | Stations online: {len(names)} | Last update: {datetime.utcnow().strftime('%H:%M:%S')} UTC"

    return fig_spec, fig_mat, fig_bar, status

def background_loop():
    fallback_names,fallback_arrays=[],[]
    while True:
        try:
            if engine.sensor.ser and engine.sensor.ser.is_open:
                data = engine.sensor.get_buffer()
            else:
                fallback_names,fallback_arrays = fetch_intermagnet_multi()
                if len(fallback_arrays)>0:
                    # Weighted average by inverse variance
                    weights = [1/np.var(arr) if np.var(arr)>0 else 0 for arr in fallback_arrays]
                    data = np.average(fallback_arrays,axis=0,weights=weights)
                else:
                    data=np.array([])
            if len(data)<TARGET_FS*10:
                time.sleep(1)
                continue

            # Adaptive 11.6 Hz lock
            recent_116=0
            if hasattr(engine.dyad,'last_spec') and engine.dyad.last_spec.size:
                recent_116=np.mean(engine.dyad.last_spec[-20:,4])
            if recent_116>2e-6: filtered=bandpass_sos(data,11.3,11.9,order=8)*8
            elif recent_116>5e-7: filtered=bandpass_sos(data,10.8,12.4)*5
            else: filtered=bandpass_sos(data,10.0,13.5)*3

            # Faint broadband
            bb = sosfilt(butter(4,[0.5/(TARGET_FS/2),40/(TARGET_FS/2)],'band',output='sos'), data)
            signal = filtered + 0.15*bb

            times, spec, alpha_spec = engine.dyad.compute(signal)
            engine.latest_times=times
            engine.latest_spec=spec
            engine.latest_alpha=alpha_spec
            engine.dyad.last_spec=spec

            # Sonify 11.6 + alpha average
            amp_116 = spec[-1,4] if spec.size else 0
            amp_alpha = np.mean(alpha_spec[-1,:]) if alpha_spec.size else 0
            sono.play_amp(amp_116, amp_alpha)

            time.sleep(STEP_S)
        except Exception as e:
            print(f"Background loop error: {e}")
            time.sleep(1)

# =================== RUN ===================
if __name__=="__main__":
    threading.Thread(target=background_loop,daemon=True).start()
    app.run_server(debug=True)
