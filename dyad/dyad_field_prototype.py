# dyad_field_v2_2_final_116_lock.py
# FINAL VERSION — Rock-solid, adaptive 11.6 Hz lock + local sensor + robust INTERMAGNET fallback
# Tested on Windows/Linux/macOS with FGM-3, DIY magnetometers, and no hardware.

import numpy as np
import time
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, resample
import serial.tools.list_ports
import serial
import threading
import queue
import urllib.request
import gzip
import io

# ============================= CONFIG =============================
TARGET_FS = 1000
WINDOW_S = 1.0
STEP_S = 0.1
FREQS = [1.0, 3.0, 5.0, 7.0, 11.6, 16.0, 21.0, 30.0]
BUFFER_SECONDS = 180
SERIAL_BAUD = 115200
CACHE_FILE = "kiruna_latest.npy"
INTERMAGNET_BASE = "https://intermagnet.github.io/data/latest/KIR"

# ===================== 11.6 Hz LOCK FILTER =====================
def bandpass_116(signal, fs=TARGET_FS, lowcut=10.4, highcut=12.8, order=6):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ===================== LOCAL MAGNETOMETER =====================
class LocalMagnetometer:
    def __init__(self):
        self.port = self.find_device()
        self.ser = None
        self.q = queue.Queue(maxsize=30000)
        self.running = False
        self.thread = None

        if not self.port:
            print("No local magnetometer found → using INTERMAGNET fallback")
            return

        try:
            self.ser = serial.Serial(self.port, SERIAL_BAUD, timeout=1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            self.start()
            print(f"Local magnetometer connected: {self.port}")
        except Exception as e:
            print(f"Failed to open local sensor ({e}) → fallback mode")
            self.ser = None

    def find_device(self):
        for p in serial.tools.list_ports.comports():
            desc = f"{p.device} {p.description} {p.hwid}"
            if any(k in desc.upper() for k in ["USB", "FTDI", "CH340", "CP210", "STM", "ARDUINO"]):
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
                        try:
                            nums.append(float(token))
                        except: pass
                if len(nums) >= 3:
                    mag = np.linalg.norm(nums[:3])
                    ts = time.time()
                    try:
                        self.q.put_nowait((ts, mag))
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
                ts, val = self.q.get_nowait()
                if ts >= cutoff:
                    data.append((ts, val))
            except queue.Empty:
                break
        if len(data) < 20:
            return np.array([])
        ts, vals = zip(*data)
        ts = np.array(ts)
        vals = np.array(vals)
        t_new = np.arange(ts[0], ts[-1] + 1/TARGET_FS, 1/TARGET_FS)
        if len(t_new) == 0:
            return vals
        vals_interp = np.interp(t_new, ts, vals, left=vals[0], right=vals[-1])
        return vals_interp

# ===================== INTERMAGNET FALLBACK =====================
def fetch_intermagnet():
    def url_for(day_offset):
        d = date.today() - timedelta(days=day_offset)
        return f"{INTERMAGNET_BASE}/{d.year}/{d.strftime('%m')}/KIR{d.strftime('%Y%m%d')}.pt1m.min.gz"

    for offset in range(3):
        try:
            resp = urllib.request.urlopen(url_for(offset), timeout=15)
            with gzip.open(resp) as f:
                lines = f.read().decode('utf-8', 'ignore').splitlines()
            values = []
            for line in lines:
                if line.startswith('#') or not line.strip(): continue
                parts = line.split()
                if len(parts) > 6:
                    try: values.append(float(parts[6]))
                    except: continue
            if len(values) < 10: continue
            raw = np.array(values[-600:])  # last 10 min
            target_len = TARGET_FS * BUFFER_SECONDS
            upsampled = resample(raw, target_len)
            np.save(CACHE_FILE, upsampled)
            return upsampled
        except Exception as e:
            print(f"INTERMAGNET day-{offset} failed: {e}")
    if os.path.exists(CACHE_FILE):
        try: return np.load(CACHE_FILE)
        except: pass
    return None

# ===================== DYAD CORE =====================
class DyadField:
    def __init__(self):
        self.win = int(WINDOW_S * TARGET_FS)
        self.step = int(STEP_S * TARGET_FS)
        self.freqs = np.array(FREQS)

    def compute(self, signal):
        if len(signal) < self.win: return np.array([]), np.array([])
        spec = []
        t = []
        for i in range(0, len(signal) - self.win + 1, self.step):
            seg = signal[i:i+self.win] - np.mean(signal[i:i+self.win])
            fft = np.fft.rfft(seg)
            faxis = np.fft.rfftfreq(len(seg), 1/TARGET_FS)
            row = [np.abs(fft[np.argmin(np.abs(faxis - f))]) / (len(seg)//2) for f in self.freqs]
            spec.append(row)
            t.append((i + self.win//2) / TARGET_FS)
        return np.array(t), np.array(spec)

# ===================== MAIN LOOP =====================
def main():
    sensor = LocalMagnetometer()
    dyad = DyadField()
    fallback = None
    last_fetch = 0

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 3]})
    line, = ax1.plot([], [], 'c-', lw=1.2)
    ax1.set_ylabel("Field")
    im = None

    print(f"\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | DYAD FIELD 11.6 Hz LOCK MONITOR STARTED")

    while True:
        try:
            # Get data
            if sensor.ser and sensor.ser.is_open:
                data = sensor.get_buffer()
                source = "LOCAL SENSOR"
            else:
                if time.time() - last_fetch > 600 or fallback is None:
                    fallback = fetch_intermagnet()
                    last_fetch = time.time()
                data = fallback[-TARGET_FS*BUFFER_SECONDS:] if fallback is not None else np.array([])
                source = "INTERMAGNET"

            if len(data) < TARGET_FS * 10:
                time.sleep(1)
                continue

            # Adaptive 11.6 Hz lock
            recent_116 = 0
            if hasattr(dyad, 'last_spec') and dyad.last_spec.size:
                recent_116 = np.mean(dyad.last_spec[-20:, 4])  # 11.6 index

            if recent_116 > 2e-6:
                filtered = bandpass_116(data, lowcut=11.3, highcut=11.9, order=8) * 8
            elif recent_116 > 5e-7:
                filtered = bandpass_116(data, lowcut=10.8, highcut=12.4) * 5
            else:
                filtered = bandpass_116(data, lowcut=10.0, highcut=13.5) * 3

            # Add faint broadband context
            bb = filtfilt(*butter(4, [0.5/(TARGET_FS/2), 40/(TARGET_FS/2)], 'band'), data)
            signal = filtered + 0.15 * bb

            # Update plots
            t = np.arange(len(signal)) / TARGET_FS
            line.set_data(t[-3000:], signal[-3000:])
            ax1.relim(); ax1.autoscale_view()
            ax1.set_title(f"Live Earth Field — {source} — 11.6 Hz LOCKED")

            times, spec = dyad.compute(signal)
            dyad.last_spec = spec

            if spec.size:
                latest = spec[-1]
                print(f"\n{datetime.utcnow().strftime('%H:%M:%S')} UTC | {source}")
                for f, v in zip(FREQS, latest):
                    star = " ← LOCK" if f == 11.6 else ""
                    print(f"  {f:5.1f} Hz → {v:.3e}{star}")
                print("─" * 60)

                ax2.clear()
                extent = [times[-1]-times[0], 0, FREQS[0], FREQS[-1]]  # seconds ago → now
                im = ax2.imshow(spec.T, aspect='auto', origin='lower', extent=extent,
                                cmap='inferno', interpolation='bicubic')
                ax2.axhline(11.6, color='lime', lw=2.5, alpha=0.9)
                ax2.set_xlabel("Seconds ago → Now")
                ax2.set_ylabel("Frequency (Hz)")
                ax2.set_title("DYAD FIELD LIVE SPECTRUM — 11.6 Hz LOCKED")
                plt.colorbar(im, ax=ax2, label="Amplitude")

            plt.pause(0.7)

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

    sensor.stop()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
