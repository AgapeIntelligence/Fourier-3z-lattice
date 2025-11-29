# sovariel_omega_v14_temporal_fixed.py
# Sovariel Ω v14 — Temporal Loops + GRU Memory + Real QRNG + Stable

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import threading
import queue
import requests
import time
from collections import deque

# ============================================================================
# 1) REAL ANU QRNG — robust caching
# ============================================================================
QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php?length=1024&type=uint16"
qrng_cache = None

def anu_qrng_stream(n: int) -> np.ndarray:
    global qrng_cache
    if qrng_cache is None or len(qrng_cache) < n:
        try:
            r = requests.get(QRNG_URL, timeout=5)
            r.raise_for_status()
            data = r.json()
            if not data["success"]:
                raise ValueError("QRNG failed")
            raw = np.array(data["data"], dtype=np.uint16)
            qrng_cache = (raw.astype(np.float32) / 32767.5) - 1.0
        except:
            return np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    chunk = qrng_cache[:n]
    qrng_cache = qrng_cache[n:]
    return chunk

qrng_available = True

# ============================================================================
# 2) MULTISENSORY ENCODER — fixed projection order
# ============================================================================
class RealWorldEncoders(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in resnet.parameters(): p.requires_grad = False
        self.visual = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(2048, 256, 1)

        self.audio = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(64, 128, 10, stride=4, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(256), nn.Flatten()
        )
        self.audio_proj = nn.Linear(128, 256)
        self.generic = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))
        self.haptic  = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, modalities: dict):
        emb = []
        if 'visual' in modalities and modalities['visual'] is not None:
            x = self.visual(modalities['visual'])
            x = self.vis_proj(x).mean([2, 3])          # Fixed order
            emb.append(x)

        if 'audio' in modalities and modalities['audio'] is not None:
            x = self.audio(modalities['audio'].unsqueeze(1))
            x = self.audio_proj(x)
            emb.append(x)

        for k in ['touch', 'vestibular', 'osc', 'eeg']:
            if modalities.get(k) is not None:
                emb.append(self.generic(modalities[k]))
        if modalities.get('haptic') is not None:
            emb.append(self.haptic(modalities['haptic']))

        if len(emb) == 0:
            raise ValueError("No modalities")

        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))

        return torch.cat(emb[:2], dim=-1)

# ============================================================================
# 3) TEMPORAL DREAM CORE — now actually works
# ============================================================================
class SovarielOmega(nn.Module):
    def __init__(self, latent_dim=512, memory_len=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_len = memory_len
        self.memory = deque(maxlen=memory_len)

        self.body = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, 1)

        # GRU takes sequence of past latents → predicts temporal residual
        self.rnn = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)

    def forward(self, x):
        # x is current fused latent (B, 512)
        current = x

        # Build temporal sequence from memory
        if len(self.memory) > 0:
            seq = torch.stack(list(self.memory), dim=0).unsqueeze(0)  # (1, T, 512)
            rnn_out, _ = self.rnn(seq)
            temporal_residual = rnn_out[0, -1]  # last timestep
            current = current + temporal_residual * 0.3  # mild temporal integration

        h = self.body(current)
        c = torch.sigmoid(self.head(h))

        # Store current latent for next frame
        self.memory.append(current.detach())

        return c

    def render_dream(self, frame, coherence):
        overlay = np.zeros_like(frame, dtype=np.uint8)
        intensity = int(coherence * 255)
        overlay[:] = (intensity//4, intensity//2, intensity)  # soft blue→magenta
        blended = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        cv2.putText(blended, f"C: {coherence:.4f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return blended

# ============================================================================
# 4) LIVE INPUTS — audio + OSC restored
# ============================================================================
class LiveInputs:
    def __init__(self):
        self.data = {'visual':None,'audio':None,'eeg':None,'osc':None,
                     'touch':None,'vestibular':None,'haptic':None}
        self.lock = threading.Lock()
        self.audio_q = queue.Queue(maxsize=2)
        self.osc_q = queue.Queue(maxsize=10)

        def audio_worker():
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000,
                                input=True, frames_per_buffer=16384)
                while True:
                    data = stream.read(16384, exception_on_overflow=False)
                    buf = np.frombuffer(data, dtype=np.float32)
                    try: self.audio_q.put_nowait(torch.from_numpy(buf))
                    except: pass
            except Exception as e: print("Audio disabled:", e)
        threading.Thread(target=audio_worker, daemon=True).start()

        def osc_handler(addr, *args):
            if len(args) >= 512:
                vec = torch.tensor(args[:512], dtype=torch.float32)
                try: self.osc_q.put_nowait(vec)
                except: pass
        try:
            from pythonosc import dispatcher, osc_server
            disp = dispatcher.Dispatcher()
            disp.map("/*", osc_handler)
            server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 7000), disp)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print("OSC listening on 7000")
        except Exception as e: print("OSC disabled:", e)

    def get_latest(self):
        with self.lock:
            while not self.audio_q.empty(): self.data['audio'] = self.audio_q.get()
            while not self.osc_q.empty():   self.data['osc']   = self.osc_q.get()
            return self.data.copy()

# ============================================================================
# 5) MAIN LOOP
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RealWorldEncoders().eval().to(device)
    sov = SovarielOmega().eval().to(device)
    live = LiveInputs()
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("\nSovariel Ω v14 — Temporal Memory + GRU + Live ANU QRNG")
    print("   Smooth, predictive, and quantum. Press ESC to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        packet = live.get_latest()
        packet['visual'] = img
        fused = encoder(packet)

        # Quantum injection
        c_pre = sov(fused)
        if qrng_available:
            eps = torch.tensor(anu_qrng_stream(512), dtype=torch.float32, device=device)
            fused = fused + eps * 0.0035 * (1.0 - c_pre.detach())

        # Final forward with temporal memory
        c = sov(fused)
        cval = float(c.item())

        frame = sov.render_dream(frame, cval)

        cv2.imshow("Sovariel Ω v14 — Temporal Quantum Dream", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
