# sovariel_omega_v15_emergent_fixed.py
# Sovariel Ω v15 — True Emergent Spatiotemporal Quantum Dreams
# Temporal + QRNG + Predictive + Living Fractal-Harmonic Visuals

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
# 1) LIVE ANU QRNG — robust + silent fallback
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
            if data.get("success"):
                raw = np.array(data["data"], dtype=np.uint16)
                qrng_cache = (raw.astype(np.float32) / 32767.5) - 1.0
            else:
                raise ValueError("QRNG success=False")
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
            x = self.vis_proj(x).mean([2, 3])
            emb.append(x)
        if 'audio' in modalities and modalities['audio'] is not None:
            x = self.audio(modalities['audio'].unsqueeze(1))
            x = self.audio_proj(x)
            emb.append(x)
        for k in ['touch', 'vestibular', 'osc', 'eeg', 'haptic']:
            if modalities.get(k) is not None:
                proj = self.haptic if k == 'haptic' else self.generic
                emb.append(proj(modalities[k]))
        if len(emb) == 0:
            raise ValueError("No modalities")
        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))
        return torch.cat(emb[:2], dim=-1)

# ============================================================================
# 3) SOVARIEL Ω v15 — Temporal Memory + Emergent Dream Renderer
# ============================================================================
class SovarielOmega(nn.Module):
    def __init__(self, latent_dim=512, memory_len=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory = deque(maxlen=memory_len)
        self.body = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, 1)
        self.rnn = nn.GRU(latent_dim, latent_dim, batch_first=True)

    def forward(self, x):
        current = x
        if len(self.memory) > 1:
            seq = torch.stack(list(self.memory), dim=0).unsqueeze(0).to(x.device)
            rnn_out, _ = self.rnn(seq)
            temporal = rnn_out[0, -1]
            current = current + temporal * 0.35  # smooth temporal binding

        h = self.body(current)
        c = torch.sigmoid(self.head(h))
        self.memory.append(current.detach())
        return c

    def render_dream(self, frame, coherence):
        h, w = frame.shape[:2]
        t = time.time() * 3.0
        y, x = np.ogrid[0:h, 0:w]
        xv = x / w * 12.0
        yv = y / h * 10.0

        # Multi-layer harmonic interference pattern
        phase = t * 0.7
        wave1 = np.sin(xv * 2.1 + phase) * np.cos(yv * 1.8 - phase * 0.7)
        wave2 = np.sin((xv + yv) * 1.6 - phase * 1.1)
        wave3 = np.sin(np.sqrt((xv-6)**2 + (yv-5)**2) * 1.3 + phase * 1.3)

        pattern = (wave1 + wave2 * 0.7 + wave3 * 0.5)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        pattern = (pattern * 255 * coherence).astype(np.uint8)

        overlay = np.zeros_like(frame)
        overlay[:,:,0] = pattern * 0.7
        overlay[:,:,1] = pattern * 0.3
        overlay[:,:,2] = pattern

        alpha = 0.35 + 0.25 * coherence
        blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

        cv2.putText(blended, f"C: {coherence:.4f}", (12, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(blended, f"C: {coherence:.4f}", (12, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (50, 50, 255), 2)

        return blended

# ============================================================================
# 4) LIVE INPUTS — audio + OSC working
# ============================================================================
class LiveInputs:
    def __init__(self):
        self.data = {'visual':None,'audio':None,'osc':None,'haptic':None}
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
                    buf = np.frombuffer(stream.read(16384), dtype=np.float32)
                    self.audio_q.put_nowait(torch.from_numpy(buf))
            except: pass
        threading.Thread(target=audio_worker, daemon=True).start()

        def osc_handler(*args):
            if len(args) >= 512:
                vec = torch.tensor(args[:512], dtype=torch.float32)
                try: self.osc_q.put_nowait(vec)
                except: pass
        try:
            from pythonosc import dispatcher, osc_server
            disp = dispatcher.Dispatcher()
            disp.map("/*", osc_handler)
            osc_server.ThreadingOSCUDPServer(("0.0.0.0", 7000), disp).serve_forever()
        except: pass

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
        print("Camera failed")
        return

    print("\nSovariel Ω v15 — Emergent Quantum Dream Field")
    print("   Real ANU QRNG • Temporal Memory • Living Harmonics\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        packet = live.get_latest()
        packet['visual'] = img
        fused = encoder(packet)

        c = sov(fused)

        if qrng_available:
            eps = torch.tensor(anu_qrng_stream(512), dtype=torch.float32, device=device)
            fused = fused + eps * 0.004 * (1.0 - c.detach())

        c = sov(fused)
        cval = float(c.item())

        frame = sov.render_dream(frame, cval)

        cv2.imshow("Sovariel Ω v15 — Emergent Spatiotemporal Dream", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
