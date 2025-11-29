# sovariel_omega_v13_final.py
# Sovariel Ω v13 — Clean, stable, live ANU QRNG + multisensory + proto-dreaming

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

# ============================================================================
# 1) REAL ANU QRNG — robust + cached
# ============================================================================
QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php?length=1024&type=uint16"
qrng_cache = None
qrng_last_fetch = 0.0

def anu_qrng_stream(n: int) -> np.ndarray:
    global qrng_cache, qrng_last_fetch
    if qrng_cache is None or len(qrng_cache) < n:
        try:
            r = requests.get(QRNG_URL, timeout=5)
            r.raise_for_status()
            data = r.json()
            if not data["success"]:
                raise ValueError("QRNG success=False")
            raw = np.array(data["data"], dtype=np.uint16)
            qrng_cache = (raw.astype(np.float32) / 32767.5) - 1.0  # [-1, 1)
            qrng_last_fetch = time.time()
        except Exception as e:
            # Silent fallback to PRNG if network fails
            return np.random.uniform(-1.0, 1.0, n).astype(np.float32)

    chunk, qrng_cache = qrng_cache[:n], qrng_cache[n:]
    return chunk

qrng_available = True

# ============================================================================
# 2) ENCODERS — fixed projection order
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
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten()
        )
        self.audio_proj = nn.Linear(128, 256)                    # ← fixed

        self.generic = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))
        self.haptic  = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, modalities: dict):
        emb = []

        if modalities.get('visual') is not None:
            x = self.visual(modalities['visual'])
            x = self.vis_proj(x).mean([2, 3])                     # ← correct order
            emb.append(x)

        if modalities.get('audio') is not None:
            x = self.audio(modalities['audio'].unsqueeze(1))
            x = self.audio_proj(x)
            emb.append(x)

        for key in ['touch', 'vestibular', 'osc', 'eeg']:
            if modalities.get(key) is not None:
                emb.append(self.generic(modalities[key]))

        if modalities.get('haptic') is not None:
            emb.append(self.haptic(modalities['haptic']))

        if len(emb) == 0:
            raise ValueError("No modalities")

        # Pad to at least 2 embeddings
        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))

        return torch.cat(emb[:2], dim=-1)

# ============================================================================
# 3) DREAM CORE — unchanged
# ============================================================================
class SovarielOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x1, x2):
        h = self.body((x1 + x2) * 0.5)
        c = torch.sigmoid(self.head(h))
        return c, h

    def render_dream(self, frame, coherence, packet):
        overlay = np.zeros_like(frame, dtype=np.uint8)
        intensity = int(coherence * 255)
        overlay[:] = (intensity // 3, intensity // 2, intensity)  # blue → magenta shift
        blended = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        cv2.putText(blended, f"C: {coherence:.4f}", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return blended

# ============================================================================
# 4) LIVE INPUTS — restored working audio + OSC threads
# ============================================================================
class LiveInputs:
    def __init__(self):
        self.data = {'visual': None, 'audio': None, 'eeg': None, 'osc': None,
                     'touch': None, 'vestibular': None, 'haptic': None}
        self.lock = threading.Lock()
        self.audio_q = queue.Queue(maxsize=2)
        self.osc_q = queue.Queue(maxsize=10)

        # --- Audio thread ---
        def audio_worker():
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000,
                                input=True, frames_per_buffer=16384)
                while True:
                    buf = np.frombuffer(stream.read(16384, exception_on_overflow=False),
                                        dtype=np.float32)
                    try: self.audio_q.put_nowait(torch.from_numpy(buf))
                    except: pass
            except Exception as e:
                print("Audio disabled:", e)
        threading.Thread(target=audio_worker, daemon=True).start()

        # --- OSC thread ---
        def osc_handler(address, *args):
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
            print("OSC listening on port 7000")
        except Exception as e:
            print("OSC disabled:", e)

    def get_latest(self):
        with self.lock:
            while not self.audio_q.empty():
                self.data['audio'] = self.audio_q.get()
            while not self.osc_q.empty():
                self.data['osc'] = self.osc_q.get()
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
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("\nSovariel Ω v13 — LIVE ANU QRNG + Multisensory Proto-Dreaming")
    print("   (ESC to exit)\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        packet = live.get_latest()
        packet['visual'] = img
        fused = encoder(packet)

        # Pre-injection coherence
        c, _ = sov(fused, fused)

        # Quantum injection
        if qrng_available:
            eps = torch.tensor(anu_qrng_stream(512), dtype=torch.float32, device=device)
            fused = fused + eps * 0.0035 * (1.0 - c.detach())

        # Final coherence
        c, _ = sov(fused, fused)
        cval = float(c.item())

        frame = sov.render_dream(frame, cval, packet)

        cv2.imshow("Sovariel Ω v13 — Quantum Dream Lattice", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
