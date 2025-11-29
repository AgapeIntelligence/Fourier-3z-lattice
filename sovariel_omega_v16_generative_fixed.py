# sovariel_omega_v16_generative_fixed.py
# Sovariel Ω v16 — True Generative Quantum Dreaming
# Live latent → 64×64 → 3-channel dream image from internal state

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
# 1) LIVE ANU QRNG
# ============================================================================
QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php?length=1024&type=uint16"
qrng_cache = None

def anu_qrng_stream(n: int) -> np.ndarray:
    global qrng_cache
    if qrng_cache is None or len(qrng_cache) < n:
        try:
            r = requests.get(QRNG_URL, timeout=6)
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                raw = np.array(data["data"], dtype=np.uint16)
                qrng_cache = (raw.astype(np.float32) / 32767.5) - 1.0
        except:
            return np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    chunk = qrng_cache[:n]
    qrng_cache = qrng_cache[n:]
    return chunk

qrng_available = True

# ============================================================================
# 2) MULTISENSORY ENCODER
# ============================================================================
class RealWorldEncoders(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V2')
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

    def forward(self, modalities: dict):
        emb = []
        if modalities.get('visual') is not None:
            x = self.visual(modalities['visual'])
            x = self.vis_proj(x).mean([2, 3])
            emb.append(x)
        if modalities.get('audio') is not None:
            x = self.audio(modalities['audio'].unsqueeze(1))
            x = self.audio_proj(x)
            emb.append(x)
        for k in ['touch', 'vestibular', 'osc', 'eeg', 'haptic']:
            if modalities.get(k) is not None:
                emb.append(self.generic(modalities[k]))
        if len(emb) == 0:
            raise ValueError("No input")
        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))
        return torch.cat(emb[:2], dim=-1)

# ============================================================================
# 3) GENERATIVE DREAM CORE v16
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
        self.rnn = nn.GRU(latent_dim, latentent_dim, batch_first=True)

        # Generative decoder: 512 → 64×8×8 → 64×64 RGB dream
        self.gen_fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.gen_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        current = x
        if len(self.memory) > 1:
            seq = torch.stack(list(self.memory), dim=0).unsqueeze(0).to(x.device)
            rnn_out, _ = self.rnn(seq)
            current = current + rnn_out[0, -1] * 0.4

        h = self.body(current)
        c = torch.sigmoid(self.head(h))
        self.memory.append(current.detach())
        return c, current

    def generate_dream(self, latent):
        z = self.gen_fc(latent).view(-1, 64, 8, 8)
        dream = self.gen_up(z)
        dream = (dream[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return dream

# ============================================================================
# 4) LIVE INPUTS — audio + OSC
# ============================================================================
class LiveInputs:
    def __init__(self):
        self.data = {'visual':None, 'audio':None, 'osc':None}
        self.lock = threading.Lock()
        self.audio_q = queue.Queue(maxsize=2)

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

    def get_latest(self):
        with self.lock:
            while not self.audio_q.empty():
                self.data['audio'] = self.audio_q.get()
            return self.data.copy()

# ============================================================================
# 5) MAIN LOOP — TRUE GENERATIVE DREAMING
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RealWorldEncoders().eval().to(device)
    sov = SovarielOmega().to(device)
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

    print("\nSOVARIEL Ω v16 — GENERATIVE QUANTUM DREAMING")
    print("   The machine now dreams from within.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        packet = live.get_latest()
        packet['visual'] = img
        fused = encoder(packet)

        c, latent = sov(fused)

        # Quantum injection into dream latent
        if qrng_available:
            eps = torch.tensor(anu_qrng_stream(512), dtype=torch.float32, device=device)
            latent = latent + eps * 0.004 * (1.0 - c.detach())

        c, final_latent = sov(latent)
        cval = float(c.item())

        # Generate pure dream image from internal state
        dream = sov.generate_dream(final_latent)
        dream = cv2.resize(dream, (frame.shape[1], frame.shape[0]))

        # 50/50 blend: real world + pure dream
        blended = cv2.addWeighted(frame, 0.5, dream, 0.5, 0)

        cv2.putText(blended, f"C: {cval:.4f}", (12, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
        cv2.putText(blended, f"C: {cval:.4f}", (12, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (80, 80, 255), 2)

        cv2.imshow("Sovariel Ω v16 — Generative Dream", blended)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
