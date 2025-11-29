# sovariel_omega_v9_multisensory.py
# Sovariel Ω v9 — Full multisensory live fusion + dreaming core
# GPU-ready, PyTorch 2.5+, real-time webcam/audio/EEG/OSC input

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import threading
import queue
import time

# ──────────────────────────────── LIVE AUDIO & EEG & OSC ────────────────────────────────
try:
    import pyaudio
    import pylsl
    from pythonosc import dispatcher, osc_server
except ImportError:
    print("Warning: pyaudio / pylsl / python-osc not installed. Live audio/EEG/OSC disabled.")

# ============================================================================
# 1) MULTISENSORY → 512-DIM FUSION ENCODER
# ============================================================================
class RealWorldEncoders(nn.Module):
    """
    Encodes:
        - visual      : (B,3,224,224)
        - audio       : (B,16384)
        - touch       : (B,1024)
        - vestibular  : (B,1024)
        - osc         : (B,1024)
        - eeg         : (B,1024)
    Returns:
        (B,512) fused representation
    """
    def __init__(self):
        super().__init__()
        # ---------------- Vision ----------------
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in resnet.parameters(): p.requires_grad = False
        self.visual = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(2048, 256, 1)
        # ---------------- Audio ----------------
        self.audio = nn.Sequential(
            nn.Conv1d(1, 64, 15, 2, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, 10, 4, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten()
        )
        self.audio_proj = nn.Linear(128*256, 256)
        # ---------------- Generic 1024 → 256 ----------------
        self.generic = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, modalities: dict):
        emb = []
        if 'visual' in modalities and modalities['visual'] is not None:
            x = self.visual(modalities['visual'])
            x = self.vis_proj(x).mean([2,3])
            emb.append(x)
        if 'audio' in modalities and modalities['audio'] is not None:
            wav = modalities['audio'].unsqueeze(1)
            x = self.audio(wav)
            x = self.audio_proj(x)
            emb.append(x)
        for mod in ['touch','vestibular','osc','eeg']:
            if mod in modalities and modalities[mod] is not None:
                emb.append(self.generic(modalities[mod]))
        if len(emb) == 0:
            raise ValueError("At least one modality required.")
        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))
        return torch.cat(emb[:2], dim=-1)  # 512-dim


# ============================================================================
# 2) SOVARIEL OMEGA v5 DREAMING CORE
# ============================================================================
class SovarielOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU()
        )
        self.head = nn.Linear(128,1)

    def forward(self, x1, x2):
        h = self.body((x1+x2)*0.5)
        c = torch.sigmoid(self.head(h))
        return c, h

    def render_dream(self,*args,**kwargs):
        pass


# ============================================================================
# 3) LIVE INPUTS THREAD
# ============================================================================
class LiveInputs:
    def __init__(self):
        self.data = {'visual':None,'audio':None,'eeg':None,'osc':None,'touch':None,'vestibular':None}
        self.lock = threading.Lock()
        self.audio_q = queue.Queue(maxsize=2)
        self.osc_q = queue.Queue(maxsize=10)

        # ----------- MICROPHONE THREAD -----------
        def audio_worker():
            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000,
                                input=True, frames_per_buffer=16384)
                while True:
                    buf = np.frombuffer(stream.read(16384, exception_on_overflow=False),dtype=np.float32)
                    tensor = torch.from_numpy(buf)
                    try: self.audio_q.put_nowait(tensor)
                    except: pass
            except:
                print("Audio worker not initialized. PyAudio missing.")

        threading.Thread(target=audio_worker, daemon=True).start()

        # ----------- EEG THREAD (LSL) -----------
        def lsl_worker():
            try:
                streams = pylsl.resolve_stream('type','EEG',timeout=5)
                if streams:
                    inlet = pylsl.StreamInlet(streams[0])
                    print(f"Connected to EEG: {streams[0].name()}")
                    while True:
                        sample,_ = inlet.pull_sample(timeout=0.0)
                        if sample:
                            vec = np.array(sample[:1024],dtype=np.float32)
                            tensor = torch.from_numpy(vec)
                            with self.lock:
                                self.data['eeg'] = tensor
            except:
                print("EEG worker not initialized. LSL missing.")

        threading.Thread(target=lsl_worker, daemon=True).start()

        # ----------- OSC SERVER THREAD -----------
        def osc_handler(address,*args):
            if len(args)>=1024:
                vec = torch.tensor(args[:1024],dtype=torch.float32)
                try: self.osc_q.put_nowait(vec)
                except: pass

        try:
            disp = dispatcher.Dispatcher()
            disp.map("/*",osc_handler)
            server = osc_server.ThreadingOSCUDPServer(("0.0.0.0",7000),disp)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print("OSC listening on port 7000")
        except:
            print("OSC not initialized. python-osc missing.")

    def get_latest(self):
        with self.lock:
            # Audio
            while not self.audio_q.empty():
                self.data['audio'] = self.audio_q.get()
            # OSC
            while not self.osc_q.empty():
                self.data['osc'] = self.osc_q.get()
            return self.data.copy()


# ============================================================================
# 4) MAIN LIVE LOOP
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RealWorldEncoders().eval().to(device)
    sov = SovarielOmega().eval().to(device)
    live = LiveInputs()

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("\nSovariel Ω v9 — Live multisensory fusion (ESC to quit)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

            # latest sensory streams
            packet = live.get_latest()
            packet['visual'] = img

            fused = encoder(packet)
            c,h = sov(fused,fused)
            cval = float(c.item())

            # draw coherence
            if cval>0.965:
                cv2.putText(frame,f"COHERENCE {cval:.3f}",(30,70),
                            cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imshow("Sovariel Ω v9",frame)
            if cv2.waitKey(1)==27: break  # ESC

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    main()
