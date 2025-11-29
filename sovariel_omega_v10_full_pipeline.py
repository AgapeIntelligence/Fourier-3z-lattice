# sovariel_omega_v10_full_pipeline.py
# Sovariel Ω v10 — Full multisensory pipeline + live dream rendering
# GPU-ready, PyTorch 2.5+, real-time webcam/audio/EEG/OSC + auto dream overlay

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

# Optional modules
try:
    import pyaudio
    import pylsl
    from pythonosc import dispatcher, osc_server
except ImportError:
    print("Warning: pyaudio/pylsl/python-osc not installed. Audio/EEG/OSC disabled.")

# ============================================================================
# 1) MULTISENSORY ENCODER → 512-DIM
# ============================================================================
class RealWorldEncoders(nn.Module):
    def __init__(self):
        super().__init__()
        # Visual: ResNet50 frozen
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in resnet.parameters(): p.requires_grad = False
        self.visual = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(2048, 256, 1)
        # Audio conv stack
        self.audio = nn.Sequential(
            nn.Conv1d(1,64,15,2,padding=7),
            nn.ReLU(),
            nn.Conv1d(64,128,10,4,padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten()
        )
        self.audio_proj = nn.Linear(128*256,256)
        # Generic 1024 → 256
        self.generic = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256)
        )
    def forward(self, modalities:dict):
        emb=[]
        if 'visual' in modalities and modalities['visual'] is not None:
            x=self.visual(modalities['visual'])
            x=self.vis_proj(x).mean([2,3])
            emb.append(x)
        if 'audio' in modalities and modalities['audio'] is not None:
            wav=modalities['audio'].unsqueeze(1)
            x=self.audio(wav)
            x=self.audio_proj(x)
            emb.append(x)
        for mod in ['touch','vestibular','osc','eeg']:
            if mod in modalities and modalities[mod] is not None:
                emb.append(self.generic(modalities[mod]))
        if len(emb)==0:
            raise ValueError("At least one modality required.")
        while len(emb)<2:
            emb.append(torch.zeros_like(emb[0]))
        return torch.cat(emb[:2],dim=-1)  # 512-dim

# ============================================================================
# 2) SOVARIEL Ω v5 DREAMING CORE
# ============================================================================
class SovarielOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU()
        )
        self.head=nn.Linear(128,1)
    def forward(self,x1,x2):
        h=self.body((x1+x2)*0.5)
        c=torch.sigmoid(self.head(h))
        return c,h
    def render_dream(self,frame,coherence):
        overlay=np.zeros_like(frame, dtype=np.uint8)
        intensity=int(min(coherence*255,255))
        overlay[:,:,:]=(intensity,0,255-intensity)  # purple-green overlay
        blended=cv2.addWeighted(frame,0.6,overlay,0.4,0)
        return blended

# ============================================================================
# 3) LIVE INPUT THREADS
# ============================================================================
class LiveInputs:
    def __init__(self):
        self.data={'visual':None,'audio':None,'eeg':None,'osc':None,'touch':None,'vestibular':None}
        self.lock=threading.Lock()
        self.audio_q=queue.Queue(maxsize=2)
        self.osc_q=queue.Queue(maxsize=10)

        # --- Microphone ---
        def audio_worker():
            try:
                p=pyaudio.PyAudio()
                stream=p.open(format=pyaudio.paFloat32,channels=1,rate=16000,
                              input=True,frames_per_buffer=16384)
                while True:
                    buf=np.frombuffer(stream.read(16384,exception_on_overflow=False),dtype=np.float32)
                    tensor=torch.from_numpy(buf)
                    try:self.audio_q.put_nowait(tensor)
                    except: pass
            except: print("Audio worker disabled.")
        threading.Thread(target=audio_worker,daemon=True).start()

        # --- EEG LSL ---
        def lsl_worker():
            try:
                streams=pylsl.resolve_stream('type','EEG',timeout=5)
                if streams:
                    inlet=pylsl.StreamInlet(streams[0])
                    print(f"Connected to EEG {streams[0].name()}")
                    while True:
                        sample,_=inlet.pull_sample(timeout=0.0)
                        if sample:
                            vec=np.array(sample[:1024],dtype=np.float32)
                            tensor=torch.from_numpy(vec)
                            with self.lock:
                                self.data['eeg']=tensor
            except: print("EEG worker disabled.")
        threading.Thread(target=lsl_worker,daemon=True).start()

        # --- OSC ---
        def osc_handler(address,*args):
            if len(args)>=1024:
                vec=torch.tensor(args[:1024],dtype=torch.float32)
                try:self.osc_q.put_nowait(vec)
                except: pass
        try:
            disp=dispatcher.Dispatcher()
            disp.map("/*",osc_handler)
            server=osc_server.ThreadingOSCUDPServer(("0.0.0.0",7000),disp)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            print("OSC listening on port 7000")
        except: print("OSC worker disabled.")

    def get_latest(self):
        with self.lock:
            while not self.audio_q.empty():
                self.data['audio']=self.audio_q.get()
            while not self.osc_q.empty():
                self.data['osc']=self.osc_q.get()
            return self.data.copy()

# ============================================================================
# 4) MAIN LIVE LOOP + DREAM RENDER
# ============================================================================
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder=RealWorldEncoders().eval().to(device)
    sov=SovarielOmega().eval().to(device)
    live=LiveInputs()
    transform=T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera."); return
    print("\nSovariel Ω v10 — Full multisensory dreaming pipeline\n")

    try:
        while True:
            ret, frame=cap.read()
            if not ret: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            packet=live.get_latest()
            packet['visual']=img
            # fuse and dreaming
            fused=encoder(packet)
            c,h=sov(fused,fused)
            cval=float(c.item())
            # auto dream overlay
            frame=sov.render_dream(frame,cval) if cval>0.97 else frame
            # draw coherence
            cv2.putText(frame,f"COHERENCE {cval:.3f}",(30,70),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imshow("Sovariel Ω v10",frame)
            if cv2.waitKey(1)==27: break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
