# sovariel_omega_v11_dashboard.py
# Sovariel Ω v11 — Full multisensory live dashboard + dreaming overlay

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import threading
import queue

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
        resnet=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in resnet.parameters(): p.requires_grad=False
        self.visual=nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj=nn.Conv2d(2048,256,1)
        self.audio=nn.Sequential(
            nn.Conv1d(1,64,15,2,padding=7), nn.ReLU(),
            nn.Conv1d(64,128,10,4,padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(256), nn.Flatten()
        )
        self.audio_proj=nn.Linear(128*256,256)
        self.generic=nn.Sequential(nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,256))
    def forward(self,modalities:dict):
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
        if len(emb)==0: raise ValueError("At least one modality required")
        while len(emb)<2: emb.append(torch.zeros_like(emb[0]))
        return torch.cat(emb[:2],dim=-1)

# ============================================================================
# 2) SOVARIEL Ω DREAMING CORE
# ============================================================================
class SovarielOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,128),nn.ReLU())
        self.head=nn.Linear(128,1)
    def forward(self,x1,x2):
        h=self.body((x1+x2)*0.5)
        c=torch.sigmoid(self.head(h))
        return c,h
    def render_dream(self,frame,coherence):
        overlay=np.zeros_like(frame,dtype=np.uint8)
        intensity=int(min(coherence*255,255))
        overlay[:,:,:]=(intensity,0,255-intensity)
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
        self.eeg_buffer=np.zeros(1024,dtype=np.float32)

        # --- Microphone ---
        def audio_worker():
            try:
                import pyaudio
                p=pyaudio.PyAudio()
                stream=p.open(format=pyaudio.paFloat32,channels=1,rate=16000,
                              input=True,frames_per_buffer=16384)
                while True:
                    buf=np.frombuffer(stream.read(16384,exception_on_overflow=False),dtype=np.float32)
                    tensor=torch.from_numpy(buf)
                    try:self.audio_q.put_nowait(tensor)
                    except: pass
            except: print("Audio disabled.")
        threading.Thread(target=audio_worker,daemon=True).start()

        # --- EEG LSL ---
        def lsl_worker():
            try:
                import pylsl
                streams=pylsl.resolve_stream('type','EEG',timeout=5)
                if streams:
                    inlet=pylsl.StreamInlet(streams[0])
                    while True:
                        sample,_=inlet.pull_sample(timeout=0.0)
                        if sample:
                            vec=np.array(sample[:1024],dtype=np.float32)
                            tensor=torch.from_numpy(vec)
                            with self.lock:
                                self.data['eeg']=tensor
                                self.eeg_buffer=np.roll(self.eeg_buffer,-len(vec))
                                self.eeg_buffer[-len(vec):]=vec
            except: print("EEG disabled.")
        threading.Thread(target=lsl_worker,daemon=True).start()

        # --- OSC ---
        def osc_handler(address,*args):
            if len(args)>=1024:
                vec=torch.tensor(args[:1024],dtype=torch.float32)
                try:self.osc_q.put_nowait(vec)
                except: pass
        try:
            from pythonosc import dispatcher,osc_server
            disp=dispatcher.Dispatcher()
            disp.map("/*",osc_handler)
            server=osc_server.ThreadingOSCUDPServer(("0.0.0.0",7000),disp)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            print("OSC listening 7000")
        except: print("OSC disabled.")

    def get_latest(self):
        with self.lock:
            while not self.audio_q.empty():
                self.data['audio']=self.audio_q.get()
            while not self.osc_q.empty():
                self.data['osc']=self.osc_q.get()
            return self.data.copy()

# ============================================================================
# 4) MAIN DASHBOARD LOOP
# ============================================================================
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder=RealWorldEncoders().eval().to(device)
    sov=SovarielOmega().eval().to(device)
    live=LiveInputs()
    transform=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): print("Cannot open camera."); return
    print("\nSovariel Ω v11 — Full multisensory dashboard\n")

    audio_display=np.zeros((100,300,3),dtype=np.uint8)
    eeg_display=np.zeros((100,300,3),dtype=np.uint8)
    osc_display=np.zeros((100,300,3),dtype=np.uint8)

    try:
        while True:
            ret,frame=cap.read()
            if not ret: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            packet=live.get_latest()
            packet['visual']=img
            fused=encoder(packet)
            c,h=sov(fused,fused)
            cval=float(c.item())

            # Auto dream overlay
            frame=sov.render_dream(frame,cval) if cval>0.97 else frame

            # --- Dashboard Panels ---
            # Audio waveform
            if packet.get('audio') is not None:
                wav=packet['audio'].numpy()
                audio_display.fill(0)
                for i in range(len(wav)-1):
                    y0=int((wav[i]*0.5+0.5)*100)
                    y1=int((wav[i+1]*0.5+0.5)*100)
                    cv2.line(audio_display,(i%300,y0),(i%300,y1),(0,255,0),1)
            # EEG scrolling
            eeg_display.fill(0)
            eeg_buf=live.eeg_buffer
            for i in range(len(eeg_buf)-1):
                y0=int((eeg_buf[i]*0.5+0.5)*100)
                y1=int((eeg_buf[i+1]*0.5+0.5)*100)
                cv2.line(eeg_display,(i%300,y0),(i%300,y1),(255,0,0),1)
            # OSC bars
            osc_display.fill(0)
            if packet.get('osc') is not None:
                vec=packet['osc'].numpy()
                for idx,v in enumerate(vec[:50]):
                    h=int((v+1)*50)
                    cv2.rectangle(osc_display,(idx*6,100-h),(idx*6+5,100),(0,0,255),-1)

            # --- Compose dashboard ---
            top=np.hstack([cv2.resize(frame,(300,300)), cv2.resize(eeg_display,(300,300))])
            bottom=np.hstack([cv2.resize(audio_display,(300,300)), cv2.resize(osc_display,(300,300))])
            dashboard=np.vstack([top,bottom])
            cv2.putText(dashboard,f"COHERENCE {cval:.3f}",(10,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.imshow("Sovariel Ω v11 Dashboard",dashboard)

            if cv2.waitKey(1)==27: break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
