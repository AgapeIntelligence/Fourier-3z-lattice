# sovariel_omega_v7_production.py
# Sovariel Omega — v7 Production
# Features:
#  - Multisensory fusion (visual/audio/eeg/osc/generic)
#  - Stateful v5 core with temporal memory
#  - Coherence-weighted episodic replay
#  - Probabilistic resonance attractors (centroids + sampling)
#  - Dream engine that renders latent -> image & closed-loop re-injection
#  - Live inputs: webcam, optional pyaudio, pylsl (EEG), python-osc
#
# NOTE: This is intended as a production prototype. Tune hyperparams for your hardware.

import time
import math
import threading
import queue
import collections
import sys
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2

# Optional live deps (import-guards)
try:
    import pyaudio
    import pylsl
    from pythonosc import dispatcher, osc_server, udp_client
except Exception as e:
    pyaudio = None
    pylsl = None
    dispatcher = None
    osc_server = None
    udp_client = None
    # Only warn; system remains usable without these.
    print("Warning: some live-input packages are missing:", e, file=sys.stderr)


# -------------------------
# Utility helpers
# -------------------------
def now_str():
    return datetime.utcnow().isoformat() + "Z"


def ensure_tensor(x, device):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return x.to(device)


# -------------------------
# 1) REAL-WORLD ENCODERS (multisensory -> 512)
# -------------------------
class RealWorldEncoders(nn.Module):
    """
    Accepts a dict of modalities and returns a (B,512) fused vector.
    Supported modalities:
      - visual: (B,3,224,224)
      - audio:  (B,L)  (float32 waveform)
      - eeg:    (B,1024)
      - osc:    (B,1024)
      - generic: any 1024-d vector (touch/imu/proprio)
    """

    def __init__(self, freeze_backbone=True):
        super().__init__()

        # Vision backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in resnet.parameters():
                p.requires_grad = False
        self.visual = nn.Sequential(*list(resnet.children())[:-2])  # (B,2048,7,7)
        self.vis_proj = nn.Conv2d(2048, 256, kernel_size=1)

        # Audio conv front-end (flexible length)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=11, stride=2, padding=5),
            nn.ReLU()
        )
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_proj = nn.Linear(256 * 1, 256)

        # Generic projectors for EEG/OSC/touch (expect 1024)
        self.generic = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, inputs):
        # inputs: dict possibly containing visual/audio/eeg/osc/generic
        emb = []
        device = next(self.parameters()).device

        # Visual
        if inputs.get("visual") is not None:
            vis = inputs["visual"]
            if vis.dim() == 3:
                vis = vis.unsqueeze(0)
            vis = vis.to(device)
            x = self.visual(vis)  # (B,2048,7,7)
            x = self.vis_proj(x).mean((2, 3))  # (B,256)
            emb.append(x)

        # Audio
        if inputs.get("audio") is not None:
            wav = inputs["audio"]
            # accept numpy or tensor, pad/truncate to target length
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if isinstance(wav, torch.Tensor):
                wav = wav.flatten()
            else:
                wav = torch.tensor(wav, dtype=torch.float32)
            target_len = 16384  # ~1s @16kHz
            if wav.numel() < target_len:
                pad = torch.zeros(target_len - wav.numel(), dtype=torch.float32)
                wav = torch.cat([wav, pad], dim=0)
            elif wav.numel() > target_len:
                wav = wav[:target_len]
            wav = wav.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,L)
            x = self.audio_conv(wav)  # (1,256,T')
            x = self.audio_pool(x).view(1, -1)  # (1,256)
            x = self.audio_proj(x)  # (1,256)
            emb.append(x)

        # EEG
        if inputs.get("eeg") is not None:
            eeg = inputs["eeg"]
            if isinstance(eeg, np.ndarray):
                eeg = torch.from_numpy(eeg)
            if eeg.dim() == 1:
                eeg = eeg.unsqueeze(0)
            eeg = eeg.to(device).float()
            if eeg.shape[-1] < 1024:
                pad = torch.zeros((eeg.shape[0], 1024 - eeg.shape[-1]), device=device)
                eeg = torch.cat([eeg, pad], dim=-1)
            elif eeg.shape[-1] > 1024:
                eeg = eeg[:, :1024]
            emb.append(self.generic(eeg))

        # OSC / generic vectors
        if inputs.get("osc") is not None:
            osc = inputs["osc"]
            if isinstance(osc, np.ndarray):
                osc = torch.from_numpy(osc)
            if osc.dim() == 1:
                osc = osc.unsqueeze(0)
            osc = osc.to(device).float()
            if osc.shape[-1] < 1024:
                pad = torch.zeros((osc.shape[0], 1024 - osc.shape[-1]), device=device)
                osc = torch.cat([osc, pad], dim=-1)
            elif osc.shape[-1] > 1024:
                osc = osc[:, :1024]
            emb.append(self.generic(osc))

        # If no modalities present -> error
        if len(emb) == 0:
            raise ValueError("No modalities present. Provide at least visual, audio, eeg or osc.")

        # pad to at least two embeddings to keep 512-dim contract
        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))

        out = torch.cat(emb[:2], dim=-1)  # (B,512)
        return out


# -------------------------
# 2) SOVARIEL OMEGA v5 CORE (STATEFUL) + MEMORY
# -------------------------
class TemporalMemory(nn.Module):
    """
    A small GRU-based temporal memory for storing short-timescale state.
    Accepts fused 512-dim input and produces hidden state.
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x, hx=None):
        # x: (B,1,512) or (B,T,512)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, h = self.gru(x, hx)
        # return last timestep hidden
        return out[:, -1, :], h


class SovarielOmegaV5(nn.Module):
    """
    Stateful Sovariel core with:
      - encoder -> hidden
      - temporal memory (GRU)
      - coherence head
      - attractor centroids
      - episodic buffer (external, but class includes helpers)
      - dream renderer (latent -> image)
    """

    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder from 512 -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # temporal memory
        self.memory = TemporalMemory(input_dim=256, hidden_dim=256)
        self.coherence_head = nn.Linear(256, 1)

        # simple attractor bank (K centroids in latent 256 space)
        self.K = 16
        self.register_buffer("attractors", torch.randn(self.K, 256) * 0.05)  # initialized small noise
        self.attractor_counts = np.zeros(self.K, dtype=np.int32)

        # episodic replay buffer (list of tuples)
        self.buffer_capacity = 20000
        self.episodic_buffer = collections.deque(maxlen=self.buffer_capacity)

        # dream decoder: latent 256 -> 3x224x224 float image (sigmoid)
        # keep small so it's fast; can be replaced with a learned upconv
        self.dream_decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * 224 * 224),
        )

        # gating/clamping hyperparameters
        self.coherence_threshold = 0.97
        self.attractor_update_rate = 0.01  # EMA update towards incoming latent
        self.replay_batch = 8

        self.to(self.device)

    def encode(self, fused):
        # fused: (B,512) -> latent (B,256)
        z = self.encoder(fused)  # (B,256)
        return z

    def step(self, fused, timestep=None):
        """
        Main forward for online timestep.
        Accepts fused (B,512) tensor.
        Returns: coherence (B,1) tensor and hidden (B,256)
        """
        if fused.dim() == 2 and fused.shape[0] == 1:
            # keep batch semantics
            pass
        fused = fused.to(self.device)
        z = self.encode(fused)  # (B,256)
        # memory update (single-step)
        hidden, _ = self.memory(z)
        coherence = torch.sigmoid(self.coherence_head(hidden))
        return coherence, hidden, z

    def forward(self, x1, x2=None):
        # backward-compatible signature; call step
        return self.step(x1)

    # ------------------
    # Episodic buffer helpers
    # ------------------
    def buffered_append(self, latent, coherence_scalar, meta=None):
        # latent: 1D or 2D tensor (B,256)
        if isinstance(latent, torch.Tensor):
            latent_np = latent.detach().cpu().numpy().astype(np.float32)
        else:
            latent_np = np.array(latent, dtype=np.float32)
        entry = {
            "latent": latent_np,
            "coherence": float(coherence_scalar),
            "ts": time.time(),
            "meta": meta
        }
        self.episodic_buffer.append(entry)

    def sample_replay(self, k=None):
        """Return k latents sampled with probability proportional to coherence (prioritized)."""
        if k is None:
            k = self.replay_batch
        buf = list(self.episodic_buffer)
        if len(buf) == 0:
            return []
        coherences = np.array([b["coherence"] for b in buf], dtype=np.float64)
        # raise scores to power to concentrate on high coherence
        scores = np.power(coherences + 1e-6, 3.0)
        probs = scores / (scores.sum() + 1e-12)
        idx = np.random.choice(len(buf), size=min(k, len(buf)), replace=False, p=probs)
        samples = [buf[i]["latent"] for i in idx]
        # convert to torch
        return [torch.from_numpy(s).to(self.device).unsqueeze(0) for s in samples]

    # ------------------
    # Attractor bank
    # ------------------
    def sample_attractor(self):
        """
        Sample an attractor centroid probabilistically (closer points favored).
        Returns a latent (1,256) torch tensor on device.
        """
        # Softmax over 'counts' to bias sampling to more-used attractors
        counts = torch.from_numpy(self.attractor_counts.astype(np.float32)) + 1.0
        probs = (counts / counts.sum()).astype(np.float32)
        idx = np.random.choice(self.K, p=probs)
        centroid = self.attractors[idx:idx + 1].to(self.device)
        # add small Gaussian jitter to make sampling exploratory
        jitter = torch.randn_like(centroid) * 0.05
        return centroid + jitter

    def update_attractor(self, latent):
        """
        Update closest attractor centroid using EMA towards latent.
        latent: (1,256) tensor
        """
        with torch.no_grad():
            lat = latent.detach().cpu().numpy().reshape(-1)
            # compute distance to centroids
            dists = np.linalg.norm(self.attractors.cpu().numpy() - lat.reshape(1, -1), axis=1)
            idx = int(dists.argmin())
            self.attractor_counts[idx] += 1
            # EMA update in-place on buffer (move to CPU)
            cent = self.attractors[idx].detach().cpu().numpy()
            cent = (1.0 - self.attractor_update_rate) * cent + self.attractor_update_rate * lat
            self.attractors[idx] = torch.from_numpy(cent).to(self.device)

    # ------------------
    # Dream engine: render latent -> image, optionally re-inject
    # ------------------
    def render_dream(self, latent, save_path=None):
        """
        Input: latent (1,256) tensor on device or CPU -> returns PIL Image
        Also optionally save to disk and returns image.
        """
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).to(self.device).unsqueeze(0)
        elif torch.is_tensor(latent):
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            latent = latent.to(self.device)
        else:
            latent = torch.tensor(latent, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.dream_decoder(latent)  # (1,3*224*224)
            out = out.view(1, 3, 224, 224)
            out = torch.sigmoid(out).cpu().numpy()[0]  # (3,224,224) in [0,1]
            # convert to uint8
            img = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  # HWC
            pil = Image.fromarray(img)
            if save_path:
                pil.save(save_path)
            return pil

    # ------------------
    # Closed-loop dream rehearsal
    # ------------------
    def rehearse_dream(self, n_samples=4, inject_back=False, save_dir=None, osc_client=None):
        """
        Sample attractors and buffer items, synthesize latent dream scenes,
        optionally re-inject into episodic loop for rehearsal.
        """
        samples = []
        # mix attractor samples and high-priority replay samples
        for _ in range(n_samples // 2):
            samples.append(self.sample_attractor())
        replay = self.sample_replay(k=n_samples // 2)
        samples.extend(replay)

        imgs = []
        for i, latent in enumerate(samples):
            # ensure shape (1,256)
            if latent.dim() == 2 and latent.shape[0] != 1:
                latent = latent[:1]
            # render dream image
            fname = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"dream_{int(time.time())}_{i}.png")
            img = self.render_dream(latent, save_path=fname)
            imgs.append(img)
            # update attractor bank using this latent (reinforce)
            self.update_attractor(latent)
            # optional re-injection: encode -> feed into buffer with lowered coherence
            if inject_back:
                # create soft coherence to avoid runaway
                fake_coh = float(min(0.95, 0.5 + np.random.rand() * 0.5))
                self.buffered_append(latent.cpu(), fake_coh, meta={"dream": True})
            # optional OSC send
            if osc_client is not None:
                try:
                    # serialize coarse metadata
                    osc_client.send_message("/sovariel/dream", [time.time(), float(self.coherence_head(latent.to(self.device)).item())])
                except Exception:
                    pass
        return imgs


# -------------------------
# 3) Live inputs (webcam, mic, LSL EEG, OSC)
# -------------------------
class LiveInputs:
    def __init__(self, audio_chunk=16384, osc_port=7000):
        self.data = {"visual": None, "audio": None, "eeg": None, "osc": None}
        self.lock = threading.Lock()
        self.audio_q = queue.Queue(maxsize=4)
        self.osc_q = queue.Queue(maxsize=16)
        self.audio_chunk = audio_chunk
        self.osc_port = osc_port

        # Start live threads when libs available
        if pyaudio is not None:
            threading.Thread(target=self._audio_worker, daemon=True).start()
        else:
            print("pyaudio not installed: microphone disabled", file=sys.stderr)

        if pylsl is not None:
            threading.Thread(target=self._lsl_worker, daemon=True).start()
        else:
            print("pylsl not installed: LSL/EEG disabled", file=sys.stderr)

        if dispatcher is not None and osc_server is not None:
            self._start_osc_server()
        else:
            print("python-osc not installed: OSC server disabled", file=sys.stderr)

    def _audio_worker(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000,
                            input=True, frames_per_buffer=self.audio_chunk)
            while True:
                try:
                    raw = stream.read(self.audio_chunk, exception_on_overflow=False)
                except Exception:
                    continue
                buf = np.frombuffer(raw, dtype=np.float32).copy()
                try:
                    self.audio_q.put_nowait(buf)
                except queue.Full:
                    try:
                        _ = self.audio_q.get_nowait()
                        self.audio_q.put_nowait(buf)
                    except Exception:
                        pass
        except Exception as e:
            print("Audio worker failed:", e, file=sys.stderr)

    def _lsl_worker(self):
        try:
            streams = pylsl.resolve_stream("type", "EEG", timeout=5)
            if not streams:
                print("No EEG streams found (timeout=5s)", file=sys.stderr)
                return
            inlet = pylsl.StreamInlet(streams[0])
            print(f"Connected to EEG stream: {streams[0].name()}")
            while True:
                sample, ts = inlet.pull_sample(timeout=0.0)
                if sample:
                    arr = np.array(sample, dtype=np.float32)
                    if arr.size < 1024:
                        arr = np.pad(arr, (0, 1024 - arr.size))
                    else:
                        arr = arr[:1024]
                    try:
                        with self.lock:
                            self.data["eeg"] = arr
                    except Exception:
                        pass
        except Exception as e:
            print("LSL worker failed:", e, file=sys.stderr)

    def _start_osc_server(self):
        def handler(address, *args):
            try:
                vals = np.array([float(x) for x in args], dtype=np.float32)
                if vals.size < 1024:
                    vals = np.pad(vals, (0, 1024 - vals.size))
                else:
                    vals = vals[:1024]
                try:
                    self.osc_q.put_nowait(vals)
                except queue.Full:
                    try:
                        _ = self.osc_q.get_nowait()
                        self.osc_q.put_nowait(vals)
                    except Exception:
                        pass
            except Exception:
                pass

        disp = dispatcher.Dispatcher()
        disp.map("/*", handler)
        try:
            server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", self.osc_port), disp)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"OSC server listening on 0.0.0.0:{self.osc_port}")
        except Exception as e:
            print("Failed to start OSC server:", e, file=sys.stderr)

    def get_latest(self):
        with self.lock:
            # pull newest audio if available
            try:
                while not self.audio_q.empty():
                    self.data["audio"] = self.audio_q.get_nowait()
            except Exception:
                pass
            try:
                while not self.osc_q.empty():
                    self.data["osc"] = self.osc_q.get_nowait()
            except Exception:
                pass
            # return copy
            return {k: v for k, v in self.data.items()}


# -------------------------
# 4) Supervisor + main loop (closed-loop)
# -------------------------
class Supervisor:
    """
    Coordinates encoder, core, live inputs, attractor rehearsal, and UI.
    """

    def __init__(self, device=None, osc_out=None, dream_dir="dreams", replay_every=60.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = RealWorldEncoders().eval().to(self.device)
        self.core = SovarielOmegaV5(device=self.device)
        self.live = LiveInputs()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.osc_out_client = None
        if osc_out and udp_client is not None:
            try:
                self.osc_out_client = udp_client.SimpleUDPClient(osc_out[0], osc_out[1])
            except Exception:
                self.osc_out_client = None
        self.dream_dir = dream_dir
        self.replay_every = replay_every
        self.last_replay = time.time()

    def _prepare_visual(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t = self.transform(pil).unsqueeze(0).to(self.device)
        return t

    def run(self):
        if not self.cap.isOpened():
            print("Webcam not available. Exiting.", file=sys.stderr)
            return

        print("Sovariel Ω v7 production loop starting.", now_str())
        print("Press ESC in the display window to exit.")

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # prepare sensory packet
                sensory = self.live.get_latest()
                vis_tensor = None
                try:
                    vis_tensor = self._prepare_visual(frame)
                except Exception:
                    vis_tensor = None

                # package inputs for encoder
                inputs = {}
                if vis_tensor is not None:
                    inputs["visual"] = vis_tensor
                if sensory.get("audio") is not None:
                    inputs["audio"] = sensory["audio"]
                if sensory.get("eeg") is not None:
                    inputs["eeg"] = sensory["eeg"]
                if sensory.get("osc") is not None:
                    inputs["osc"] = sensory["osc"]

                # fuse
                try:
                    with torch.no_grad():
                        fused = self.encoder(inputs)  # (1,512)
                        coherence, hidden, latent = self.core.step(fused)
                        # coherence: (B,1) tensor
                        c = float(coherence.detach().cpu().item())
                except Exception as e:
                    print("Error during encoder/core step:", e, file=sys.stderr)
                    continue

                # append to episodic buffer
                try:
                    self.core.buffered_append(latent, c, meta={"visual": bool("visual" in inputs)})
                except Exception:
                    pass

                # update attractors with occasional reinforcement for on-line latents
                try:
                    if c > 0.90:
                        # strengthen attractor near this latent
                        self.core.update_attractor(latent)
                except Exception:
                    pass

                # Closed-loop dream trigger
                if c >= self.core.coherence_threshold:
                    # coherence high: trigger a dream event (non-blocking)
                    threading.Thread(target=self._trigger_dream_event, args=(latent,), daemon=True).start()

                # Periodic rehearsal (replay + dream)
                if time.time() - self.last_replay > self.replay_every:
                    threading.Thread(target=self._periodic_replay, daemon=True).start()
                    self.last_replay = time.time()

                # UI overlay
                overlay = frame.copy()
                cv2.putText(overlay, f"Coherence: {c:.4f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                # show attractor usage sum
                usage = int(self.core.attractor_counts.sum())
                cv2.putText(overlay, f"Attractors used: {usage}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Sovariel Omega v7 (production)", overlay)

                # optional OSC telemetry
                if self.osc_out_client is not None:
                    try:
                        self.osc_out_client.send_message("/sovariel/coherence", [time.time(), c])
                    except Exception:
                        pass

                # keyboard exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

        finally:
            self.shutdown()

    def _trigger_dream_event(self, latent):
        """
        Called when coherence crosses threshold.
        Produces a dream, stores it, and optionally re-injects into buffer.
        """
        try:
            imgs = self.core.rehearse_dream(n_samples=4, inject_back=True, save_dir=self.dream_dir,
                                           osc_client=self.osc_out_client)
            # optionally show the first dream
            if len(imgs) > 0:
                img = np.array(imgs[0].resize((400, 400)))
                cv2.imshow("Dream (trigger)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
        except Exception as e:
            print("Dream trigger failed:", e, file=sys.stderr)

    def _periodic_replay(self):
        """
        Periodic rehearsal routine which runs attractor sampling and replay.
        """
        try:
            imgs = self.core.rehearse_dream(n_samples=8, inject_back=True, save_dir=self.dream_dir,
                                           osc_client=self.osc_out_client)
            # Optionally log
            print(f"[{now_str()}] Periodic rehearsal produced {len(imgs)} dreams.")
        except Exception as e:
            print("Periodic replay failed:", e, file=sys.stderr)

    def shutdown(self):
        self.running = False
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Supervisor shutdown complete.", now_str())


# -------------------------
# 5) Entrypoint
# -------------------------
def main():
    # allow controlling OSC out via env or args; default no OSC
    osc_target = None
    if len(sys.argv) >= 3:
        try:
            host = sys.argv[1]
            port = int(sys.argv[2])
            osc_target = (host, port)
        except Exception:
            osc_target = None

    sup = Supervisor(osc_out=osc_target, dream_dir="dreams", replay_every=60.0)
    sup.run()


if __name__ == "__main__":
    main()
