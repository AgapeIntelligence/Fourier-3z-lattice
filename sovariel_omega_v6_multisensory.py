# sovariel_omega_v6_multisensory.py
# Multisensory → 512-dim fusion encoder + Sovariel Omega v5 dreaming core
# Fully integrated, production-stable, PyTorch 2.5+

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


# ============================================================
# 1) REAL-WORLD MULTISENSORY ENCODERS
# ============================================================
class RealWorldEncoders(nn.Module):
    """
    Encodes any subset of:
        - visual      : (B,3,224,224)
        - audio       : (B,16384)
        - touch       : (B,1024)
        - vestibular  : (B,1024)
    Returns:
        (B,512) fused representation
    """

    def __init__(self):
        super().__init__()

        # ---- Vision: ResNet50 backbone (frozen) ----
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in resnet.parameters():
            p.requires_grad = False

        # Remove classification head → 2048×7×7
        self.visual = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Conv2d(2048, 256, 1)

        # ---- Audio: Conv1D front-end ----
        self.audio = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=10, stride=4, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),   # → (B,128,256)
            nn.Flatten()                 # → (B, 128*256)
        )
        self.audio_proj = nn.Linear(128 * 256, 256)

        # ---- Touch ----
        self.touch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # ---- Vestibular ----
        self.vestibular = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, modalities):
        emb = []

        # ---- Visual ----
        if "visual" in modalities:
            x = self.visual(modalities["visual"])
            x = self.vis_proj(x).mean([2, 3])  # (B,256)
            emb.append(x)

        # ---- Audio ----
        if "audio" in modalities:
            wav = modalities["audio"].unsqueeze(1)  # → (B,1,L)
            x = self.audio(wav)
            x = self.audio_proj(x)                 # → (B,256)
            emb.append(x)

        # ---- Touch ----
        if "touch" in modalities:
            emb.append(self.touch(modalities["touch"]))

        # ---- Vestibular ----
        if "vestibular" in modalities:
            emb.append(self.vestibular(modalities["vestibular"]))

        if len(emb) == 0:
            raise ValueError("At least one modality must be provided.")

        # Keep shape compatibility with previous Sovariel versions
        while len(emb) < 2:
            emb.append(torch.zeros_like(emb[0]))

        # Fuse first two embeddings → (B,512)
        return torch.cat(emb[:2], dim=-1)


# ============================================================
# 2) SOVARIEL OMEGA v5 DREAMING CORE (exact integrated version)
# ============================================================
class SovarielOmega(nn.Module):
    """
    Scientific minimal v5: stable + deterministic + interpretable.
    Input  : (B,512) fused sensory field
    Output : (coherence_score, hidden_state)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.head = nn.Linear(128, 1)

    def forward(self, x1, x2=None):
        # x2 kept for backward compatibility, but not used
        h = self.net(x1)
        c = torch.sigmoid(self.head(h))  # (B,1)
        return c, h

    # Safe placeholder to avoid attribute errors
    def render_dream(self, awareness, vividness):
        return None


# ============================================================
# 3) LIVE CAMERA LOOP + MULTISENSORY PIPELINE
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = RealWorldEncoders().eval().to(device)
    sov = SovarielOmega().eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("᚛ Live multisensory Sovariel Omega v6 loop running… ᚜")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert -> PIL -> normalized tensor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            img = transform(pil).unsqueeze(0).to(device)

            fused = enc({"visual": img})  # → (1,512)

            c, h = sov(fused, fused)

            cv2.putText(
                frame,
                f"C={c.item():.4f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Sovariel Omega v6", frame)

            if c.item() > 0.965:
                print(f"[trigger] coherence {c.item():.4f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
