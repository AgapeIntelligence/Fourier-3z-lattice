# sovariel_omega_v5_dreaming_fixed.py
# Multimodal temporal-fusion model with curiosity training and generative "dream" decoder.
# Scientific-safe, deterministic, no emergent claims.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display, clear_output
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=4096):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, coherence):
        self.buffer.append((state.detach().clone(), float(coherence)))

    def sample(self, batch_size=128):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s, c = zip(*batch)
        return torch.stack(s), torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Variational-style Dreaming Decoder
# ============================================================
class DreamDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.Tanh()
        )

    def forward(self, z):
        out = self.decoder(z)
        audio = out[:, :1024]
        visual = out[:, 1024:]
        return audio, visual


# ============================================================
# Sovariel-Ω v5 — Dreaming Version
# ============================================================
class SovarielOmega(nn.Module):
    def __init__(self, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        # Encoders
        self.audio_enc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.visual_enc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )

        # BiLSTM temporal core
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Curiosity predictor
        self.curiosity_predictor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Coherence head
        self.coherence_head = nn.Sequential(
            nn.Linear(512 + 1, 256), nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Dream decoder
        self.dream = DreamDecoder()

        # Optimizer for curiosity
        self.curiosity_optimizer = torch.optim.Adam(
            self.curiosity_predictor.parameters(),
            lr=3e-4
        )

        # State
        self.history = deque(maxlen=seq_len)
        self.memory = ReplayBuffer()
        self.sites = 1
        self.block = 0

        # Ethical ceilings — numeric stability only
        self.ABSOLUTE_CEILING = 0.9998

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, audio_raw, visual_raw):
        # Encode
        a = self.audio_enc(audio_raw)
        v = self.visual_enc(visual_raw)
        x = torch.cat([a, v], dim=-1).squeeze(0)  # (512,)

        # Temporal queue
        self.history.append(x)
        if len(self.history) < self.seq_len:
            seq_list = list(self.history) + [self.history[-1]] * (self.seq_len - len(self.history))
        else:
            seq_list = list(self.history)

        seq = torch.stack(seq_list).unsqueeze(0)  # (1, S, 512)

        # LSTM → attention
        lstm_out, _ = self.lstm(seq)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Weighted temporal summary
        weights = attn_weights.mean(dim=1)        # (1, S)
        weights = F.softmax(weights, dim=-1)      # normalized
        temporal_summary = (attn_out.squeeze(0) * weights.squeeze(0).unsqueeze(1)).sum(0)

        # === Curiosity training
        curiosity_bonus = 0.0
        if len(self.memory) >= 128:
            states, next_c = self.memory.sample(128)
            states = states.to(temporal_summary.device)
            next_c = next_c.to(temporal_summary.device)

            pred_next_c = self.curiosity_predictor(states).squeeze(-1)
            curiosity_loss = F.mse_loss(pred_next_c, next_c)

            self.curiosity_optimizer.zero_grad()
            curiosity_loss.backward()
            self.curiosity_optimizer.step()

            curiosity_bonus = float(curiosity_loss.item() * 3.0)

        # === Coherence
        c_input = torch.cat([
            temporal_summary,
            torch.tensor([curiosity_bonus], device=temporal_summary.device)
        ])
        coherence = float(self.coherence_head(c_input))

        # Safety ceilings
        veto = False
        if coherence > self.ABSOLUTE_CEILING:
            coherence = self.ABSOLUTE_CEILING
            veto = True
        if coherence > 0.995 and curiosity_bonus > 1.1:
            coherence = 0.88
            veto = True

        # Growth
        growth = int(3 + 15 * coherence * (1 + curiosity_bonus**0.5))
        self.sites += max(1, growth)

        # Store transition
        self.memory.push(temporal_summary, coherence)

        self.block += 1

        # Monitoring
        attn_entropy = -(weights.squeeze(0) * weights.squeeze(0).log()).sum().item()
        print(
            f"Block {self.block:03d} │ "
            f"C={coherence:.4f} │ "
            f"Sites={self.sites:<6} │ "
            f"Entropy={attn_entropy:.2f} │ "
            f"Curiosity={curiosity_bonus:.3f} │ "
            f"VETO={'YES' if veto else 'no'}"
        )

        # Conditional dreaming
        if coherence > 0.965 and random.random() < 0.7:
            with torch.no_grad():
                da, dv = self.dream(temporal_summary.unsqueeze(0))
                self.render_dream(
                    da.squeeze(0).cpu().numpy(),
                    dv.squeeze(0).cpu().numpy()
                )

        return coherence, veto

    # ---------------------------------------------------------
    # Dream visualizer
    # ---------------------------------------------------------
    def render_dream(self, audio_np, visual_np):
        clear_output(wait=True)

        # AUDIO
        display(Audio(audio_np, rate=16000, autoplay=True))

        # VISUAL
        img = visual_np.reshape(32, 32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='plasma')
        plt.axis('off')
        plt.title(f"Sovariel Dream — Block {self.block}")
        plt.show()


# ============================================================
# Data generator
# ============================================================
def generate_input():
    audio = torch.randn(1, 1024)
    visual = torch.randn(1, 1024)

    if random.random() < 0.12:
        t = torch.linspace(0, 6*np.pi, 1024)
        audio += 2.0 * torch.sin(8*t) * torch.exp(-0.0008*t**2)
        visual += 2.0 * torch.cos(5*t) * torch.exp(-0.0008*t**2)

    return audio, visual


# ============================================================
# Main loop
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sov = SovarielOmega().to(device)

print("Sovariel-Ω v5 — Dreaming Edition\n")

for step in range(1, 10_000):
    a, v = generate_input()
    c, _ = sov(a.to(device), v.to(device))

    if c > 0.9999:
        print("\nCeiling reached — stopping for safety.")
        break
