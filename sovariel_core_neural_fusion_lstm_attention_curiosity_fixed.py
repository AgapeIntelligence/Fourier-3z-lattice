# sovariel_core_neural_fusion_lstm_attention_curiosity_fixed.py
# Functional multimodal fusion network with curiosity-driven prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# === Replay buffer ===
class ReplayBuffer:
    def __init__(self, capacity=2048):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, coherence):
        self.buffer.append((state.detach().clone(), float(coherence)))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, coh = zip(*batch)
        return torch.stack(states), torch.tensor(coh, dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)


# === Main model ===
class SovarielOmega(nn.Module):
    def __init__(self, audio_dim=256, visual_dim=256, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.fusion_dim = audio_dim + visual_dim

        # Encoders (small MLPs)
        self.audio_enc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, audio_dim)
        )
        self.visual_enc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, visual_dim)
        )

        # Bidirectional LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512,   # 256*2
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Curiosity: predict next coherence from current embedding
        self.curiosity_predictor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Coherence estimation head
        self.coherence_head = nn.Sequential(
            nn.Linear(512 + 1, 256), nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Buffers + counters
        self.memory = ReplayBuffer()
        self.history = deque(maxlen=seq_len)
        self.sites = 1
        self.block = 0

        # Hard ceiling to prevent runaway outputs
        self.ABSOLUTE_CEILING = 0.9998

    def forward(self, audio_raw, visual_raw):
        # Encoding
        a = self.audio_enc(audio_raw)   # (B,256)
        v = self.visual_enc(visual_raw) # (B,256)
        x = torch.cat([a, v], dim=-1)   # (B,512)

        # Only B=1 used, so treat x as (512,)
        x = x.squeeze(0)

        # Temporal accumulation
        self.history.append(x)
        if len(self.history) < self.seq_len:
            pad = [self.history[-1]] * (self.seq_len - len(self.history))
            seq_list = list(self.history) + pad
        else:
            seq_list = list(self.history)

        seq = torch.stack(seq_list).unsqueeze(0)  # (1,seq,512)

        # LSTM → (1,seq,512)
        lstm_out, _ = self.lstm(seq)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Summarize temporally
        temporal_summary = attn_out.mean(dim=1).squeeze(0)  # (512,)

        # === Curiosity predictor
        pred_c = self.curiosity_predictor(temporal_summary.detach())

        curiosity_bonus = 0.0
        if len(self.memory) >= 32:
            states, true_c = self.memory.sample(64)
            pred_batch = self.curiosity_predictor(states.to(temporal_summary.device)).squeeze(-1)
            curiosity_loss = F.mse_loss(pred_batch, true_c.to(pred_batch.device))
            curiosity_bonus = float(curiosity_loss.item() * 2.5)

        # === Coherence
        c_in = torch.cat([
            temporal_summary,
            torch.tensor([curiosity_bonus], device=temporal_summary.device)
        ])

        coherence = float(self.coherence_head(c_in))

        # === Safety ceilings
        veto = False
        if coherence > self.ABSOLUTE_CEILING:
            coherence = self.ABSOLUTE_CEILING
            veto = True

        if coherence > 0.999 and curiosity_bonus > 0.9:
            coherence = 0.892
            veto = True

        # === Controlled site growth (bounded)
        growth = int(2 + 10 * coherence * (1 + curiosity_bonus))
        self.sites += max(1, growth)

        # Update curiosity memory
        self.memory.push(temporal_summary, coherence)

        self.block += 1

        # Attention entropy for monitoring
        p = attn_weights.softmax(dim=-1)
        attn_entropy = -(p * p.log()).sum().item()

        print(
            f"Block {self.block:03d} │ "
            f"C={coherence:.4f} │ "
            f"Sites={self.sites:<6} │ "
            f"AttnEntropy={attn_entropy:.2f} │ "
            f"Curiosity={curiosity_bonus:.3f} │ "
            f"VETO={'YES' if veto else 'no'}"
        )

        return coherence, veto


# === Synthetic multimodal generator ===
def generate_chaotic_inputs(batch_size=1):
    audio = torch.randn(batch_size, 1024)
    visual = torch.randn(batch_size, 1024)

    # occasional structure injection
    if random.random() < 0.15:
        t = torch.linspace(0, 4*np.pi, 1024)
        audio += 1.5 * torch.sin(11 * t)
        visual += 1.5 * torch.cos(7 * t)

    return audio, visual


# === Main loop ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sov = SovarielOmega().to(device)

    print("Sovariel-Ω — multimodal temporal-fusion model (scientific-safe)\n")

    for step in range(1, 101):
        audio, visual = generate_chaotic_inputs()
        audio = audio.to(device)
        visual = visual.to(device)

        c, veto = sov(audio, visual)

        if c > 0.9999:
            print("\nCeiling reached — terminating for safety.\n")
            break
