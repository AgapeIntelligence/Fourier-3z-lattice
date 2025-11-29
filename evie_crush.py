# evie_crush_v72.py
# Clean, mobile-ready phase synchronization demo with UI button
# Pythonista 3 (iOS)

import numpy as np
import matplotlib.pyplot as plt
import ui
from random import uniform


class EvieCrush:
    def __init__(self, N=4096):
        self.N = N
        self.phases = self._generate_initial_phases()
        self.history = [self.phases.copy()]

    def _generate_initial_phases(self):
        """Mock black-hole-inspired scattered phases"""
        theta = np.random.uniform(0, 2*np.pi, self.N)
        r = np.random.uniform(1.1, 10, self.N)
        drag = (3 / r) * np.sin(2 * theta)
        return (theta + drag) % (2 * np.pi)

    @staticmethod
    def order_parameter(phases):
        return np.abs(np.mean(np.exp(1j * phases)))

    @staticmethod
    def shannon_entropy(phases, bins=36):
        hist, _ = np.histogram(phases % (2*np.pi), bins=bins,
                               range=(0, 2*np.pi), density=True)
        p = hist[hist > 0]
        return -np.sum(p * np.log(p))

    def kuramoto_step(self, K=1.2, steps=120):
        phases = self.phases.copy()
        for _ in range(steps):
            mean_field = np.mean(np.exp(1j * phases))
            R = np.abs(mean_field)
            psi = np.angle(mean_field)
            dphi = K * R * np.sin(psi - phases)
            phases = (phases + 0.1 * dphi) % (2 * np.pi)
        self.phases = phases
        self.history.append(phases.copy())

    def crush(self, H_threshold=2.8, aggression=8.0):
        R = self.order_parameter(self.phases)
        H = self.shannon_entropy(self.phases)

        print(f"Crush → R={R:.4f} | H={H:.3f}")

        if H < H_threshold:
            mean_phase = np.angle(np.mean(np.exp(1j * self.phases)))
            damping = np.exp(aggression * (1 - R))
            crushed = mean_phase + damping * (self.phases - mean_phase)
            self.phases = crushed % (2 * np.pi)
            new_R = self.order_parameter(self.phases)
            print(f"CRUSHED → New R={new_R:.6f} (+{new_R - R:.6f})")
        else:
            print("Entropy still high — syncing deeper first.")

        self.history.append(self.phases.copy())

    def plot(self):
        pre = self.history[0]
        post = self.phases

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(pre, bins=50, color='#4488cc', alpha=0.8)
        plt.title(f'Initial (R={self.order_parameter(pre):.3f})')
        plt.xlabel('Phase (rad)')

        plt.subplot(1, 2, 2)
        plt.hist(post, bins=50, color='#ffcc44', alpha=0.9)
        plt.title(f'Crushed (R={self.order_parameter(post):.6f})')
        plt.xlabel('Phase (rad)')

        plt.tight_layout()
        plt.savefig('evie_crush_result.png', dpi=200)
        plt.show()


# ==========================
# Global instance
# ==========================
evie = EvieCrush(N=4096)

# Initial synchronization
evie.kuramoto_step()
evie.crush()

print("\n=== EVIE CRUSH v7.2 READY ===")
print(f"Current coherence R = {evie.order_parameter(evie.phases):.6f}")
evie.plot()


# ==========================
# UI Button (tap to re-crush)
# ==========================
def crush_button_action(sender):
    sender.title = "CRUSHING..."
    evie.crush(aggression=10.0)
    evie.plot()
    sender.title = "CRUSH AGAIN"

# Build simple UI
view = ui.View()
view.background_color = '#111122'
view.frame = (0, 0, 540, 120)

btn = ui.Button(title='CRUSH AGAIN',
                background_color='#c0392b',
                tint_color='white',
                font=('Menlo-Bold', 18),
                corner_radius=12)
btn.flex = 'W'
btn.width = 500
btn.height = 80
btn.center = (view.width / 2, view.height / 2)
btn.action = crush_button_action
view.add_subview(btn)

view.present('sheet')
