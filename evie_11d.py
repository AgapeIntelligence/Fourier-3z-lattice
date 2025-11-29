# evie_11d_prototype.py
# Full 11D Phase Sync Prototype — scientific, production-ready
# - Generates (or loads) an 11D correlated phase manifold
# - PhaseSyncController (Kuramoto-style control law, drift, adaptive gain)
# - Real-time visualization: per-dim unit circles, PCA projection, R(t) trace, error heatmap
# Requirements: numpy, matplotlib
# Author: (adapted for Evie) — MIT license

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, CheckButtons

# ---------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------
MANIFOLD_FILE = "evie_11d_manifold.npy"
MANIFOLD_N = 10000         # number of samples in manifold (used only for generation)
D = 11                     # dimensionality (11D)
NOISE_STD = 0.6            # initial controller phase noise
DRIFT_SIGMA = 0.08         # phase drift per cycle
CRUSH_GAIN = 14.0          # base control gain
AUTO_CYCLES = 30           # number of auto crushing iterations in demo mode
PAUSE_INTERVAL = 0.25      # seconds between UI updates in auto mode
SEED = 1

np.random.seed(SEED)


# ---------------------------
# MANIFOLD GENERATION / LOADING
# ---------------------------
def generate_correlated_manifold(N=MANIFOLD_N, dims=D, correlation=0.75):
    """
    Generate a correlated manifold on the N-torus.
    This produces samples with a shared base phase + small independent noise,
    then wrapped to [0, 2π).
    """
    base = np.random.uniform(0, 2*np.pi, size=(N, 1))
    noise = np.random.normal(0, (1.0 - correlation), size=(N, dims))
    M = (base + noise) % (2*np.pi)
    np.save(MANIFOLD_FILE, M)
    return M


def load_or_create_manifold():
    if os.path.exists(MANIFOLD_FILE):
        M = np.load(MANIFOLD_FILE)
        if M.ndim != 2 or M.shape[1] != D:
            raise ValueError(f"Found {MANIFOLD_FILE} but shape mismatch; expected (N,{D})")
        return M % (2*np.pi)
    else:
        print(f"No manifold file found; generating correlated {D}D manifold ({MANIFOLD_N} samples)")
        return generate_correlated_manifold(N=MANIFOLD_N, dims=D, correlation=0.75)


# ---------------------------
# PHASE SYNCHRONIZATION CONTROLLER
# ---------------------------
class PhaseSyncController:
    """
    N-dimensional phase synchronization controller (Kuramoto-style control law).
    - ref_phase: target/reference phase vector (N,)
    - phases: current phases (N,)
    """

    def __init__(self, ref_phase, noise=NOISE_STD):
        self.ref = np.asarray(ref_phase) % (2*np.pi)
        self.N = len(self.ref)
        # Initialize phases with noise around reference
        self.phases = (self.ref + np.random.normal(0, noise, self.N)) % (2*np.pi)

    @staticmethod
    def wrap_to_pi(x):
        """Wrap to [-pi, pi]."""
        return (x + np.pi) % (2*np.pi) - np.pi

    def order_parameter(self):
        """Kuramoto order parameter R = | mean(exp(i theta)) |"""
        return np.abs(np.mean(np.exp(1j * self.phases)))

    def phase_error(self):
        """Wrapped phase error vector relative to reference in [-pi, pi]."""
        return self.wrap_to_pi(self.phases - self.ref)

    def crush(self, gain=CRUSH_GAIN):
        """
        Apply a control step that reduces error toward the reference.
        Adaptive effect: stronger correction when R is small (disordered).
        """
        err = self.phase_error()
        R = self.order_parameter()
        damping = np.exp(gain * (1.0 - R))
        # New phases move toward reference plus scaled error; wrap to [0,2π)
        self.phases = (self.ref + damping * err) % (2*np.pi)
        return R

    def apply_drift(self, sigma=DRIFT_SIGMA):
        """Apply stochastic drift to the phases (random walk step)."""
        self.phases = (self.phases + np.random.normal(0, sigma, self.N)) % (2*np.pi)

    def step(self, drift_sigma=DRIFT_SIGMA):
        """One combined step: drift then (optionally) nothing else."""
        self.apply_drift(sigma=drift_sigma)


# ---------------------------
# PCA UTILITY (for visualization)
# ---------------------------
def pca_2d_projection(samples):
    """
    Compute a 2D PCA projection for the given samples (samples x D).
    Returns projected points (samples x 2), and the two principal components.
    Uses mean-centering on complex embedding via cos/sin transform to respect circularity.
    """
    # Convert phases to 2D embedding per-dimension: stack cos and sin -> 2D*D features
    X = np.concatenate([np.cos(samples), np.sin(samples)], axis=1)  # shape (N, 2D)
    # mean-center
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD for PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt.T[:, :2]  # projection onto first two principal axes
    return coords, Vt[:2, :]


# ---------------------------
# VISUALIZATION / UI
# ---------------------------
def build_visualization(controller: PhaseSyncController, manifold):
    """
    Create Matplotlib figures & return update function and UI widgets.
    Visualization includes:
      - Left: grid of unit circles with arrows for each dimension (up to 11)
      - Top-right: phase histogram (current vs reference)
      - Bottom-right: R(t) trace
      - Bottom-right heatmap: per-dimension error magnitude
    """

    N = controller.N

    # Layout
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 0.9])
    ax_circles = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1:])
    ax_R = fig.add_subplot(gs[1, 1:3])
    ax_err = fig.add_subplot(gs[1, 3])
    ax_pca = fig.add_subplot(gs[2, 1:3])

    # Title and basic settings
    ax_circles.set_title(f"{N}D Phase Indicators")
    ax_circles.set_xlim(-2, 2)
    ax_circles.set_ylim(-2, 2)
    ax_circles.axis('off')
    ax_circles.set_aspect('equal')

    # Precompute positions for unit circles arranged in a roughly circular layout
    angles_layout = np.linspace(0, 2*np.pi, N, endpoint=False)
    radius_layout = 1.6
    pos = np.vstack([radius_layout * np.cos(angles_layout), radius_layout * np.sin(angles_layout)]).T

    # History containers
    R_history = []

    # Draw static circles and labels once
    circle_patches = []
    for i in range(N):
        xi, yi = pos[i]
        c = Circle((xi, yi), 0.5, fill=False, lw=1.2, color='gray', alpha=0.6)
        ax_circles.add_patch(c)
        circle_patches.append(c)
        ax_circles.text(xi, yi + 0.65, f"Dim {i+1}", ha='center', fontsize=9)

    # initial arrows and resultant
    arrow_artists = []  # store as (x0,y0,dx,dy,artist) for redraw
    for i in range(N):
        xi, yi = pos[i]
        dx = 0.45 * np.cos(controller.phases[i])
        dy = 0.45 * np.sin(controller.phases[i])
        art = ax_circles.arrow(xi, yi, dx, dy, width=0.03, color='cyan', length_includes_head=True)
        arrow_artists.append(art)
    resultant_art = ax_circles.arrow(0, 0, 0, 0, width=0.06, color='gold', length_includes_head=True)

    # Histogram setup
    ax_hist.set_title("Phase Histogram (Current vs Reference)")
    ax_hist.set_xlim(0, 2*np.pi)

    # R trace setup
    ax_R.set_title("Order parameter R(t)")
    ax_R.set_ylim(0, 1.02)
    ax_R.set_xlabel("Iteration")
    ax_R.set_ylabel("R")

    # Error heatmap setup
    ax_err.set_title("Per-dimension |error| (rad)")
    ax_err.set_xticks(np.arange(N))
    ax_err.set_xticklabels([f"{i+1}" for i in range(N)])
    ax_err.set_ylim(0, np.pi)
    # PCA axis
    ax_pca.set_title("PCA projection (cos,sin embedding)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Controls: button for APPLY, checkbox for auto mode
    ax_button = plt.axes([0.82, 0.02, 0.12, 0.05])
    btn_apply = Button(ax_button, 'APPLY CONTROL', color='#c0392b', hovercolor='#e74c3c')

    ax_check = plt.axes([0.64, 0.02, 0.16, 0.05])
    check = CheckButtons(ax_check, ['AUTO RUN'], [False])

    auto_run_state = {'enabled': False}

    def on_check(label):
        auto_run_state['enabled'] = not auto_run_state['enabled']
    check.on_clicked(on_check)

    # Update function to be called each iteration
    def update_frame(iteration_index=None):
        # Optionally apply a drift step before plotting
        controller.apply_drift(sigma=DRIFT_SIGMA)

        # Update arrows: to keep simple, clear and redraw arrows each time
        # Remove old arrow artists
        nonlocal arrow_artists, resultant_art
        for art in arrow_artists:
            try:
                art.remove()
            except Exception:
                pass
        try:
            resultant_art.remove()
        except Exception:
            pass
        arrow_artists = []
        for i in range(N):
            xi, yi = pos[i]
            dx = 0.45 * np.cos(controller.phases[i])
            dy = 0.45 * np.sin(controller.phases[i])
            art = ax_circles.arrow(xi, yi, dx, dy, width=0.03, color='cyan', length_includes_head=True)
            arrow_artists.append(art)

        # resultant
        mean_vec = np.mean(np.exp(1j * controller.phases))
        resultant_art = ax_circles.arrow(0, 0, mean_vec.real, mean_vec.imag,
                                         width=0.06, color='gold', length_includes_head=True)

        # Histogram
        ax_hist.clear()
        ax_hist.hist(controller.phases, bins=36, range=(0, 2*np.pi), density=True, alpha=0.7, color='cyan')
        ax_hist.hist(ref_phase, bins=36, range=(0, 2*np.pi), density=True, histtype='step', color='gold', lw=2)
        ax_hist.set_xlim(0, 2*np.pi)
        ax_hist.set_title(f"Phase Histogram | R = {controller.order_parameter():.6f}")

        # R trace
        R_history.append(controller.order_parameter())
        ax_R.clear()
        ax_R.plot(R_history, color='gold', marker='o', markersize=3)
        ax_R.set_ylim(0, 1.02)
        ax_R.set_title("Order parameter R(t)")
        ax_R.set_xlabel("Iteration")
        ax_R.set_ylabel("R")
        ax_R.grid(True, alpha=0.3)

        # Per-dimension error heatmap (absolute wrapped error)
        err = np.abs(PhaseSyncController.wrap_to_pi(controller.phases - controller.ref))
        ax_err.clear()
        ax_err.bar(np.arange(N), err, color='tab:orange')
        ax_err.set_ylim(0, np.pi)
        ax_err.set_xticks(np.arange(N))
        ax_err.set_xticklabels([str(i+1) for i in range(N)], rotation=45)
        ax_err.set_title("Per-dimension |error| (rad)")

        # PCA projection: sample from manifold + include current phases as highlighted point
        # We'll select a subset of manifold points for speed
        sample_idx = np.random.choice(manifold.shape[0], size=min(512, manifold.shape[0]), replace=False)
        subset = manifold[sample_idx]
        pts, _ = pca_2d_projection(subset)
        current_point, _ = pca_2d_projection(controller.phases.reshape(1, -1))
        ax_pca.clear()
        ax_pca.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.35, color='tab:blue', label='manifold samples')
        ax_pca.scatter(current_point[0, 0], current_point[0, 1], s=80, color='red', label='current phase')
        ax_pca.set_title("PCA projection (cos,sin embedding)")
        ax_pca.legend(loc='upper right')

        plt.pause(0.001)

    # Button callback
    def on_apply(event):
        R_before = controller.order_parameter()
        controller.crush(gain=CRUSH_GAIN)
        R_after = controller.order_parameter()
        print(f"Applied control: R_before={R_before:.6f} → R_after={R_after:.6f}")
        update_frame()

    btn_apply.on_clicked(on_apply)

    return update_frame, auto_run_state, btn_apply, check, fig


# ---------------------------
# MAIN: initialize and run
# ---------------------------
if __name__ == "__main__":
    print(f"Loading or generating {D}D manifold...")
    manifold = load_or_create_manifold()
    ref_phase = np.mean(manifold, axis=0)  # global reference

    controller = PhaseSyncController(ref_phase=ref_phase, noise=NOISE_STD)

    update_frame, auto_state, btn, check, fig = build_visualization(controller, manifold)

    # Auto-run loop (desktop friendly)
    try:
        iter_count = 0
        print("Interactive prototype running. Click 'APPLY CONTROL' to enforce phase-lock.")
        print("Toggle 'AUTO RUN' to enable continuous auto-crush.")
        while True:
            if auto_state['enabled']:
                # Apply drift + control each cycle (simulate periodic control)
                controller.apply_drift(sigma=DRIFT_SIGMA)
                controller.crush(gain=CRUSH_GAIN)
                update_frame(iter_count)
                iter_count += 1
                time.sleep(PAUSE_INTERVAL)
            else:
                # idle update to show drift (small)
                controller.apply_drift(sigma=DRIFT_SIGMA * 0.3)
                update_frame(iter_count)
                iter_count += 1
                time.sleep(PAUSE_INTERVAL)
    except KeyboardInterrupt:
        print("\nPrototype stopped by user.")
    except Exception as e:
        print("Error occurred:", str(e))
    finally:
        plt.close(fig)
        print("Exiting. You may rerun the script to continue experiments.")
