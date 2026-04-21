"""
MELU Activation Functions — Analysis & Robustness Suite
========================================================
Generates three figures:
  1. melu_v1_base.png        — Version 1 (no t-CDF): shape, gradient, negative zone
  2. melu_v2_tdt.png         — Version 2 (with t-CDF): shape, CDF comparison, delta
  3. melu_comparison.png     — Side-by-side: inlier/outlier/gradient/saturation/table

Run:
    pip install numpy scipy matplotlib
    python melu_analysis.py

Output: three PNG files saved to ./outputs/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.special import betainc
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Core math
# ══════════════════════════════════════════════════════════════════════════════

def t_cdf(x, nu):
    """
    Student-t CDF via regularised incomplete beta function.
    Vectorised, works for any x shape.

    T_nu(x) = 1 - I_z(nu/2, 1/2) / 2    for x >= 0
            = I_z(nu/2, 1/2) / 2         for x <  0
    where z = nu / (nu + x^2)
    """
    nu = max(float(nu), 2.0)          # enforce nu >= 2
    x  = np.asarray(x, dtype=float)
    z  = nu / (nu + np.clip(x**2, 1e-30, None))
    ib = betainc(nu / 2, 0.5, np.clip(z, 1e-12, 1 - 1e-12))
    return np.where(x >= 0, 1.0 - ib / 2.0, ib / 2.0)


def melu_v1(x, tau, alpha=1.0, beta=0.4, m=None):
    """
    MELU Version 1 — base, no Student-t CDF.

    Term 1 (inlier base):   x          (pure identity)
    Term 2 (amplifier):     alpha * sign(x) * (exp(beta*(m-tau)) - 1)
                            only active when m >= tau

    Parameters
    ----------
    x     : float or array — activation input (scalar xi)
    tau   : float          — adaptive threshold (EMA of mean Mahalanobis)
    alpha : float          — amplifier scale         (learnable, init 1.0)
    beta  : float          — exponential rate        (learnable, init 0.4)
    m     : float or None  — Mahalanobis distance of the full vector h.
                             If None, uses |x| as a 1-D stand-in.
    """
    x   = np.asarray(x, dtype=float)
    m_  = np.abs(x) if m is None else np.full_like(x, float(m))
    # --- Term 1: identity ---
    t1  = x
    # --- Term 2: amplifier ---
    amp = alpha * np.sign(x) * (np.exp(beta * np.maximum(m_ - tau, 0)) - 1)
    gate = (m_ >= tau).astype(float)
    return t1 + gate * amp


def melu_v2(x, tau, alpha=1.0, beta=0.4, nu=4, m=None):
    """
    MELU-Δt Version 2 — with Student-t CDF base.

    Term 1 (inlier base):   x * T_nu(x)     (Student-t Swish)
    Term 2 (amplifier):     alpha * sign(x) * (exp(beta*(m-tau)) - 1)
                            only active when m >= tau

    Parameters
    ----------
    x     : float or array — activation input (scalar xi)
    tau   : float          — adaptive threshold (EMA of mean Mahalanobis)
    alpha : float          — amplifier scale         (learnable, init 1.0)
    beta  : float          — exponential rate        (learnable, init 0.4)
    nu    : float          — Student-t degrees of freedom (learnable, nu >= 2)
    m     : float or None  — Mahalanobis distance of the full vector h.
    """
    x   = np.asarray(x, dtype=float)
    m_  = np.abs(x) if m is None else np.full_like(x, float(m))
    # --- Term 1: Student-t Swish ---
    t1  = x * t_cdf(x, nu)
    # --- Term 2: amplifier ---
    amp = alpha * np.sign(x) * (np.exp(beta * np.maximum(m_ - tau, 0)) - 1)
    gate = (m_ >= tau).astype(float)
    return t1 + gate * amp


# ── Baseline activations ──────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0.0, x)

def elu(x, alpha=1.0):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x / (1.0 + np.exp(-x))

def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# ── Numerical gradient ────────────────────────────────────────────────────────

def numerical_grad(fn, x, eps=1e-4, **kw):
    return (fn(x + eps, **kw) - fn(x - eps, **kw)) / (2 * eps)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Robustness helpers
# ══════════════════════════════════════════════════════════════════════════════

class MELULayer:
    """
    Stateful MELU-Δt layer with EMA-tracked statistics.
    Simulates what a PyTorch nn.Module would do.
    """
    def __init__(self, dim, alpha=1.0, beta=0.4, nu=4, momentum=0.99):
        self.dim      = dim
        self.alpha    = alpha
        self.beta     = beta
        self.nu       = max(float(nu), 2.0)
        self.momentum = momentum
        # EMA buffers (initialised on first batch)
        self.mu_ema   = None
        self.cov_ema  = None
        self.tau_ema  = 1.0
        self.tau_history = []     # for plotting EMA trace

    def _mahalanobis(self, H):
        """H: [batch, dim] → distances [batch]"""
        diff = H - self.mu_ema
        try:
            Sinv = np.linalg.inv(self.cov_ema + 1e-5 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            Sinv = np.eye(self.dim)
        d2 = np.einsum('bi,ij,bj->b', diff, Sinv, diff)
        return np.sqrt(np.maximum(d2, 0.0))

    def _update_ema(self, H):
        """Update running mu, cov, and tau from a new batch."""
        m    = self.momentum
        mu_b = H.mean(axis=0)
        d    = H - mu_b
        cov_b = (d.T @ d) / max(len(H) - 1, 1)

        if self.mu_ema is None:
            self.mu_ema  = mu_b.copy()
            self.cov_ema = cov_b.copy()
        else:
            self.mu_ema  = m * self.mu_ema  + (1 - m) * mu_b
            self.cov_ema = m * self.cov_ema + (1 - m) * cov_b

        dists = self._mahalanobis(H)
        self.tau_ema = m * self.tau_ema + (1 - m) * dists.mean()
        self.tau_history.append(self.tau_ema)

    def forward(self, H, training=True):
        """H: [batch, dim] → output [batch, dim]"""
        if training:
            self._update_ema(H)
        dists = self._mahalanobis(H)        # [batch]
        tau   = self.tau_ema
        T1    = H * t_cdf(H, self.nu)
        gate  = (dists >= tau).astype(float)[:, None]
        exp_m = np.clip(self.beta * (dists - tau), -20, 20)
        amp   = self.alpha * np.sign(H) * (np.exp(exp_m[:, None]) - 1)
        return T1 + gate * amp


def run_gradient_stability(n_trials=40):
    """
    Measure output sensitivity vs contamination rate for each activation.
    Returns dict: name → list of median sensitivity per contamination level.
    """
    contam_rates = np.linspace(0.0, 0.40, 18)
    dim  = 16
    results = {k: [] for k in ["MELU-Δt", "ELU", "Swish", "ReLU"]}

    for cr in contam_rates:
        trial = {k: [] for k in results}
        for _ in range(n_trials):
            n_total = 128
            n_out   = int(n_total * cr)
            n_in    = n_total - n_out
            H_in    = np.random.randn(n_in, dim)
            H_out   = np.random.randn(n_out, dim) * 5.0 + 8.0
            H       = np.vstack([H_in, H_out]) if n_out > 0 else H_in

            layer = MELULayer(dim, momentum=0.9)
            for _ in range(5):
                layer.forward(np.random.randn(64, dim))
            out = layer.forward(H)
            trial["MELU-Δt"].append(np.abs(out - H).max())

            for name, fn in [("ELU", elu), ("Swish", swish), ("ReLU", relu)]:
                trial[name].append(np.abs(fn(H) - H).max())

        for k in results:
            results[k].append(np.median(trial[k]))

    return contam_rates, results


def run_outlier_separation(n_trials=150):
    """
    Compute outlier/inlier activation-norm ratio for each activation.
    Higher ratio = better at making outliers distinguishable.
    """
    dim   = 8
    names = ["MELU-Δt", "ELU", "GELU", "Swish", "ReLU"]
    ratios = {n: [] for n in names}

    for _ in range(n_trials):
        H_in  = np.random.randn(200, dim)
        H_out = np.random.randn(20,  dim) * 4.0 + np.random.randn(dim) * 2

        layer = MELULayer(dim, momentum=0.9)
        for _ in range(10):
            layer.forward(np.random.randn(64, dim))

        out_in  = layer.forward(H_in,  training=False)
        out_out = layer.forward(H_out, training=False)
        ratios["MELU-Δt"].append(
            np.abs(out_out).mean() / (np.abs(out_in).mean() + 1e-8))

        for name, fn in [("ELU",elu),("GELU",gelu),("Swish",swish),("ReLU",relu)]:
            r = np.abs(fn(H_out)).mean() / (np.abs(fn(H_in)).mean() + 1e-8)
            ratios[name].append(r)

    return names, ratios


def run_mcd_detection(n_trials=30):
    """
    Compare outlier detection rate: standard covariance vs MCD-style.
    """
    contam_rates = np.linspace(0.0, 0.35, 15)
    dim = 8
    detect_std, detect_mcd = [], []

    for cr in contam_rates:
        ts, tm = [], []
        for _ in range(n_trials):
            n_total = 200
            n_out   = int(n_total * cr)
            n_in    = n_total - n_out
            H_in    = np.random.randn(n_in, dim)
            H_out   = np.random.randn(n_out, dim) * 1.2 + 4.0
            H       = np.vstack([H_in, H_out]) if n_out > 0 else H_in
            labels  = np.array([0] * n_in + [1] * n_out)

            # --- Standard covariance ---
            mu_s  = H.mean(0)
            d_    = H - mu_s
            cov_s = (d_.T @ d_) / (len(H) - 1) + 1e-4 * np.eye(dim)
            Si_s  = np.linalg.inv(cov_s)
            dist_s = np.sqrt(np.einsum('bi,ij,bj->b', d_, Si_s, d_))
            thr_s  = np.percentile(dist_s, 85)
            ts.append(((dist_s > thr_s) & (labels == 1)).sum() / max(n_out, 1))

            # --- MCD-style (use cleanest 75%) ---
            idx   = np.argsort(np.linalg.norm(H - mu_s, axis=1))[:int(0.75*len(H))]
            H_sub = H[idx]
            mu_m  = H_sub.mean(0)
            d_m   = H_sub - mu_m
            cov_m = (d_m.T @ d_m) / (len(H_sub) - 1) + 1e-4 * np.eye(dim)
            Si_m  = np.linalg.inv(cov_m)
            d_all = H - mu_m
            dist_m = np.sqrt(np.einsum('bi,ij,bj->b', d_all, Si_m, d_all))
            thr_m  = np.percentile(dist_m[idx], 85)
            tm.append(((dist_m > thr_m) & (labels == 1)).sum() / max(n_out, 1))

        detect_std.append(np.mean(ts))
        detect_mcd.append(np.mean(tm))

    return contam_rates, detect_std, detect_mcd


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "MELU-base" : "#D85A30",
    "MELU-Δt"   : "#1D9E75",
    "ReLU"      : "#888780",
    "ELU"       : "#7F77DD",
    "Swish"     : "#BA7517",
    "GELU"      : "#534AB7",
    "MCD"       : "#1D9E75",
    "std"       : "#E24B4A",
}

def style_ax(ax):
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#CCCCCC")
    ax.tick_params(labelsize=8)

def add_threshold_band(ax, tau, ymin, ymax):
    ax.axvline( tau, color="#534AB7", lw=1.0, ls="--", alpha=0.45)
    ax.axvline(-tau, color="#534AB7", lw=1.0, ls="--", alpha=0.45)
    ax.fill_betweenx([ymin, ymax], -tau, tau, alpha=0.05, color="#534AB7")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Version 1  (no t-CDF)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_v1():
    xs  = np.linspace(-4.2, 4.8, 900)
    tau = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "MELU  —  Version 1: Base (no Student-t CDF)\n"
        r"Inlier: $f(h)_i = h_i$   "
        r"Outlier: $f(h)_i = h_i + \alpha \cdot \mathrm{sign}(h_i)"
        r"\cdot (e^{\beta(m-\tau)}-1)$",
        fontsize=11, y=1.02, color="#1a1a1a")

    # ── Panel A: shape at different m ──────────────────────────────────────
    ax = axes[0]; style_ax(ax)
    for fn, lbl, c, ls in [
        (relu,  "ReLU",  PALETTE["ReLU"],  "--"),
        (elu,   "ELU",   PALETTE["ELU"],   ":"),
        (swish, "Swish", PALETTE["Swish"], "-."),
    ]:
        ax.plot(xs, fn(xs), color=c, lw=1.2, ls=ls, label=lbl, alpha=0.6)

    for m_val, lbl, c in [
        (0.6,  "m=0.6  (inlier)",   "#9FE1CB"),
        (1.5,  "m=τ  (boundary)",   "#1D9E75"),
        (2.5,  "m=2.5 (outlier)",   "#085041"),
    ]:
        ax.plot(xs, melu_v1(xs, tau, m=m_val), color=c, lw=2.4, label=lbl)

    add_threshold_band(ax, tau, -3, 9)
    ax.set_xlim(-4.2, 4.8); ax.set_ylim(-3, 9)
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("f(x)", fontsize=9)
    ax.set_title("Activation shape\n(at three m values)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ax.text(1.7, 8.3, f"τ = {tau}", fontsize=8, color="#534AB7")

    # ── Panel B: gradient ──────────────────────────────────────────────────
    ax = axes[1]; style_ax(ax)
    for m_val, lbl, c in [
        (0.6,  "m=0.6",  "#9FE1CB"),
        (1.5,  "m=τ",    "#1D9E75"),
        (2.5,  "m=2.5",  "#085041"),
    ]:
        g = numerical_grad(melu_v1, xs, tau=tau, m=m_val)
        ax.plot(xs, np.clip(g, -0.5, 9), color=c, lw=2.4, label=lbl)

    ax.axhline(1, color="#888780", lw=1.0, ls="--", alpha=0.6, label="grad = 1")
    add_threshold_band(ax, tau, -0.5, 9)
    ax.set_xlim(-4.2, 4.8); ax.set_ylim(-0.5, 9)
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("f'(x)", fontsize=9)
    ax.set_title("Gradient  (clipped at 9)\nNote: always 1 for inliers", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel C: negative-zone weakness ───────────────────────────────────
    ax = axes[2]; style_ax(ax)
    xn = np.linspace(-4.2, 0.3, 400)
    for fn, lbl, c, ls in [
        (swish, "Swish", PALETTE["Swish"], "--"),
        (elu,   "ELU",   PALETTE["ELU"],   ":"),
    ]:
        ax.plot(xn, fn(xn), color=c, lw=1.3, ls=ls, label=lbl, alpha=0.7)

    ax.plot(xn, melu_v1(xn, tau, m=0.5), color="#9FE1CB", lw=2.4, label="MELU-v1 inlier")
    ax.plot(xn, melu_v1(xn, tau, m=2.5), color="#085041", lw=2.4, label="MELU-v1 outlier")

    ax.axhline(0, color="gray", lw=0.4)
    ax.fill_betweenx([-4.5, 0.5], -4.2, 0, alpha=0.04, color="#D85A30")
    ax.set_xlim(-4.2, 0.3); ax.set_ylim(-4.5, 0.5)
    ax.set_xlabel("x  (negative inputs only)", fontsize=9)
    ax.set_ylabel("f(x)", fontsize=9)
    ax.set_title("Negative region zoom\nProblem: f(x)=x everywhere (no shaping)", fontsize=10)
    ax.legend(fontsize=8)
    ax.text(-4.0, -3.8,
            "Identity line f(x)=x\n"
            "gradient = 1 always\n"
            "→ zero nonlinear shaping\n"
            "  of negative activations",
            fontsize=8, color="#D85A30",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0", edgecolor="#F0997B", lw=0.5))

    plt.tight_layout()
    path = "outputs/melu_v1_base.png"
    plt.savefig(path, dpi=148, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Version 2  (with t-CDF)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_v2():
    xs  = np.linspace(-4.2, 4.8, 900)
    tau = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "MELU-Δt  —  Version 2: With Student-t CDF base\n"
        r"Inlier: $f(h)_i = h_i \cdot T_\nu(h_i)$   "
        r"Outlier: $f(h)_i = h_i\cdot T_\nu(h_i) + \alpha\cdot\mathrm{sign}(h_i)"
        r"\cdot(e^{\beta(m-\tau)}-1)$",
        fontsize=11, y=1.02, color="#1a1a1a")

    # ── Panel A: shape for different nu ───────────────────────────────────
    ax = axes[0]; style_ax(ax)
    for fn, lbl, c, ls in [
        (relu,  "ReLU",  PALETTE["ReLU"],  "--"),
        (elu,   "ELU",   PALETTE["ELU"],   ":"),
        (swish, "Swish", PALETTE["Swish"], "-."),
    ]:
        ax.plot(xs, fn(xs), color=c, lw=1.2, ls=ls, label=lbl, alpha=0.5)

    nu_palette = ["#E1F5EE", "#5DCAA5", "#1D9E75", "#085041"]
    for nu_val, c in zip([2, 4, 10, 30], nu_palette):
        ax.plot(xs, melu_v2(xs, tau, nu=nu_val, m=2.5),
                color=c, lw=2.3, label=f"MELU-Δt  ν={nu_val}")

    add_threshold_band(ax, tau, -3, 9)
    ax.set_xlim(-4.2, 4.8); ax.set_ylim(-3, 9)
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("f(x)", fontsize=9)
    ax.set_title("Shape for four ν values\n(m=2.5, outlier regime)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)

    # ── Panel B: T_nu vs sigmoid ───────────────────────────────────────────
    ax = axes[1]; style_ax(ax)
    sig = 1.0 / (1.0 + np.exp(-xs))
    ax.plot(xs, sig, color="#D85A30", lw=1.8, ls="--", label="sigmoid  σ(x)", alpha=0.8)
    for nu_val, c in zip([2, 4, 10], ["#085041", "#1D9E75", "#9FE1CB"]):
        ax.plot(xs, t_cdf(xs, nu_val), color=c, lw=2.2, label=f"T_ν  ν={nu_val}")

    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.axvline(0,   color="gray", lw=0.4)
    ax.set_xlim(-4.2, 4.8); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("CDF value", fontsize=9)
    ax.set_title("T_ν(x) vs sigmoid\n(the gating function in Term 1)", fontsize=10)
    ax.legend(fontsize=9)
    ax.annotate(
        "heavier tail = slower\nsaturation = stays\nactive longer",
        xy=(2.8, t_cdf(2.8, 2)), xytext=(1.5, 0.65),
        arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=0.8),
        fontsize=8, color="#085041",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1F5EE", edgecolor="#1D9E75", lw=0.5))

    # ── Panel C: Δf = v2 − v1  (what t-CDF adds) ─────────────────────────
    ax = axes[2]; style_ax(ax)
    for m_val, lbl, c in [
        (0.5,  "m=0.5  (inlier)",   "#9FE1CB"),
        (1.5,  "m=τ   (boundary)",  "#1D9E75"),
        (2.5,  "m=2.5 (outlier)",   "#085041"),
    ]:
        diff = melu_v2(xs, tau, nu=4, m=m_val) - melu_v1(xs, tau, m=m_val)
        ax.plot(xs, diff, color=c, lw=2.3, label=lbl)
        ax.fill_between(xs, 0, diff, alpha=0.08, color=c)

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("f_v2(x) − f_v1(x)", fontsize=9)
    ax.set_title("Δf = MELU-Δt minus MELU-base\n(what the t-CDF adds)", fontsize=10)
    ax.legend(fontsize=9)
    ax.text(-4.0, ax.get_ylim()[0] * 0.55 if ax.get_ylim()[0] < 0 else 0.05,
            "Negative inputs get\nnon-trivial shaping\nin Version 2",
            fontsize=8, color="#085041",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1F5EE", edgecolor="#1D9E75", lw=0.5))

    plt.tight_layout()
    path = "outputs/melu_v2_tdt.png"
    plt.savefig(path, dpi=148, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Full comparison + robustness
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_comparison():
    xs  = np.linspace(-4.2, 5.0, 900)
    tau = 1.5

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.44, wspace=0.34)

    axes = {
        "inlier"   : fig.add_subplot(gs[0, 0]),
        "outlier"  : fig.add_subplot(gs[0, 1]),
        "grad"     : fig.add_subplot(gs[0, 2]),
        "neg"      : fig.add_subplot(gs[1, 0]),
        "sat"      : fig.add_subplot(gs[1, 1]),
        "table"    : fig.add_subplot(gs[1, 2]),
        "stability": fig.add_subplot(gs[2, 0]),
        "sep"      : fig.add_subplot(gs[2, 1]),
        "mcd"      : fig.add_subplot(gs[2, 2]),
        "ema"      : fig.add_subplot(gs[3, :]),
    }
    for ax in axes.values():
        style_ax(ax)

    baselines = [
        (relu,  "ReLU",  PALETTE["ReLU"],  "--"),
        (elu,   "ELU",   PALETTE["ELU"],   ":"),
        (swish, "Swish", PALETTE["Swish"], "-."),
        (gelu,  "GELU",  PALETTE["GELU"],  (0,(3,1,1,1))),
    ]

    # ── [0,0] Inlier regime ───────────────────────────────────────────────
    ax = axes["inlier"]
    ax.set_title("Inlier regime  (m=0.6 < τ=1.5)", fontsize=10)
    for fn, lbl, c, ls in baselines:
        ax.plot(xs, fn(xs), color=c, lw=1.2, ls=ls, label=lbl, alpha=0.6)
    ax.plot(xs, melu_v1(xs, tau, m=0.6), color=PALETTE["MELU-base"],
            lw=2.4, label="MELU-v1")
    ax.plot(xs, melu_v2(xs, tau, nu=4, m=0.6), color=PALETTE["MELU-Δt"],
            lw=2.4, ls="--", label="MELU-Δt")
    add_threshold_band(ax, tau, -3, 7)
    ax.set_xlim(-4.2, 5); ax.set_ylim(-3, 7)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)

    # ── [0,1] Outlier regime ──────────────────────────────────────────────
    ax = axes["outlier"]
    ax.set_title("Outlier regime  (m=2.8 > τ=1.5)", fontsize=10)
    for fn, lbl, c, ls in baselines:
        ax.plot(xs, fn(xs), color=c, lw=1.2, ls=ls, label=lbl, alpha=0.6)
    ax.plot(xs, melu_v1(xs, tau, m=2.8), color=PALETTE["MELU-base"],
            lw=2.4, label="MELU-v1")
    ax.plot(xs, melu_v2(xs, tau, nu=4, m=2.8), color=PALETTE["MELU-Δt"],
            lw=2.4, ls="--", label="MELU-Δt")
    add_threshold_band(ax, tau, -3, 10)
    ax.set_xlim(-4.2, 5); ax.set_ylim(-3, 10)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)

    # ── [0,2] Gradient (outlier) ──────────────────────────────────────────
    ax = axes["grad"]
    ax.set_title("Gradient  (m=2.8, clipped at 9)", fontsize=10)
    g1 = numerical_grad(melu_v1, xs, tau=tau, m=2.8)
    g2 = numerical_grad(melu_v2, xs, tau=tau, nu=4, m=2.8)
    ax.plot(xs, np.clip(g1, -0.5, 9), color=PALETTE["MELU-base"], lw=2.4, label="MELU-v1")
    ax.plot(xs, np.clip(g2, -0.5, 9), color=PALETTE["MELU-Δt"],   lw=2.4, ls="--", label="MELU-Δt")
    for fn, lbl, c, ls in [(swish,"Swish",PALETTE["Swish"],"-."),
                            (elu,  "ELU",  PALETTE["ELU"],  ":")]:
        ax.plot(xs, np.clip(numerical_grad(fn,xs),-0.5,9),
                color=c, lw=1.2, ls=ls, label=lbl, alpha=0.6)
    ax.axhline(1, color="gray", lw=0.7, ls=":", label="grad=1")
    ax.set_xlim(-4.2, 5); ax.set_ylim(-0.5, 9)
    ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("f'(x)")

    # ── [1,0] Negative zoom ───────────────────────────────────────────────
    ax = axes["neg"]
    xn = np.linspace(-4.2, 0.3, 400)
    ax.set_title("Negative input shaping  (x < 0 zoom)", fontsize=10)
    for fn, lbl, c, ls in baselines:
        ax.plot(xn, fn(xn), color=c, lw=1.2, ls=ls, label=lbl, alpha=0.6)
    ax.plot(xn, melu_v1(xn, tau, m=2.5), color=PALETTE["MELU-base"],
            lw=2.4, label="MELU-v1  (identity)")
    ax.plot(xn, melu_v2(xn, tau, nu=4, m=2.5), color=PALETTE["MELU-Δt"],
            lw=2.4, ls="--", label="MELU-Δt  (t-Swish)")
    ax.axhline(0, color="gray", lw=0.4)
    ax.fill_betweenx([-4.5, 0.5], -4.2, 0, alpha=0.04, color="#D85A30")
    ax.set_xlim(-4.2, 0.3); ax.set_ylim(-4.5, 0.5)
    ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.text(-4.0, -3.5,
            "v1 = straight line\nv2 = shaped by T_ν",
            fontsize=8, color="#085041",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E1F5EE", ec="#1D9E75", lw=0.5))

    # ── [1,1] Saturation test ─────────────────────────────────────────────
    ax = axes["sat"]
    xp = np.linspace(0, 5.2, 400)
    ax.set_title("Large positive inputs  (saturation)", fontsize=10)
    for fn, lbl, c, ls in baselines:
        ax.plot(xp, fn(xp), color=c, lw=1.2, ls=ls, label=lbl, alpha=0.6)
    ax.plot(xp, melu_v1(xp, tau, m=3.0), color=PALETTE["MELU-base"],
            lw=2.4, label="MELU-v1")
    ax.plot(xp, melu_v2(xp, tau, nu=4, m=3.0), color=PALETTE["MELU-Δt"],
            lw=2.4, ls="--", label="MELU-Δt")
    ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("f(x)")

    # ── [1,2] Property table ──────────────────────────────────────────────
    ax = axes["table"]; ax.axis("off")
    rows = [
        ["Property",           "MELU-v1",     "MELU-Δt"],
        ["Inlier Term 1",      "identity x",  "x · T_ν(x)"],
        ["Negative shaping",   "none",        "t-CDF gates"],
        ["Heavy-tail base",    "no",          "yes (ν param)"],
        ["Grad at x=0",        "1.0",         "0.5 (smooth)"],
        ["Recovers Swish",     "no",          "yes (ν → ∞)"],
        ["Extra learnable",    "0",           "1  (ν ≥ 2)"],
        ["Continuity at τ",    "C⁰",          "C¹"],
        ["Outlier amplifier",  "shared",      "shared"],
        ["Adaptive τ",         "EMA",         "EMA"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor("#E1F5EE")
            cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#FFF5F0")
        elif c == 2:
            cell.set_facecolor("#E8F8F2")
    ax.set_title("Property comparison", fontsize=10, pad=14)

    # ── [2,0] Gradient stability ──────────────────────────────────────────
    print("  Running robustness tests (this may take ~30s)...")
    ax = axes["stability"]
    ax.set_title("Gradient stability vs contamination", fontsize=10)
    cr, stab = run_gradient_stability(n_trials=35)
    colors_stab = {
        "MELU-Δt": PALETTE["MELU-Δt"],
        "ELU":     PALETTE["ELU"],
        "Swish":   PALETTE["Swish"],
        "ReLU":    PALETTE["ReLU"],
    }
    for name, vals in stab.items():
        lw = 2.5 if name == "MELU-Δt" else 1.4
        ls = "-"  if name == "MELU-Δt" else "--"
        ax.plot(cr * 100, vals, color=colors_stab[name], lw=lw, ls=ls, label=name)
    ax.set_xlabel("Contamination %"); ax.set_ylabel("Median max |output sensitivity|")
    ax.legend(fontsize=9)

    # ── [2,1] Outlier separation ──────────────────────────────────────────
    ax = axes["sep"]
    ax.set_title("Outlier separation power", fontsize=10)
    names, ratios = run_outlier_separation(n_trials=120)
    means = [np.mean(ratios[n]) for n in names]
    stds  = [np.std(ratios[n])  for n in names]
    cols  = [PALETTE.get(n, "#888780") for n in names]
    bars  = ax.bar(names, means, color=cols, alpha=0.85, width=0.6)
    ax.errorbar(names, means, yerr=stds, fmt="none",
                color="black", capsize=4, lw=1.1)
    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m_val + 0.04,
                f"{m_val:.2f}×", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Outlier / inlier activation ratio")

    # ── [2,2] MCD detection rate ──────────────────────────────────────────
    ax = axes["mcd"]
    ax.set_title("MCD vs standard covariance\nOutlier detection rate", fontsize=10)
    cr2, d_std, d_mcd = run_mcd_detection(n_trials=25)
    ax.plot(cr2 * 100, d_std, color=PALETTE["std"], lw=2.0, ls="--",
            label="Standard Σ")
    ax.plot(cr2 * 100, d_mcd, color=PALETTE["MCD"], lw=2.5, label="MCD Σ")
    ax.fill_between(cr2 * 100, d_std, d_mcd, alpha=0.12, color=PALETTE["MCD"])
    ax.set_xlabel("Contamination %"); ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=9)

    # ── [3, :] EMA tau adaptation trace ───────────────────────────────────
    ax = axes["ema"]
    ax.set_title("EMA τ adaptation — threshold tracks distribution shifts across training",
                 fontsize=10)
    dim   = 16
    layer = MELULayer(dim, momentum=0.99)
    true_means = []
    for i in range(160):
        # distribution shifts at batch 60 and again at batch 110
        if i < 60:
            H = np.random.randn(64, dim) * 1.0
        elif i < 110:
            H = np.random.randn(64, dim) * 2.0 + 1.5
        else:
            H = np.random.randn(64, dim) * 1.2
        layer.forward(H)
        diff = H - layer.mu_ema
        try:
            Si = np.linalg.inv(layer.cov_ema + 1e-5 * np.eye(dim))
        except Exception:
            Si = np.eye(dim)
        d2 = np.einsum('bi,ij,bj->b', diff, Si, diff)
        true_means.append(np.sqrt(np.maximum(d2, 0)).mean())

    batches = np.arange(len(layer.tau_history))
    ax.plot(batches, layer.tau_history, color=PALETTE["MELU-Δt"],
            lw=2.8, label="τ  (EMA)")
    ax.plot(np.arange(len(true_means)), true_means, color="#888780",
            lw=1.0, alpha=0.65, label="true batch mean(d_M)")
    for vx, lbl in [(60, "shift 1: σ×2"), (110, "shift 2: back")]:
        ax.axvline(vx, color=PALETTE["std"], lw=1.2, ls="--", alpha=0.7)
        ax.text(vx + 1.5, max(layer.tau_history) * 0.9,
                lbl, fontsize=8, color=PALETTE["std"])
    ax.set_xlabel("Training batch"); ax.set_ylabel("τ value")
    ax.legend(fontsize=9)

    fig.suptitle(
        "MELU  vs  MELU-Δt  vs  Baselines  —  Full Analysis & Robustness",
        fontsize=13, fontweight="500", y=1.002, color="#1a1a1a")

    path = "outputs/melu_comparison.png"
    plt.savefig(path, dpi=148, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nMELU Activation Analysis")
    print("=" * 42)
    print("Generating Figure 1 — MELU v1 (base)...")
    make_figure_v1()
    print("Generating Figure 2 — MELU-Δt (t-CDF)...")
    make_figure_v2()
    print("Generating Figure 3 — Full comparison + robustness...")
    make_figure_comparison()
    print("\nAll done. Check the outputs/ folder.")
