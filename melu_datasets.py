"""
MELU-Δt — Dataset Generator & Analysis Suite
=============================================
Generates diverse synthetic datasets and runs the full MELU analysis on each.

Datasets available:
  1. gaussian_clean       — standard multivariate Gaussian, no outliers
  2. gaussian_contaminated— Gaussian with random outlier injection
  3. heavy_tailed         — Student-t distributed (naturally heavy tails)
  4. clustered            — multiple inlier clusters + outliers between them
  5. correlated           — high inter-feature correlation structure
  6. anisotropic          — very different variance per dimension
  7. time_series          — sequential drift + spike outliers
  8. mixed_type           — mix of normal + uniform + Laplace features
  9. high_dim             — high-dimensional (dim=64) sparse outliers
 10. adversarial          — outliers placed close to inlier boundary (hardest)

Run:
    pip install numpy scipy matplotlib
    python melu_datasets.py

    # Or import and use programmatically:
    from melu_datasets import DatasetFactory, run_dataset_analysis
    ds = DatasetFactory.make("clustered", n=300, dim=8, contamination=0.10)
    run_dataset_analysis(ds, save_prefix="my_experiment")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import betainc
from scipy.stats import chi2, laplace
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Activation functions  (self-contained, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

def t_cdf(x, nu):
    nu = max(float(nu), 2.0)
    x  = np.asarray(x, dtype=float)
    z  = nu / (nu + np.clip(x**2, 1e-30, None))
    ib = betainc(nu/2, 0.5, np.clip(z, 1e-12, 1-1e-12))
    return np.where(x >= 0, 1.0 - ib/2.0, ib/2.0)

def melu_v1(x, tau, alpha=1.0, beta=0.4, m=None):
    x   = np.asarray(x, dtype=float)
    m_  = np.abs(x) if m is None else np.full_like(x, float(m))
    amp = alpha * np.sign(x) * (np.exp(beta * np.maximum(m_ - tau, 0)) - 1)
    return x + (m_ >= tau).astype(float) * amp

def melu_v2(x, tau, alpha=1.0, beta=0.4, nu=4, m=None):
    x   = np.asarray(x, dtype=float)
    m_  = np.abs(x) if m is None else np.full_like(x, float(m))
    t1  = x * t_cdf(x, nu)
    amp = alpha * np.sign(x) * (np.exp(beta * np.maximum(m_ - tau, 0)) - 1)
    return t1 + (m_ >= tau).astype(float) * amp

def relu(x):  return np.maximum(0.0, x)
def elu(x):   return np.where(x>=0, x, np.exp(x)-1)
def swish(x): return x / (1.0 + np.exp(-x))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Stateful MELU layer with EMA
# ══════════════════════════════════════════════════════════════════════════════

class MELULayer:
    def __init__(self, dim, alpha=1.0, beta=0.4, nu=4, momentum=0.99):
        self.dim      = dim
        self.alpha    = alpha
        self.beta     = beta
        self.nu       = max(float(nu), 2.0)
        self.momentum = momentum
        self.mu_ema   = None
        self.cov_ema  = None
        self.tau_ema  = 1.0

    def _mahal(self, H):
        diff = H - self.mu_ema
        try:
            Si = np.linalg.inv(self.cov_ema + 1e-5 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            Si = np.eye(self.dim)
        return np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', diff, Si, diff), 0))

    def _update(self, H):
        m    = self.momentum
        mu_b = H.mean(0)
        d    = H - mu_b
        cv_b = (d.T @ d) / max(len(H)-1, 1)
        if self.mu_ema is None:
            self.mu_ema, self.cov_ema = mu_b.copy(), cv_b.copy()
        else:
            self.mu_ema  = m*self.mu_ema  + (1-m)*mu_b
            self.cov_ema = m*self.cov_ema + (1-m)*cv_b
        self.tau_ema = m*self.tau_ema + (1-m)*self._mahal(H).mean()

    def forward(self, H, training=True):
        if training: self._update(H)
        dists = self._mahal(H)
        T1    = H * t_cdf(H, self.nu)
        gate  = (dists >= self.tau_ema).astype(float)[:, None]
        exp_m = np.clip(self.beta*(dists - self.tau_ema), -20, 20)
        amp   = self.alpha * np.sign(H) * (np.exp(exp_m[:, None]) - 1)
        return T1 + gate * amp

    def warmup(self, n_batches=15, batch_size=64):
        for _ in range(n_batches):
            self.forward(np.random.randn(batch_size, self.dim))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Dataset factory
# ══════════════════════════════════════════════════════════════════════════════

class DatasetFactory:
    """
    Factory for generating diverse synthetic datasets.
    Every dataset is a dict with keys:
        X          : ndarray [n, dim]   — all samples
        y          : ndarray [n]        — 0=inlier, 1=outlier
        name       : str
        description: str
        dim        : int
        n_inliers  : int
        n_outliers : int
        true_cov   : ndarray [dim,dim]  (may be None for non-Gaussian)
    """

    @staticmethod
    def make(kind, n=300, dim=8, contamination=0.10, seed=42, **kwargs):
        """
        Parameters
        ----------
        kind          : str   — dataset type (see module docstring)
        n             : int   — total number of samples
        dim           : int   — feature dimensionality
        contamination : float — fraction of outliers  [0, 0.40]
        seed          : int   — random seed
        **kwargs             — dataset-specific overrides
        """
        np.random.seed(seed)
        fn = getattr(DatasetFactory, f"_{kind}", None)
        if fn is None:
            raise ValueError(
                f"Unknown dataset '{kind}'. Choose from:\n  " +
                "\n  ".join(DatasetFactory.list_kinds()))
        return fn(n=n, dim=dim, contamination=contamination, **kwargs)

    @staticmethod
    def list_kinds():
        return [k[1:] for k in dir(DatasetFactory)
                if k.startswith("_") and not k.startswith("__")]

    # ── 1. Gaussian clean ────────────────────────────────────────────────────
    @staticmethod
    def _gaussian_clean(n, dim, contamination, **kw):
        mu  = np.zeros(dim)
        cov = np.eye(dim)
        X   = np.random.multivariate_normal(mu, cov, n)
        y   = np.zeros(n, dtype=int)
        return dict(X=X, y=y, name="gaussian_clean",
                    description="Standard Gaussian — no outliers. "
                                "Baseline: MELU should pass through cleanly.",
                    dim=dim, n_inliers=n, n_outliers=0, true_cov=cov)

    # ── 2. Gaussian contaminated ─────────────────────────────────────────────
    @staticmethod
    def _gaussian_contaminated(n, dim, contamination, **kw):
        n_out = max(1, int(n * contamination))
        n_in  = n - n_out
        cov   = np.eye(dim)
        X_in  = np.random.multivariate_normal(np.zeros(dim), cov, n_in)
        # outliers uniformly scattered far from center
        X_out = (np.random.rand(n_out, dim) - 0.5) * 12 + \
                np.random.choice([-1,1], size=(n_out,dim)) * 4
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="gaussian_contaminated",
                    description=f"Gaussian inliers + {contamination*100:.0f}% "
                                "random outliers scattered uniformly far away.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=cov)

    # ── 3. Heavy-tailed ──────────────────────────────────────────────────────
    @staticmethod
    def _heavy_tailed(n, dim, contamination, nu=3, **kw):
        """Student-t distributed data — naturally produces extreme values."""
        n_out = max(1, int(n * contamination))
        n_in  = n - n_out
        # t-distributed inliers: many near-zero, occasional large values
        from scipy.stats import t as t_dist
        X_in  = t_dist.rvs(df=nu, size=(n_in, dim))
        # inject deliberate outliers even further out
        X_out = t_dist.rvs(df=1.5, size=(n_out, dim)) * 6
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="heavy_tailed",
                    description=f"Student-t(ν={nu}) inliers — naturally extreme "
                                "values. Tests heavy-tail robustness.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=None)

    # ── 4. Clustered ─────────────────────────────────────────────────────────
    @staticmethod
    def _clustered(n, dim, contamination, n_clusters=3, **kw):
        """Multiple inlier clusters. Outliers placed between clusters."""
        n_out = max(1, int(n * contamination))
        n_in  = n - n_out
        # cluster centers spread across hypercube
        centers = np.random.randn(n_clusters, dim) * 4
        sizes   = np.random.multinomial(n_in, [1/n_clusters]*n_clusters)
        X_in    = np.vstack([
            np.random.randn(sz, dim) * 0.8 + centers[i]
            for i, sz in enumerate(sizes)])
        # outliers = midpoints between cluster pairs (confusing)
        X_out   = np.zeros((n_out, dim))
        for i in range(n_out):
            c1, c2 = np.random.choice(n_clusters, 2, replace=False)
            t_     = np.random.rand()
            X_out[i] = t_*centers[c1] + (1-t_)*centers[c2] + \
                       np.random.randn(dim)*0.5
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="clustered",
                    description=f"{n_clusters} inlier clusters. Outliers placed "
                                "between clusters — hardest for centroid methods.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=None)

    # ── 5. Correlated ────────────────────────────────────────────────────────
    @staticmethod
    def _correlated(n, dim, contamination, rho=0.75, **kw):
        """Strong inter-feature correlations. Mahalanobis crucial here."""
        n_out = max(1, int(n * contamination))
        n_in  = n - n_out
        # AR(1) covariance: cov[i,j] = rho^|i-j|
        cov = np.array([[rho**abs(i-j) for j in range(dim)]
                        for i in range(dim)])
        X_in  = np.random.multivariate_normal(np.zeros(dim), cov, n_in)
        # outliers: large deviations AGAINST the correlation direction
        L     = np.linalg.cholesky(cov)
        X_out = np.zeros((n_out, dim))
        for i in range(n_out):
            z        = np.random.randn(dim) * 3
            z[::2]  *= -1          # flip every other dim → anti-correlated
            X_out[i] = L @ z
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="correlated",
                    description=f"AR(1) covariance (ρ={rho}). Outliers deviate "
                                "against correlation — Euclidean fails here.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=cov)

    # ── 6. Anisotropic ───────────────────────────────────────────────────────
    @staticmethod
    def _anisotropic(n, dim, contamination, **kw):
        """Very different variance per dimension."""
        n_out  = max(1, int(n * contamination))
        n_in   = n - n_out
        scales = np.logspace(-1, 1, dim)      # σ: 0.1 → 10
        cov    = np.diag(scales**2)
        X_in   = np.random.multivariate_normal(np.zeros(dim), cov, n_in)
        # outliers: large in LOW-variance dims (most surprising)
        X_out  = np.zeros((n_out, dim))
        for i in range(n_out):
            x           = np.random.randn(dim) * scales
            low_var_dim = np.argmin(scales)
            x[low_var_dim] += np.random.choice([-1,1]) * scales[low_var_dim]*5
            X_out[i] = x
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="anisotropic",
                    description="Variance spans 3 orders of magnitude across dims. "
                                "Outliers are extreme in low-variance dimensions.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=cov)

    # ── 7. Time series ───────────────────────────────────────────────────────
    @staticmethod
    def _time_series(n, dim, contamination, drift=0.02, **kw):
        """Gradual drift + sudden spike outliers."""
        n_out = max(1, int(n * contamination))
        n_in  = n - n_out
        # inliers: slow drift in mean
        t     = np.arange(n_in)
        drift_vec = drift * np.outer(t, np.ones(dim))
        X_in  = np.random.randn(n_in, dim) * 0.8 + drift_vec
        # outliers: sudden spikes (random time, random dims)
        X_out = np.zeros((n_out, dim))
        for i in range(n_out):
            t_spike       = np.random.randint(0, n_in)
            base          = drift_vec[t_spike] + np.random.randn(dim)*0.8
            spike_dims    = np.random.choice(dim, size=max(1,dim//3), replace=False)
            base[spike_dims] += np.random.choice([-1,1], len(spike_dims)) * 5
            X_out[i] = base
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="time_series",
                    description=f"Gradual mean drift (δ={drift}/step) + "
                                "sudden spike outliers in random dimensions.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=None)

    # ── 8. Mixed type ────────────────────────────────────────────────────────
    @staticmethod
    def _mixed_type(n, dim, contamination, **kw):
        """Features from different distributions: Gaussian + Uniform + Laplace."""
        n_out  = max(1, int(n * contamination))
        n_in   = n - n_out
        g      = max(1, dim // 3)           # Gaussian dims
        u      = max(1, dim // 3)           # Uniform dims
        lp     = dim - g - u                # Laplace dims
        X_in   = np.hstack([
            np.random.randn(n_in, g),
            np.random.uniform(-1.7, 1.7, (n_in, u)),
            laplace.rvs(size=(n_in, lp))
        ])
        X_out  = np.hstack([
            np.random.randn(n_out, g) * 4,
            np.random.uniform(-8, 8, (n_out, u)),
            laplace.rvs(size=(n_out, lp)) * 6
        ])
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="mixed_type",
                    description=f"Features: {g} Gaussian + {u} Uniform + "
                                f"{lp} Laplace. Tests distribution mismatch.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=None)

    # ── 9. High dimensional ──────────────────────────────────────────────────
    @staticmethod
    def _high_dim(n, dim, contamination, **kw):
        """High-dim Gaussian where outliers are sparse (few dims deviate)."""
        dim    = max(dim, 32)           # override dim if too small
        n_out  = max(1, int(n * contamination))
        n_in   = n - n_out
        cov    = np.eye(dim)
        X_in   = np.random.multivariate_normal(np.zeros(dim), cov, n_in)
        # outliers: only 2-3 dims are extreme, rest normal → hard for Euclidean
        X_out  = np.random.randn(n_out, dim)
        for i in range(n_out):
            hot = np.random.choice(dim, size=3, replace=False)
            X_out[i, hot] += np.random.choice([-1,1], 3) * 6
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="high_dim",
                    description=f"dim={dim}. Outliers deviate in only 3 dims — "
                                "dense distance metrics diluted by noise dims.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=cov)

    # ── 10. Adversarial ──────────────────────────────────────────────────────
    @staticmethod
    def _adversarial(n, dim, contamination, **kw):
        """Outliers placed just outside the inlier boundary — hardest case."""
        n_out = max(1, int(n * contamination))
        n_in  = n - n_out
        cov   = np.eye(dim)
        X_in  = np.random.multivariate_normal(np.zeros(dim), cov, n_in)
        # chi2 95th percentile radius for dim dimensions
        boundary_r = np.sqrt(chi2.ppf(0.95, df=dim))
        X_out = np.zeros((n_out, dim))
        for i in range(n_out):
            direction  = np.random.randn(dim)
            direction /= np.linalg.norm(direction)
            # place just beyond the 95% boundary
            r          = boundary_r * (1.05 + np.random.rand()*0.3)
            X_out[i]   = direction * r
        X = np.vstack([X_in, X_out])
        y = np.array([0]*n_in + [1]*n_out)
        return dict(X=X, y=y, name="adversarial",
                    description="Outliers sit just outside the 95% inlier ellipse. "
                                "Hardest: small margin, no easy separation.",
                    dim=dim, n_inliers=n_in, n_outliers=n_out, true_cov=cov)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Per-dataset analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_mahal(X, mu, cov):
    diff = X - mu
    try:
        Si = np.linalg.inv(cov + 1e-5*np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        Si = np.eye(cov.shape[0])
    d2 = np.einsum('bi,ij,bj->b', diff, Si, diff)
    return np.sqrt(np.maximum(d2, 0))


def activation_scores(X, y, layer):
    """Pass X through a warmed-up MELU layer, return output norms."""
    out = layer.forward(X, training=False)
    return np.linalg.norm(out, axis=1)


def detection_metrics(scores, y, percentile=85):
    """Threshold at given percentile of scores, return precision/recall/F1."""
    thr = np.percentile(scores, percentile)
    pred = (scores > thr).astype(int)
    tp = ((pred==1) & (y==1)).sum()
    fp = ((pred==1) & (y==0)).sum()
    fn = ((pred==0) & (y==1)).sum()
    precision = tp / max(tp+fp, 1)
    recall    = tp / max(tp+fn, 1)
    f1        = 2*precision*recall / max(precision+recall, 1e-8)
    return dict(precision=precision, recall=recall, f1=f1, threshold=thr)


def run_dataset_analysis(ds, save_prefix=None, show=False):
    """
    Full analysis for one dataset dict (as returned by DatasetFactory.make).
    Produces a 3×3 figure:
      Row 0: data scatter (first 2 dims) | Mahalanobis dist histogram | EMA tau trace
      Row 1: MELU-v1 scores | MELU-Δt scores | activation norm comparison
      Row 2: detection metrics bar | hyperparameter sensitivity | property table
    """
    X, y   = ds["X"], ds["y"]
    dim    = ds["dim"]
    name   = ds["name"]
    desc   = ds["description"]
    n_in   = ds["n_inliers"]
    n_out  = ds["n_outliers"]

    # ── fit stats ─────────────────────────────────────────────────────────────
    X_in  = X[y==0]
    mu    = X_in.mean(0)
    d_    = X_in - mu
    cov   = (d_.T @ d_) / max(len(X_in)-1, 1) + 1e-5*np.eye(dim)
    d_M   = compute_mahal(X, mu, cov)
    tau   = d_M[y==0].mean()

    # ── build layers ──────────────────────────────────────────────────────────
    layer_v1 = MELULayer(dim, alpha=1.0, beta=0.4, nu=4, momentum=0.95)
    layer_v2 = MELULayer(dim, alpha=1.0, beta=0.4, nu=4, momentum=0.95)
    # warm up on clean inliers
    for _ in range(20):
        idx = np.random.choice(n_in, min(64, n_in), replace=False)
        layer_v1.forward(X_in[idx])
        layer_v2.forward(X_in[idx])

    # compute scores
    tau_v1  = layer_v1.tau_ema
    T1_v1   = X                                                       # identity base
    amp_v1  = 1.0 * np.sign(X) * (np.exp(0.4 * np.maximum(
                  d_M[:, None] - tau_v1, 0)) - 1)
    gate_v1 = (d_M >= tau_v1).astype(float)[:, None]
    out_v1  = np.linalg.norm(T1_v1 + gate_v1 * amp_v1, axis=1)
    out_v2 = layer_v2.forward(X, training=False)
    scores_v2 = np.linalg.norm(out_v2, axis=1)

    # baseline scores (Euclidean norm after activation)
    scores_elu   = np.linalg.norm(elu(X),   axis=1)
    scores_swish = np.linalg.norm(swish(X), axis=1)
    scores_relu  = np.linalg.norm(relu(X),  axis=1)

    m_v1 = detection_metrics(out_v1,    y)
    m_v2 = detection_metrics(scores_v2, y)
    m_eu = detection_metrics(scores_elu,   y)
    m_sw = detection_metrics(scores_swish, y)
    m_rl = detection_metrics(scores_relu,  y)

    # EMA trace on full dataset
    layer_trace = MELULayer(dim, momentum=0.99)
    tau_hist    = []
    batch_size  = 32
    for i in range(0, len(X_in), batch_size):
        layer_trace.forward(X_in[i:i+batch_size])
        tau_hist.append(layer_trace.tau_ema)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.34)

    COLORS = {"inlier":"#1D9E75","outlier":"#D85A30",
              "v1":"#D85A30","v2":"#1D9E75",
              "ELU":"#7F77DD","Swish":"#BA7517","ReLU":"#888780"}

    def sax(r, c):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#CCCCCC")
        ax.tick_params(labelsize=8)
        return ax

    # ── [0,0] scatter (dims 0 and 1) ─────────────────────────────────────────
    ax = sax(0,0)
    ax.scatter(X[y==0,0], X[y==0,1], s=12, alpha=0.45,
               color=COLORS["inlier"], label=f"inliers  ({n_in})")
    if n_out > 0:
        ax.scatter(X[y==1,0], X[y==1,1], s=30, alpha=0.85,
                   color=COLORS["outlier"], marker="x", lw=1.2,
                   label=f"outliers ({n_out})")
    ax.set_title("Data scatter  (dims 0 & 1)", fontsize=10)
    ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
    ax.legend(fontsize=8)

    # ── [0,1] Mahalanobis histogram ───────────────────────────────────────────
    ax = sax(0,1)
    bins = np.linspace(0, d_M.max()*1.05, 35)
    ax.hist(d_M[y==0], bins=bins, alpha=0.65, color=COLORS["inlier"],
            label="inliers",  density=True)
    if n_out > 0:
        ax.hist(d_M[y==1], bins=bins, alpha=0.65, color=COLORS["outlier"],
                label="outliers", density=True)
    ax.axvline(tau, color="#534AB7", lw=1.5, ls="--",
               label=f"τ = {tau:.2f}")
    thr_chi = np.sqrt(chi2.ppf(0.95, df=min(dim,4)))
    ax.axvline(thr_chi, color="#BA7517", lw=1.2, ls=":",
               label=f"χ² 95%  ({thr_chi:.2f})")
    ax.set_title("Mahalanobis distance distribution", fontsize=10)
    ax.set_xlabel("d_M"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # ── [0,2] EMA tau trace ───────────────────────────────────────────────────
    ax = sax(0,2)
    ax.plot(tau_hist, color="#1D9E75", lw=2.2, label="τ (EMA)")
    ax.axhline(tau, color="#534AB7", lw=1.2, ls="--",
               label=f"true mean(d_M) = {tau:.2f}")
    ax.set_title("EMA τ adaptation on this dataset", fontsize=10)
    ax.set_xlabel("batch"); ax.set_ylabel("τ")
    ax.legend(fontsize=8)

    # ── [1,0] MELU-v1 score histogram ────────────────────────────────────────
    ax = sax(1,0)
    sc_bins = np.linspace(0, max(out_v1.max(), 0.1)*1.05, 35)
    ax.hist(out_v1[y==0], bins=sc_bins, alpha=0.65, color=COLORS["inlier"],
            label="inliers",  density=True)
    if n_out > 0:
        ax.hist(out_v1[y==1], bins=sc_bins, alpha=0.65, color=COLORS["outlier"],
                label="outliers", density=True)
    ax.axvline(m_v1["threshold"], color="#534AB7", lw=1.2, ls="--",
               label=f"thr  F1={m_v1['f1']:.2f}")
    ax.set_title(f"MELU-v1 scores  (F1={m_v1['f1']:.2f})", fontsize=10)
    ax.set_xlabel("||f(X)||"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # ── [1,1] MELU-Δt score histogram ────────────────────────────────────────
    ax = sax(1,1)
    sc_bins2 = np.linspace(0, max(scores_v2.max(), 0.1)*1.05, 35)
    ax.hist(scores_v2[y==0], bins=sc_bins2, alpha=0.65, color=COLORS["inlier"],
            label="inliers",  density=True)
    if n_out > 0:
        ax.hist(scores_v2[y==1], bins=sc_bins2, alpha=0.65,
                color=COLORS["outlier"], label="outliers", density=True)
    ax.axvline(m_v2["threshold"], color="#534AB7", lw=1.2, ls="--",
               label=f"thr  F1={m_v2['f1']:.2f}")
    ax.set_title(f"MELU-Δt scores  (F1={m_v2['f1']:.2f})", fontsize=10)
    ax.set_xlabel("||f(X)||"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # ── [1,2] activation norm comparison box ─────────────────────────────────
    ax = sax(1,2)
    names_   = ["MELU-v1","MELU-Δt","ELU","Swish","ReLU"]
    scores_  = [out_v1,   scores_v2, scores_elu, scores_swish, scores_relu]
    cols_    = [COLORS["v1"],COLORS["v2"],
                COLORS["ELU"],COLORS["Swish"],COLORS["ReLU"]]
    if n_out > 0:
        ratios = [s[y==1].mean()/(s[y==0].mean()+1e-8) for s in scores_]
        bars   = ax.bar(names_, ratios, color=cols_, alpha=0.85, width=0.6)
        for b, r in zip(bars, ratios):
            ax.text(b.get_x()+b.get_width()/2, r+0.02,
                    f"{r:.2f}×", ha="center", va="bottom", fontsize=9)
        ax.axhline(1, color="gray", lw=0.8, ls="--")
        ax.set_title("Outlier / inlier score ratio\n(higher = better separation)",
                     fontsize=10)
        ax.set_ylabel("ratio")
    else:
        ax.text(0.5, 0.5, "No outliers\nin this dataset",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Outlier / inlier ratio", fontsize=10)

    # ── [2,0] F1 / precision / recall bar ────────────────────────────────────
    ax = sax(2,0)
    method_names = ["MELU-v1","MELU-Δt","ELU","Swish","ReLU"]
    metrics_list = [m_v1, m_v2, m_eu, m_sw, m_rl]
    x_pos = np.arange(len(method_names))
    w     = 0.25
    ax.bar(x_pos-w, [m["precision"] for m in metrics_list],
           width=w, label="Precision", color="#9FE1CB", alpha=0.85)
    ax.bar(x_pos,   [m["recall"]    for m in metrics_list],
           width=w, label="Recall",    color="#1D9E75", alpha=0.85)
    ax.bar(x_pos+w, [m["f1"]        for m in metrics_list],
           width=w, label="F1",        color="#085041", alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(method_names, fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.set_title("Detection metrics  (threshold @ 85th pct)", fontsize=10)
    for xp, m in zip(x_pos, metrics_list):
        ax.text(xp+w, m["f1"]+0.03, f"{m['f1']:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    # ── [2,1] beta sensitivity ────────────────────────────────────────────────
    ax = sax(2,1)
    betas   = [0.1, 0.3, 0.6, 1.0, 1.5]
    f1s     = []
    palette = plt.cm.YlGn(np.linspace(0.3, 0.95, len(betas)))
    for beta_val, c in zip(betas, palette):
        lyr = MELULayer(dim, beta=beta_val, momentum=0.95)
        for _ in range(15):
            idx = np.random.choice(n_in, min(64,n_in), replace=False)
            lyr.forward(X_in[idx])
        sc = np.linalg.norm(lyr.forward(X, training=False), axis=1)
        mtr = detection_metrics(sc, y)
        f1s.append(mtr["f1"])
    ax.plot(betas, f1s, color="#1D9E75", lw=2.4, marker="o", ms=5)
    ax.set_xlabel("β  (amplification rate)"); ax.set_ylabel("F1")
    ax.set_title("F1 vs β  (MELU-Δt)", fontsize=10)
    ax.axvline(0.4, color="#D85A30", lw=1, ls="--", alpha=0.6,
               label="default β=0.4")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # ── [2,2] dataset info table ───────────────────────────────────────────────
    ax = sax(2,2); ax.axis("off")
    rows = [
        ["Property",     "Value"],
        ["Dataset",      name],
        ["Total n",      str(len(X))],
        ["Inliers",      str(n_in)],
        ["Outliers",     str(n_out)],
        ["Contamination",f"{n_out/max(len(X),1)*100:.1f}%"],
        ["Dimensions",   str(dim)],
        ["Best F1",      f"{max(m_v2['f1'],m_v1['f1']):.3f}  "
                         f"({'MELU-Δt' if m_v2['f1']>=m_v1['f1'] else 'MELU-v1'})"],
        ["τ (EMA)",      f"{layer_v2.tau_ema:.3f}"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if r==0:
            cell.set_facecolor("#E1F5EE")
            cell.set_text_props(fontweight="bold")
        elif c==1:
            cell.set_facecolor("#F5FFFE")
    ax.set_title("Dataset summary", fontsize=10, pad=14)

    # ── title and save ────────────────────────────────────────────────────────
    fig.suptitle(
        f"Dataset: {name}\n{desc}",
        fontsize=11, y=1.01, color="#1a1a1a", fontweight="500")

    prefix = save_prefix or f"outputs/dataset_{name}"
    path   = f"{prefix}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  [{name}]  F1: MELU-Δt={m_v2['f1']:.3f}  "
          f"MELU-v1={m_v1['f1']:.3f}  ELU={m_eu['f1']:.3f}"
          f"  → {path}")
    return dict(dataset=name, f1_v2=m_v2["f1"], f1_v1=m_v1["f1"],
                f1_elu=m_eu["f1"], f1_swish=m_sw["f1"], f1_relu=m_rl["f1"])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Cross-dataset summary figure
# ══════════════════════════════════════════════════════════════════════════════

def make_summary_figure(all_results, path="outputs/dataset_summary.png"):
    """
    Bar chart comparing F1 across all datasets and all methods.
    """
    names   = [r["dataset"]  for r in all_results]
    f1_v2   = [r["f1_v2"]   for r in all_results]
    f1_v1   = [r["f1_v1"]   for r in all_results]
    f1_elu  = [r["f1_elu"]  for r in all_results]
    f1_sw   = [r["f1_swish"]for r in all_results]
    f1_rl   = [r["f1_relu"] for r in all_results]

    x   = np.arange(len(names))
    w   = 0.15
    fig, ax = plt.subplots(figsize=(max(14, len(names)*1.6), 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    ax.bar(x - 2*w, f1_v2,  width=w, label="MELU-Δt",color="#1D9E75",alpha=0.9)
    ax.bar(x -   w, f1_v1,  width=w, label="MELU-v1", color="#D85A30",alpha=0.85)
    ax.bar(x,       f1_elu, width=w, label="ELU",      color="#7F77DD",alpha=0.85)
    ax.bar(x +   w, f1_sw,  width=w, label="Swish",    color="#BA7517",alpha=0.85)
    ax.bar(x + 2*w, f1_rl,  width=w, label="ReLU",     color="#888780",alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_","\n") for n in names], fontsize=9)
    ax.set_ylabel("F1 score  (threshold @ 85th percentile)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=10, ncol=5)
    ax.set_title("Cross-dataset F1 comparison — all methods", fontsize=12)

    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#CCCCCC")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"\nSummary figure saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nMELU Dataset Analysis Suite")
    print("=" * 48)
    print(f"Available datasets: {DatasetFactory.list_kinds()}\n")

    # ── configure which datasets to run ────────────────────────────────────
    RUNS = [
        dict(kind="gaussian_contaminated", n=300, dim=8,  contamination=0.10),
        dict(kind="heavy_tailed",          n=300, dim=8,  contamination=0.10),
        dict(kind="clustered",             n=400, dim=8,  contamination=0.12),
        dict(kind="correlated",            n=300, dim=8,  contamination=0.10),
        dict(kind="anisotropic",           n=300, dim=8,  contamination=0.10),
        dict(kind="time_series",           n=300, dim=8,  contamination=0.10),
        dict(kind="mixed_type",            n=300, dim=9,  contamination=0.10),
        dict(kind="high_dim",              n=400, dim=32, contamination=0.10),
        dict(kind="adversarial",           n=400, dim=8,  contamination=0.12),
    ]

    all_results = []
    print("Running per-dataset analysis...")
    for cfg in RUNS:
        ds  = DatasetFactory.make(**cfg)
        res = run_dataset_analysis(ds)
        all_results.append(res)

    make_summary_figure(all_results)

    print("\n" + "="*48)
    print("All outputs saved to outputs/")
    print("\nTo run a single custom dataset:")
    print("  from melu_datasets import DatasetFactory, run_dataset_analysis")
    print("  ds = DatasetFactory.make('clustered', n=500, dim=16, contamination=0.15)")
    print("  run_dataset_analysis(ds, save_prefix='outputs/my_run')")
