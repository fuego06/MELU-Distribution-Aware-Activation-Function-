# MELU-Distribution-Aware-Activation-Function-
MELU: A Distribution Aware Activation Function with Adaptive Mahalonobis Gating For Anomaly Detection

We introduce MELU-Dt (Mahalanobis ELU with Student-t gating), a novel activation function designed for
autoencoder-based unsupervised anomaly detection on tabular data. MELU-Dt extends the standard Student-t
Swish base with a Mahalanobis-distance-gated exponential amplifier that selectively increases the
reconstruction error of outlier samples without requiring any labelled anomalies. We formally prove three
properties that distinguish MELU-Dt from all existing activation functions: (P1) distribution-awareness — the
activation is strictly monotone in Mahalanobis distance; (P2) threshold convergence — the adaptive EMA
threshold converges geometrically to the true population mean; and (P3) C1 continuity — the activation and its
gradient are continuous everywhere, enabling stable gradient-based training. We further introduce a two-stage
training protocol that computes a reliable Minimum Covariance Determinant (MCD) gate on a pre-trained frozen
latent space, resolving the dimension-concentration failure mode that makes Mahalanobis gating unreliable in
high dimensions with small samples. Empirical evaluation on 33 datasets (7 real sklearn benchmarks + 26
ADBench-profile simulations, 10 seeds, 50/50 train/test split) shows that MELU-Dt achieves competitive overall
performance (Friedman p=0.0075, rank 2.91 vs 2.14 for Swish) and wins on 22/33 datasets. On datasets with
correlated Gaussian structure — the regime for which Mahalanobis gating is theoretically motivated — MELU-Dt
wins 14/18 non-trivial cases with improvements up to +0.040 AUROC.
Keywords: anomaly detection, activation function, Mahalanobis distance, autoencoder, unsupervised learning, tabular
data, Student-t distribution
