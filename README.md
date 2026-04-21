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
high dimensions with small samples.

#### Keywords: anomaly detection, activation function, Mahalanobis distance, autoencoder, unsupervised learning, tabular data, Student-t distribution
