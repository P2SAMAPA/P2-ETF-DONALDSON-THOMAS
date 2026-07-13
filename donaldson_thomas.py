import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def calabi_yau_moduli(returns, macro_factor, n_partitions=10):
    """
    Construct a Calabi-Yau moduli space representation of the market.
    The moduli space is a discretisation of the correlation manifold.
    """
    if len(returns) < 20:
        return np.array([]), np.array([])
    # Compute rolling statistics to define the moduli
    window = 20
    moduli = []
    for i in range(window, len(returns)):
        seg = returns[i-window:i]
        mean = np.mean(seg)
        vol = np.std(seg)
        skew = np.mean(((seg - mean) / (vol + 1e-8))**3) if vol > 0 else 0.0
        kurt = np.mean(((seg - mean) / (vol + 1e-8))**4) - 3 if vol > 0 else 0.0
        # Add macro factor as a moduli parameter
        macro_val = macro_factor[i] if i < len(macro_factor) else 0.5
        moduli.append([mean, vol, skew, kurt, macro_val])
    moduli = np.array(moduli)
    if len(moduli) < 5:
        return np.array([]), np.array([])
    # Normalise moduli
    moduli = (moduli - moduli.mean(axis=0)) / (moduli.std(axis=0) + 1e-8)
    # Partition the moduli space into discrete regions (Calabi-Yau "cells")
    # Use k-means clustering to find stable configurations
    n_clusters = min(n_partitions, len(moduli) // 5)
    if n_clusters < 2:
        return moduli, np.zeros(len(moduli))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(moduli)
    return moduli, labels

def donaldson_thomas_invariant(moduli, labels, stability_threshold=0.5):
    """
    Compute the Donaldson-Thomas invariant: integer count of stable configurations.
    Stable configurations are clusters with sufficient "mass" (number of points).
    """
    if len(labels) == 0:
        return 0, {}
    # Count points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique)
    # Compute the "mass" of each cluster (fraction of total points)
    masses = counts / len(labels)
    # Stable configurations: clusters with mass > stability_threshold
    stable_clusters = [u for u, m in zip(unique, masses) if m > stability_threshold]
    # DT invariant = number of stable clusters
    dt_invariant = len(stable_clusters)
    # Per-cluster DT contributions
    cluster_dt = {int(c): int(counts[i]) for i, c in enumerate(unique)}
    return dt_invariant, cluster_dt

def dt_score(returns, macro_df, n_partitions=10, stability_threshold=0.5):
    """
    Compute per-ETF DT score.
    Higher DT invariant = more stable market structures = more coherent regime.
    """
    if len(returns) < 30 or macro_df is None or len(macro_df) < 30:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 30:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Build Calabi-Yau moduli space
    moduli, labels = calabi_yau_moduli(returns, macro_factor, n_partitions)
    if len(labels) == 0:
        return 0.0
    # Compute DT invariant
    dt_invariant, cluster_dt = donaldson_thomas_invariant(moduli, labels, stability_threshold)
    # Score = DT invariant (higher = more stable market structures)
    return float(dt_invariant)
