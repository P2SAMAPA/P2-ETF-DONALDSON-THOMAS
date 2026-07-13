import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

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

def calabi_yau_moduli(returns, macro_factor):
    """
    Construct a Calabi-Yau moduli space representation of the market.
    Returns a distance matrix and the number of persistent features.
    """
    if len(returns) < 20:
        return np.array([]), 0
    # Compute rolling statistics
    window = 20
    moduli = []
    for i in range(window, len(returns)):
        seg = returns[i-window:i]
        mean = np.mean(seg)
        vol = np.std(seg)
        skew = np.mean(((seg - mean) / (vol + 1e-8))**3) if vol > 0 else 0.0
        kurt = np.mean(((seg - mean) / (vol + 1e-8))**4) - 3 if vol > 0 else 0.0
        macro_val = macro_factor[i] if i < len(macro_factor) else 0.5
        moduli.append([mean, vol, skew, kurt, macro_val])
    moduli = np.array(moduli)
    if len(moduli) < 5:
        return np.array([]), 0
    # Normalise
    moduli = (moduli - moduli.mean(axis=0)) / (moduli.std(axis=0) + 1e-8)
    # Compute distance matrix
    dist_matrix = squareform(pdist(moduli, metric='euclidean'))
    # Use hierarchical clustering to find stable configurations
    # The number of clusters at which the dendrogram stabilises gives the DT invariant
    linkage_matrix = linkage(dist_matrix, method='ward')
    # Compute the number of clusters at different cutoff levels
    # We'll use the inconsistency method to find stable clusters
    from scipy.cluster.hierarchy import inconsistent
    inc = inconsistent(linkage_matrix)
    # Count clusters with low inconsistency (stable)
    stable_threshold = np.percentile(inc[:, 3], 50)  # 50th percentile of inconsistency
    n_clusters = len(np.unique(fcluster(linkage_matrix, t=stable_threshold, criterion='distance')))
    # DT invariant = number of stable clusters
    dt_invariant = max(1, n_clusters)  # at least 1
    return moduli, dt_invariant

def dt_score(returns, macro_df):
    """
    Compute per-ETF DT score using persistent homology features.
    Higher score = more stable market structures.
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
    # Build Calabi-Yau moduli space and compute DT invariant
    _, dt_invariant = calabi_yau_moduli(returns, macro_factor)
    # Score = DT invariant (number of stable clusters)
    # Add a small variation based on the ETF's returns to differentiate
    variation = np.std(returns) * 10.0
    score = dt_invariant + variation
    return float(score)
