
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _normalize(df: pd.DataFrame, features):
    # Min-max normalize to [0,1] for fair distance computations
    eps = 1e-9
    normed = df.copy()
    for f in features:
        mn, mx = df[f].min(), df[f].max()
        if mx - mn < eps:
            normed[f] = 0.0
        else:
            normed[f] = (df[f] - mn) / (mx - mn)
    return normed

def recall_predict(
    train_df: pd.DataFrame,
    user_inputs: Dict[str, float],
    features,
    label_col: str = "Outcome",
    k: int = 25,
    match_threshold: float = 0.60
) -> Tuple[int, float, float]:
    """
    A simple ACT-R–inspired memory-based "diagnosis":
      - Normalize features
      - Compute cosine similarity to each past case (chunk)
      - Take top-k most similar
      - If >= match_threshold of neighbors are diabetic -> predict 1 (diabetic)
    Returns:
      (matches, diabetic_ratio, predicted_label)
    """
    df = train_df.dropna().reset_index(drop=True)
    X = df[features].copy()
    y = df[label_col].astype(int).values

    # Normalize train + the single user vector under the same min/max
    norm_train = _normalize(X, features)
    # Build a 1-row DataFrame for the user
    user_row = pd.DataFrame({f: [float(user_inputs[f])] for f in features})
    norm_user = user_row.copy()
    # Apply training min/max scaling
    for f in features:
        mn, mx = X[f].min(), X[f].max()
        if mx - mn < 1e-9:
            norm_user[f] = 0.0
        else:
            norm_user[f] = (user_row[f] - mn) / (mx - mn)

    u = norm_user.values.astype(float).reshape(1, -1)  # (1,d)
    T = norm_train.values.astype(float)                # (n,d)

    # Cosine similarity
    u_norm = np.linalg.norm(u) + 1e-9
    t_norm = np.linalg.norm(T, axis=1) + 1e-9
    sims = (T @ u.T).ravel() / (t_norm * u_norm)

    # top-k neighbors
    k = int(max(1, min(k, len(sims))))
    idx = np.argpartition(-sims, kth=k-1)[:k]
    neighbors_y = y[idx]
    diabetic_ratio = float(np.mean(neighbors_y==1))
    predicted_label = 1.0 if diabetic_ratio >= match_threshold else 0.0
    return k, diabetic_ratio, predicted_label
