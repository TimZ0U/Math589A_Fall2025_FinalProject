import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================

def power_method(A, x0, maxit, tol):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]

    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != n:
        raise ValueError("x0 must have length n")

    nx = np.linalg.norm(x)
    if nx == 0:
        x = np.ones(n, dtype=float)
        nx = np.linalg.norm(x)
    x /= nx

    lam_prev = float(x @ (A @ x))
    eps = np.finfo(float).eps

    for iters in range(1, int(maxit) + 1):
        y = A @ x
        ny = np.linalg.norm(y)

        # Keep contract: return a unit vector
        if ny == 0:
            return 0.0, x.copy(), iters

        x = y / ny
        lam = float(x @ (A @ x))

        denom = max(abs(lam_prev), eps)
        if abs(lam - lam_prev) / denom < tol:
            return lam, x, iters

        lam_prev = lam

    return lam_prev, x, int(maxit)


# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================

def svd_compress(image, k):
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    m, n = A.shape

    k = int(k)
    if k < 1 or k > min(m, n):
        raise ValueError("k must satisfy 1 <= k <= min(m, n)")

    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :k]
    Sk = S[:k]
    Vhk = Vh[:k, :]

    A_k = (Uk * Sk) @ Vhk

    denom = np.linalg.norm(A, ord="fro")
    if denom == 0:
        rel_error = 0.0 if np.linalg.norm(A_k, ord="fro") == 0 else float("inf")
    else:
        rel_error = float(np.linalg.norm(A - A_k, ord="fro") / denom)

    compression_ratio = float((k * (m + n + 1)) / (m * n))
    return A_k, rel_error, compression_ratio


# =========================================================
# 3. SVD-based feature extraction
# =========================================================

def svd_features(image, p):
    """
    Return a (p+2,) feature vector:
      [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    m, n = A.shape

    p = int(p)
    if p < 1 or p > min(m, n):
        raise ValueError("p must satisfy 1 <= p <= min(m, n)")

    s = np.linalg.svd(A, compute_uv=False)

    # Normalize top singular values (probability-like) for the first p entries
    s_sum = float(np.sum(s))
    if s_sum > 0:
        s_norm = s / s_sum
    else:
        s_norm = s  # all zeros

    # Rank features: use ENERGY capture (sigma^2) for r_0.9, r_0.95
    # This is usually what "90% / 95% captured" intends.
    energy = s * s
    e_sum = float(np.sum(energy))
    if e_sum > 0:
        c = np.cumsum(energy) / e_sum
    else:
        c = np.cumsum(energy)  # all zeros

    r_0_9 = float(np.searchsorted(c, 0.90) + 1)
    r_0_95 = float(np.searchsorted(c, 0.95) + 1)

    top_p = s_norm[:p]
    feat = np.concatenate([top_p, np.array([r_0_9, r_0_95], dtype=float)])
    return feat


# =========================================================
# 4. Two-class LDA: training
# =========================================================

def lda_train(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array of length N")

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes 0 and 1 must be present in y")

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1

    # Within-class scatter (pooled, unnormalized)
    Sw = X0c.T @ X0c + X1c.T @ X1c
    d = Sw.shape[0]

    # Regularization for stability/generalization:
    # - mild shrinkage toward diagonal (helps when features are correlated/noisy)
    # - plus a small ridge
    diag_Sw = np.diag(np.diag(Sw))
    alpha = 0.05  # small shrinkage; typically safe
    Sw_shrunk = (1.0 - alpha) * Sw + alpha * diag_Sw

    tr = float(np.trace(Sw_shrunk))
    lam = 1e-6 * (tr / d) if tr > 0 else 1e-6
    Sw_reg = Sw_shrunk + lam * np.eye(d)

    # Solve Sw_reg w = (mu1 - mu0)
    w = np.linalg.solve(Sw_reg, (mu1 - mu0))

    # Better threshold using LDA discriminant with class priors:
    # boundary at w^T x >= 0.5 (mu1+mu0)^T w - log(pi1/pi0)
    n0 = X0.shape[0]
    n1 = X1.shape[0]
    pi0 = n0 / (n0 + n1)
    pi1 = n1 / (n0 + n1)

    midpoint = 0.5 * float((mu0 + mu1) @ w)
    prior_shift = np.log(pi1 / pi0) if (pi0 > 0 and pi1 > 0) else 0.0
    threshold = midpoint - prior_shift

    return w, threshold


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================

def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    scores = X @ w
    return (scores >= threshold).astype(int)


# =========================================================
# Simple self-test on the example data
# =========================================================

def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Sanity check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(5, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    # Build feature matrices
    def build_features(X):
        feats = []
        for img in X:
            feats.append(svd_features(img, p))
        return np.vstack(feats)

    try:
        Xf_train = build_features(X_train)
        Xf_test = build_features(X_test)
    except NotImplementedError:
        print("Implement 'svd_features' first to run this example.")
        return

    print("Feature dimension:", Xf_train.shape[1])

    try:
        w, threshold = lda_train(Xf_train, y_train)
    except NotImplementedError:
        print("Implement 'lda_train' first to run this example.")
        return

    try:
        y_pred = lda_predict(Xf_test, w, threshold)
    except NotImplementedError:
        print("Implement 'lda_predict' first to run this example.")
        return

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # This allows students to run a quick local smoke test.
    _example_run()
