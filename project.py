import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================
def power_method(A, x0, maxit, tol):
    M = np.asarray(A, dtype=float)
    v = np.asarray(x0, dtype=float).reshape(-1)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("A must be a square (n,n) array.")
    n = M.shape[0]
    if v.size != n:
        raise ValueError("x0 must have shape (n,).")
    if maxit <= 0:
        raise ValueError("maxit must be positive.")
    if tol < 0:
        raise ValueError("tol must be nonnegative.")

    # Normalize initial vector; if near-zero, use a deterministic nonzero vector.
    v_norm = np.linalg.norm(v, ord=2)
    if v_norm < 1e-12:
        v = np.ones(n, dtype=float)
        v_norm = np.linalg.norm(v, ord=2)
    v = v / v_norm

    lam_prev = float(v @ (M @ v))
    iters = 0

    for iters in range(1, int(maxit) + 1):
        w = M @ v
        w_norm = np.linalg.norm(w, ord=2)
        if w_norm < 1e-15:
            # M maps v close to zero; dominant eigenvalue estimate goes to 0.
            lam = 0.0
            v = np.zeros_like(v)
            break

        v = w / w_norm
        lam = float(v @ (M @ v))

        denom = max(abs(lam), 1e-15)
        rel_err = abs(lam - lam_prev) / denom
        if rel_err < tol:
            break

        lam_prev = lam

    return float(lam), v, int(iters)


# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================
def svd_compress(image, k):
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array.")
    m, n = A.shape
    r = min(m, n)

    if not (1 <= int(k) <= r):
        raise ValueError(f"Provided rank out of bounds. Should be in [1, {r}]")
    k = int(k)

    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * S[:k]) @ Vh[:k, :]

    fro_A = np.linalg.norm(A, ord="fro")
    if fro_A == 0:
        rel_err = 0.0
    else:
        rel_err = float(np.linalg.norm(A - Ak, ord="fro") / fro_A)

    comp_ratio = float(k * (m + n + 1) / (m * n))
    return Ak, rel_err, comp_ratio


# =========================================================
# 3. SVD-based feature extraction
# =========================================================
def svd_features(image, p):
    """Extract SVD-based features from a grayscale image.

    Returns
    -------
    feat : (p + 2,) ndarray
        Feature vector consisting of:
        [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array.")
    m, n = A.shape
    r = min(m, n)
    if not (1 <= int(p) <= r):
        raise ValueError(f"p must be in [1, {r}]")
    p = int(p)

    # Reference logic: normalize by sum of singular values (not squared energy).
    sig = np.linalg.svd(A, compute_uv=False)
    s_sum = float(np.sum(sig))
    if s_sum <= 0.0:
        # Degenerate case: all zeros
        cumulative = np.zeros_like(sig, dtype=float)
    else:
        cumulative = np.cumsum(sig) / s_sum

    r_0_9 = int(np.argmax(cumulative >= 0.9) + 1) if cumulative.size else 1
    r_0_95 = int(np.argmax(cumulative >= 0.95) + 1) if cumulative.size else 1

    # Feature layout matches the reference: first p entries are cumulative proportions.
    return np.hstack((cumulative[:p], [r_0_9, r_0_95]))


# =========================================================
# 4. Two-class LDA: training
# =========================================================
def lda_train(X, y):
    """Train a two-class LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix (rows = samples, columns = features).
    y : (N,) ndarray
        Labels, each 0 or 1.

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction vector (not necessarily unit length).
    threshold : float
        Threshold in 1D projected space for classifying 0 vs 1.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.size != X.shape[0]:
        raise ValueError("y must have the same number of rows as X.")
    if np.unique(y).size != 2:
        raise ValueError("lda_train expects exactly two classes labeled 0 and 1.")

    XA = X[y == 0, :]
    XB = X[y == 1, :]
    if XA.shape[0] == 0 or XB.shape[0] == 0:
        raise ValueError("Both classes must contain at least one sample.")

    muA = np.mean(XA, axis=0)
    muB = np.mean(XB, axis=0)

    XA_c = XA - muA
    XB_c = XB - muB
    SW = XA_c.T @ XA_c + XB_c.T @ XB_c

    # Reference uses a fixed lambda = 1e-6; keep that.
    reg = 1e-6
    SW_reg = SW + reg * np.eye(SW.shape[0])

    w = np.linalg.solve(SW_reg, (muB - muA))

    mA = float(np.mean(XA @ w))
    mB = float(np.mean(XB @ w))
    thresh = 0.5 * (mA + mB)

    return w, float(thresh)


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================
def lda_predict(X, w, threshold):
    """Predict class labels using a trained LDA classifier.

    Returns
    -------
    y_pred : (N,) ndarray
        Predicted labels (0 or 1).
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    scores = X @ w
    return (scores >= float(threshold)).astype(int)


# =========================================================
# Simple self-test on the example data
# =========================================================
def _example_run():
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(5, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    def build_features(Ximgs):
        return np.vstack([svd_features(img, p) for img in Ximgs])

    Xf_train = build_features(X_train)
    Xf_test = build_features(X_test)

    w, threshold = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, threshold)

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    _example_run()
