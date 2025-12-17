import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================

def power_method(A, x0, maxit, tol):
    """Approximate the dominant eigenvalue and eigenvector of a real symmetric matrix A.

    Parameters
    ----------
    A : (n, n) ndarray
        Real symmetric matrix.
    x0 : (n,) ndarray
        Initial guess for eigenvector (nonzero).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence in relative change of eigenvalue.

    Returns
    -------
    lam : float
        Approximate dominant eigenvalue.
    v : (n,) ndarray
        Approximate unit eigenvector (||v||_2 = 1).
    iters : int
        Number of iterations performed.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.asarray(x0, dtype=float).reshape(-1)

    if x.size != n:
        # fall back to a deterministic nonzero vector if x0 shape is unexpected
        x = np.ones(n, dtype=float)

    # normalize (guard against near-zero)
    nx = np.linalg.norm(x)
    if nx <= 0.0 or not np.isfinite(nx):
        x = np.ones(n, dtype=float)
        nx = np.linalg.norm(x)
    x = x / nx

    lam_old = 0.0
    iters = 0

    for k in range(int(maxit)):
        y = A @ x
        ny = np.linalg.norm(y)

        # If A @ x is ~0, then x is in (approx) nullspace; stop.
        if ny == 0.0 or not np.isfinite(ny):
            lam = 0.0
            iters = k + 1
            return lam, x, iters

        x = y / ny

        # Rayleigh quotient (stable for symmetric A)
        lam = float(x @ (A @ x))

        iters = k + 1
        if k > 0:
            denom = max(1.0, abs(lam))
            if abs(lam - lam_old) <= tol * denom:
                break
        lam_old = lam

    return lam, x, iters


# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================

def svd_compress(image, k):
    """Compute a rank-k approximation of a grayscale image using SVD.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    k : int
        Target rank (1 <= k <= min(m, n)).

    Returns
    -------
    image_k : (m, n) ndarray
        Rank-k approximation of the image.
    rel_error : float
        Relative Frobenius error ||image - image_k||_F / ||image||_F.
    compression_ratio : float
        (Number of stored parameters in image_k) / (m * n).
    """
    A = np.asarray(image, dtype=float)
    m, n = A.shape
    r = min(m, n)
    k = int(k)
    if k < 1:
        k = 1
    if k > r:
        k = r

    # Full_matrices=False gives compact SVD, cheaper and sufficient for rank-k reconstruction.
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]

    # rank-k reconstruction: Uk diag(Sk) Vtk
    A_k = (Uk * Sk) @ Vtk

    # Relative Frobenius error
    denom = np.linalg.norm(A, ord="fro")
    if denom == 0.0:
        rel_error = 0.0
    else:
        rel_error = float(np.linalg.norm(A - A_k, ord="fro") / denom)

    # store Uk (m*k) + Vtk (k*n) + Sk (k)
    compression_ratio = float(k * (m + n + 1) / (m * n))

    return A_k, rel_error, compression_ratio


# =========================================================
# 3. SVD-based feature extraction
# =========================================================

def svd_features(image, p):
    """Extract SVD-based features from a grayscale image.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    p : int
        Number of leading singular values to use (p <= min(m, n)).

    Returns
    -------
    feat : (p + 2,) ndarray
        Feature vector consisting of:
        [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]
    """
    A = np.asarray(image, dtype=float)
    m, n = A.shape
    r = min(m, n)
    p = int(p)
    if p < 1:
        p = 1
    if p > r:
        p = r

    # singular values only
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # normalize singular values (avoid divide-by-zero)
    ssum = float(np.sum(S))
    if ssum > 0.0 and np.isfinite(ssum):
        sig_norm = S[:p] / ssum
    else:
        sig_norm = S[:p].copy()

    # energy ratios based on squared singular values
    energy = S * S
    total_energy = float(np.sum(energy))
    if total_energy > 0.0 and np.isfinite(total_energy):
        cum = np.cumsum(energy) / total_energy
        r_90 = float(np.searchsorted(cum, 0.90) + 1)
        r_95 = float(np.searchsorted(cum, 0.95) + 1)
    else:
        # degenerate (all-zero) image: define ranks as 0
        r_90 = 0.0
        r_95 = 0.0

    feat = np.concatenate([sig_norm.astype(float), np.array([r_90, r_95], dtype=float)])
    return feat


# =========================================================
# 4. Two-class LDA: training
# =========================================================

def lda_train(X, y):
    """Train a two-class Linear Discriminant Analysis (LDA) classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    y : (N,) ndarray
        Labels (0 or 1).

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction vector (not necessarily unit length).
    threshold : float
        Threshold in 1D projected space for classifying 0 vs 1.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    X0 = X[y == 0]
    X1 = X[y == 1]

    # Means (handle edge cases defensively)
    if X0.size == 0 or X1.size == 0:
        # If a class is missing, return a trivial classifier
        d = X.shape[1]
        w = np.zeros(d, dtype=float)
        threshold = 0.0
        return w, threshold

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    # Within-class scatter
    C0 = X0 - mu0
    C1 = X1 - mu1
    Sw = C0.T @ C0 + C1.T @ C1

    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    lam = 1e-6 * tr / d if tr > 0.0 and np.isfinite(tr) else 1e-6
    Sw_reg = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw_reg, (mu1 - mu0))

    # Threshold in projected space: midpoint of projected class means
    m0 = float(w @ mu0)
    m1 = float(w @ mu1)
    threshold = 0.5 * (m0 + m1)

    return w, threshold


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================

def lda_predict(X, w, threshold):
    """Predict class labels using a trained LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    w : (d,) ndarray
        Discriminant direction (from lda_train).
    threshold : float
        Threshold (from lda_train).

    Returns
    -------
    y_pred : (N,) ndarray
        Predicted labels (0 or 1).
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    z = X @ w
    return (z >= threshold).astype(int)


def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        try:
            data = np.load("project_data.npz")
        except OSError:
            print("No example data file found ('project_data_example.npz' or 'project_data.npz').")
            return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    p = min(32, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    def build_features(X):
        return np.vstack([svd_features(img, p) for img in X])

    Xf_train = build_features(X_train)
    Xf_test = build_features(X_test)

    w, threshold = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, threshold)

    acc = float(np.mean(y_pred == y_test))
    print(f"Example test accuracy: {acc:.3f}")


if __name__ == "__main__":
    _example_run()
