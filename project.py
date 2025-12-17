import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair (symmetric matrix)
# =========================================================
def power_method(A, x0, maxit, tol):
    """
    Approximate the dominant eigenvalue/eigenvector of a real symmetric matrix A
    using the power method.

    Parameters
    ----------
    A : (n, n) ndarray
        Real symmetric matrix.
    x0 : (n,) ndarray
        Initial guess (must be nonzero).
    maxit : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on successive eigenvalue estimates.

    Returns
    -------
    lam : float
        Approximate dominant eigenvalue.
    x : (n,) ndarray
        Approximate dominant eigenvector (unit 2-norm).
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]

    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != n:
        raise ValueError("x0 must have shape (n,)")
    nx = np.linalg.norm(x)
    if nx == 0:
        # fall back to a deterministic nonzero vector
        x = np.ones(n, dtype=float)
        nx = np.linalg.norm(x)
    x = x / nx

    lam_old = None
    lam = float(x @ (A @ x))

    for _ in range(int(maxit)):
        z = A @ x
        nz = np.linalg.norm(z)
        if nz == 0:
            # A maps x to 0; dominant eigenvalue is 0
            lam = 0.0
            break
        x = z / nz

        lam = float(x @ (A @ x))  # Rayleigh quotient
        if lam_old is not None:
            # relative/absolute hybrid stopping
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)):
                break
        lam_old = lam

    return lam, x


# =========================================================
# 2. Rank-k image approximation using SVD
# =========================================================
def svd_compress(image, k):
    """
    Compute a rank-k approximation of a grayscale image using SVD.

    Parameters
    ----------
    image : (m, n) ndarray
        Image matrix.
    k : int
        Target rank (k >= 1).

    Returns
    -------
    image_k : (m, n) ndarray
        Rank-k approximation.
    rel_error : float
        Relative Frobenius error ||A - A_k||_F / ||A||_F.
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    if k < 1:
        raise ValueError("k must be >= 1")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = s.size
    k_eff = int(min(k, r))

    # A_k = U_k diag(s_k) V_k^T
    Ak = (U[:, :k_eff] * s[:k_eff]) @ Vt[:k_eff, :]

    denom = np.linalg.norm(A, ord="fro")
    if denom == 0:
        rel_error = 0.0
    else:
        rel_error = float(np.linalg.norm(A - Ak, ord="fro") / denom)

    return Ak, rel_error


# =========================================================
# 3. Build feature vector from image singular values
# =========================================================
def svd_features(image, p):
    """
    Extract SVD-based features from a grayscale image.

    Feature vector:
        [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]

    where r_alpha is the smallest integer r such that
        sum_{i=1}^r sigma_i^2 >= alpha * sum_{i} sigma_i^2.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    p : int
        Number of leading singular values to use.

    Returns
    -------
    feat : (p + 2,) ndarray
        Feature vector.
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    m, n = A.shape
    rmax = min(m, n)
    if p < 1 or p > rmax:
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    s = np.linalg.svd(A, compute_uv=False)
    # Normalize singular values (scale-invariant features)
    s_sum = float(np.sum(s))
    if s_sum == 0.0:
        lead = np.zeros(p, dtype=float)
    else:
        lead = (s[:p] / s_sum).astype(float)

    # Energy ratios using squared singular values
    s2 = s * s
    total_energy = float(np.sum(s2))
    if total_energy == 0.0:
        r90 = 0.0
        r95 = 0.0
    else:
        c = np.cumsum(s2) / total_energy
        r90 = float(np.searchsorted(c, 0.90) + 1)
        r95 = float(np.searchsorted(c, 0.95) + 1)

    feat = np.concatenate([lead, np.array([r90, r95], dtype=float)])
    return feat


# =========================================================
# 4. Two-class LDA: training
# =========================================================
def lda_train(X, y):
    """
    Train a two-class Linear Discriminant Analysis classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    y : (N,) ndarray
        Binary labels (0/1 or -1/+1). Any two distinct values are accepted.

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction.
    threshold : float
        Classification threshold on the score z = X @ w:
            predict 1 if z >= threshold else 0.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.size != X.shape[0]:
        raise ValueError("y must have length N")

    # Map labels to {0,1} deterministically
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")
    y01 = (y == classes[1]).astype(int)

    X0 = X[y01 == 0]
    X1 = X[y01 == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes must have at least one sample")

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)

    # Within-class scatter
    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = X0c.T @ X0c + X1c.T @ X1c

    # Light Tikhonov regularization for numerical stability
    d = Sw.shape[0]
    trace = float(np.trace(Sw))
    lam = 1e-6 * (trace / d if d > 0 else 1.0) + 1e-12
    Sw_reg = Sw + lam * np.eye(d)

    b = (mu1 - mu0)

    # Solve Sw w = (mu1 - mu0)
    try:
        w = np.linalg.solve(Sw_reg, b)
    except np.linalg.LinAlgError:
        # Fallback: least squares
        w = np.linalg.lstsq(Sw_reg, b, rcond=None)[0]

    w = w.reshape(-1)

    # Threshold at the midpoint of projected class means
    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    # Ensure class-1 has larger projection than class-0 for the >= rule
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================
def lda_predict(X, w, threshold):
    """
    Predict labels for samples using the trained LDA model.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    w : (d,) ndarray
        Discriminant direction from lda_train.
    threshold : float
        Threshold from lda_train.

    Returns
    -------
    y_pred : (N,) ndarray of int
        Predicted labels in {0,1}.
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch between X and w")

    z = X @ w
    y_pred = (z >= float(threshold)).astype(int)
    return y_pred


# =========================================================
# Local smoke test (not used by autograder)
# =========================================================
def _example_run():
    """
    Run a tiny end-to-end test if 'project_data_example.npz' exists.
    This is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example dataset found (project_data_example.npz).")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Extract features per image
    p = min(20, min(X_train.shape[1], X_train.shape[2]))
    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test  = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)
    acc = np.mean(y_pred == (y_test == np.unique(y_test)[1]).astype(int))
    print(f"Example test accuracy: {acc:.3f}")


if __name__ == "__main__":
    _example_run()
