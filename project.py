import numpy as np

# 1. Power method for dominant eigenpair

def power_method(A, x0, maxit, tol):
    """Approximate the dominant eigenvalue and eigenvector of a real symmetric matrix A."""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square (n, n).")
    if x.size != n:
        raise ValueError("x0 must have shape (n,).")
    xnorm = np.linalg.norm(x)
    if xnorm == 0:
        raise ValueError("x0 must be nonzero.")

    v = x / xnorm
    lam_old = 0.0

    for it in range(1, int(maxit) + 1):
        w = A @ v
        wnorm = np.linalg.norm(w)
        if wnorm == 0:
            # A v = 0 => eigenvalue 0 and v is in nullspace direction
            return 0.0, v, it

        v = w / wnorm
        # For symmetric A, Rayleigh quotient is stable
        lam = float(v @ (A @ v))

        # relative change in eigenvalue
        denom = max(1.0, abs(lam))
        if abs(lam - lam_old) / denom < tol:
            return lam, v, it

        lam_old = lam

    return lam_old, v, int(maxit)


# 2. Rank-k image compression via SVD

def svd_compress(image, k):
    """Compute a rank-k approximation of a grayscale image using SVD."""
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array.")
    m, n = A.shape
    r = min(m, n)
    if not (1 <= int(k) <= r):
        raise ValueError("k must satisfy 1 <= k <= min(m, n).")
    k = int(k)

    # economy SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]
    A_k = (Uk * Sk) @ Vtk  # (m,k)*(k,n)

    # relative Frobenius error
    denom = np.linalg.norm(A, ord="fro")
    if denom == 0:
        rel_error = 0.0
    else:
        rel_error = float(np.linalg.norm(A - A_k, ord="fro") / denom)

    # compression ratio: store Uk (m*k), Vtk (k*n), Sk (k)
    stored = k * (m + n + 1)
    compression_ratio = float(stored / (m * n))

    return A_k, rel_error, compression_ratio


# 3. SVD-based feature extraction

def svd_features(image, p):
    """Extract SVD-based features from a grayscale image."""
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array.")
    m, n = A.shape
    r = min(m, n)
    p = int(p)
    if not (1 <= p <= r):
        raise ValueError("p must satisfy 1 <= p <= min(m, n).")

    # singular values only
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # normalized leading singular values: sigma_i / sum_j sigma_j
    ssum = float(np.sum(S))
    if ssum == 0.0:
        sig_norm = np.zeros(p, dtype=float)
    else:
        sig_norm = (S[:p] / ssum).astype(float)

    # energy ratios using sigma^2
    E = S**2
    Etot = float(np.sum(E))
    if Etot == 0.0:
        r90 = 0
        r95 = 0
    else:
        cume = np.cumsum(E) / Etot
        # smallest r such that cum energy >= target
        r90 = int(np.searchsorted(cume, 0.90, side="left") + 1)
        r95 = int(np.searchsorted(cume, 0.95, side="left") + 1)

    feat = np.empty(p + 2, dtype=float)
    feat[:p] = sig_norm
    feat[p] = float(r90)
    feat[p + 1] = float(r95)
    return feat


# 4. Two-class LDA: training

def lda_train(X, y):
    """Train a two-class LDA classifier."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be (N, d).")
    N, d = X.shape
    if y.size != N:
        raise ValueError("y must have length N.")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0/1 labels.")

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes must be present in training data.")

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)

    # within-class scatter (covariance up to a scalar)
    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = X0c.T @ X0c + X1c.T @ X1c

    # small Tikhonov regularization for stability
    tr = float(np.trace(Sw))
    lam = 1e-6 * (tr / d if d > 0 else 1.0) + 1e-12
    Sw_reg = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw_reg, (mu1 - mu0))
    w = w.reshape(-1)

    # midpoint threshold in projected space (equal priors)
    threshold = float(0.5 * (w @ (mu0 + mu1)))

    return w, threshold


# 5. Two-class LDA: prediction

def lda_predict(X, w, threshold):
    """Predict class labels using a trained LDA classifier."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be (N, d).")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch: X has d columns but w has length d.")

    scores = X @ w
    y_pred = (scores >= threshold).astype(int)
    return y_pred
