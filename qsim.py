"""Quantum circuit simulator + hybrid quantum-classical classifier (VQC)
for PneumoniaMNIST chest X-rays.

Circuit (n qubits, L layers):
  1. angle encoding : RY(feature angle) on every qubit
  2. L x [ RY(theta) on every qubit , then CNOT ring q -> q+1 (mod n) ]
  3. readout        : <Z> on qubit 0,  P(pneumonia) = (1 - <Z0>) / 2

Qubit 0 is the most-significant bit of the state index. Feature k is encoded on
qubit (n-1-k); the website (js/quantum.js) uses exactly the same convention.
"""
import numpy as np


# ----------------------------------------------------------------- data ----
def load_data(path):
    d = np.load(path)
    splits = ("train", "val", "test")
    X = {s: d[f"{s}_images"] for s in splits}
    y = {s: d[f"{s}_labels"].ravel().astype(int) for s in splits}
    return X, y


def extract_features(imgs):
    """imgs: (B, 28, 28) uint8  ->  (B, 8) classical features."""
    a = imgs.astype(float) / 255.0
    gx = np.abs(np.diff(a, axis=2)).mean((1, 2))
    gy = np.abs(np.diff(a, axis=1)).mean((1, 2))
    up, lo = a[:, :14].mean((1, 2)), a[:, 14:].mean((1, 2))
    le, ri = a[:, :, :14].mean((1, 2)), a[:, :, 14:].mean((1, 2))
    return np.stack([
        a.mean((1, 2)),                 # 0 mean intensity
        a.std((1, 2)),                  # 1 contrast
        gx + gy,                        # 2 edge energy
        lo / (up + 1e-6),               # 3 lower / upper brightness
        le / (ri + 1e-6),               # 4 left / right brightness
        a[:, 7:21, 7:21].mean((1, 2)),  # 5 centre brightness
        (a < 0.25).mean((1, 2)),        # 6 dark-pixel fraction
        (a > 0.60).mean((1, 2)),        # 7 bright-pixel fraction
    ], axis=1)


FEATURE_NAMES = ["Mean intensity", "Contrast", "Edge energy", "Lower/upper brightness",
                 "Left/right brightness", "Centre brightness", "Dark fraction", "Bright fraction"]


# -------------------------------------------------------------- metrics ----
def metrics(p, y, th):
    pr = (p > th).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    return dict(acc=(tp + tn) / len(y), sens=tp / (tp + fn), spec=tn / (tn + fp),
                prec=tp / max(tp + fp, 1), tp=tp, tn=tn, fp=fp, fn=fn)


def best_threshold(p, y):
    ths = np.linspace(0.05, 0.95, 91)
    return float(ths[int(np.argmax([metrics(p, y, t)["acc"] for t in ths]))])


# ----------------------------------------------------------- simulator ----
class VQC:
    def __init__(self, n_qubits=6, n_layers=3, seed=0):
        self.n, self.L, self.N = n_qubits, n_layers, 2 ** n_qubits
        self.idx = np.arange(self.N)
        self.zsign = 1 - 2 * self.bit(0)           # eigenvalue of Z on qubit 0
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.uniform(-np.pi, np.pi, (n_layers, n_qubits)) * 0.5
        self.lo = self.hi = None

    def bit(self, q):
        return (self.idx >> (self.n - 1 - q)) & 1

    # --- encoding ---
    def fit_scaler(self, F):
        self.lo = np.percentile(F[:, :self.n], 2, axis=0)
        self.hi = np.percentile(F[:, :self.n], 98, axis=0)

    def angles(self, F):
        return np.clip((F[:, :self.n] - self.lo) / (self.hi - self.lo), 0, 1) * np.pi

    def product_state(self, ang):
        """RY(angle) |0> on every qubit; feature k sits on qubit n-1-k."""
        ang = ang[:, ::-1]
        s = np.ones((len(ang), 1))
        for q in range(self.n):
            cs = np.stack([np.cos(ang[:, q] / 2), np.sin(ang[:, q] / 2)], 1)
            s = (s[:, :, None] * cs[:, None, :]).reshape(len(ang), -1)
        return s

    # --- trainable part of the circuit as a 2^n x 2^n orthogonal matrix ---
    def unitary(self, w):
        M = np.eye(self.N)
        for l in range(self.L):
            for q in range(self.n):
                c, s = np.cos(w[l, q] / 2), np.sin(w[l, q] / 2)
                i0 = self.idx[self.bit(q) == 0]; i1 = i0 | (1 << (self.n - 1 - q))
                a, b = M[i0].copy(), M[i1].copy()
                M[i0] = c * a - s * b; M[i1] = s * a + c * b
            for q in range(self.n):
                t = (q + 1) % self.n
                src = self.idx[(self.bit(q) == 1) & (self.bit(t) == 0)]
                dst = src | (1 << (self.n - 1 - t))
                tmp = M[src].copy(); M[src] = M[dst]; M[dst] = tmp
        return M

    def expval_z(self, S, w):
        out = S @ self.unitary(w).T
        return (out ** 2) @ self.zsign

    # --- training (parameter-shift gradients + Adam) ---
    def fit(self, F, y, steps=120, lr=0.08, log=print):
        self.fit_scaler(F)
        S = self.product_state(self.angles(F))
        wt = np.where(y == 1, 0.5 * len(y) / y.sum(), 0.5 * len(y) / (len(y) - y.sum()))
        w = self.w.copy(); m = np.zeros_like(w); v = np.zeros_like(w); hist = []
        for t in range(1, steps + 1):
            p = (1 - self.expval_z(S, w)) / 2
            loss = float(np.mean(wt * (p - y) ** 2)); hist.append(loss)
            dLdz = wt * 2 * (p - y) * (-0.5) / len(y)
            g = np.zeros_like(w)
            for l in range(self.L):
                for q in range(self.n):
                    wp = w.copy(); wp[l, q] += np.pi / 2
                    wm = w.copy(); wm[l, q] -= np.pi / 2
                    g[l, q] = np.sum(dLdz * (self.expval_z(S, wp) - self.expval_z(S, wm)) / 2)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
            w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if t % 20 == 0 or t == 1:
                log(f"    step {t:3d}/{steps}  loss {loss:.4f}")
        self.w = w
        return hist

    def predict_proba(self, F):
        S = self.product_state(self.angles(F))
        return (1 - self.expval_z(S, self.w)) / 2
