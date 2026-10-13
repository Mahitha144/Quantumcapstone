"""Train + test the quantum classifier on PneumoniaMNIST and export everything
the website needs.

    python python/train.py --data data/pneumoniamnist.npz

Outputs: results/metrics.json, results/training_log.txt, results/*.png, js/results.js
"""
import argparse, base64, json, pathlib, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from qsim import VQC, load_data, extract_features, metrics, best_threshold, FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "pneumoniamnist.npz"))
ap.add_argument("--qubits", type=int, default=6)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--no-plots", action="store_true")
a = ap.parse_args()

LOG = []
def log(*s):
    line = " ".join(str(x) for x in s); print(line); LOG.append(line)

t0 = time.time()
X, y = load_data(a.data)
F = {s: extract_features(X[s]) for s in X}
log(f"Dataset: train {len(y['train'])}  val {len(y['val'])}  test {len(y['test'])}")
log(f"Quantum model: {a.qubits} qubits, {a.layers} layers, {a.layers * a.qubits} trainable parameters")

# ---- train several random initialisations, keep the best on VALIDATION data ----
best = None
for seed in range(a.seeds):
    log(f"\n[seed {seed}] training ...")
    m = VQC(a.qubits, a.layers, seed=seed)
    hist = m.fit(F["train"], y["train"], steps=a.steps, log=log)
    pv = m.predict_proba(F["val"]); th = best_threshold(pv, y["val"])
    vacc = metrics(pv, y["val"], th)["acc"]
    log(f"[seed {seed}] validation accuracy {vacc:.4f} (threshold {th:.2f})")
    if best is None or vacc > best[0]:
        best = (vacc, seed, m, hist, th)
_, seed, m, hist, th = best
log(f"\nBest seed (by validation): {seed}   decision threshold: {th:.2f}")

# ---- final test (test data used only here) ----
P = {s: m.predict_proba(F[s]) for s in F}
M = {s: metrics(P[s], y[s], th) for s in P}
auc = float(roc_auc_score(y["test"], P["test"]))
fpr, tpr, _ = roc_curve(y["test"], P["test"])
for s in ("train", "val", "test"):
    r = M[s]; log(f"{s:5s}  acc {r['acc']:.4f}  sens {r['sens']:.4f}  spec {r['spec']:.4f}  prec {r['prec']:.4f}")
log(f"TEST ROC-AUC {auc:.4f}")
r = M["test"]
log(f"TEST confusion matrix: TN {r['tn']}  FP {r['fp']}  FN {r['fn']}  TP {r['tp']}")

# ---- classical baseline on the same features ----
Xn = lambda s: np.clip((F[s][:, :a.qubits] - m.lo) / (m.hi - m.lo), 0, 1)
lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xn("train"), y["train"])
pb = lr.predict_proba(Xn("test"))[:, 1]
base = metrics(pb, y["test"], 0.5); base["auc"] = float(roc_auc_score(y["test"], pb))
log(f"Classical baseline (logistic regression, same features): acc {base['acc']:.4f}  auc {base['auc']:.4f}")

# ---- export ----
k = np.linspace(0, len(fpr) - 1, 30).astype(int)
samples = []
for c in (0, 1):
    for i in np.where(y["test"] == c)[0][:8]:
        samples.append(dict(label=int(c), p=float(P["test"][i]),
                            px=base64.b64encode(X["test"][i].astype(np.uint8).tobytes()).decode()))
out = dict(nq=a.qubits, layers=a.layers, seed=seed, steps=a.steps, w=m.w.tolist(),
           lo=m.lo.tolist(), hi=m.hi.tolist(), th=th, metrics=M, auc=auc,
           roc=[[float(fpr[i]), float(tpr[i])] for i in k], loss=hist[::4], baseline=base,
           samples=samples, feature_names=FEATURE_NAMES[:a.qubits],
           n={s: len(y[s]) for s in y},
           dist={s: [int((y[s] == 0).sum()), int((y[s] == 1).sum())] for s in y})
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "metrics.json").write_text(json.dumps({k_: out[k_] for k_ in
    ("nq", "layers", "seed", "steps", "th", "metrics", "auc", "baseline", "n", "dist")}, indent=2))
(ROOT / "js" / "results.js").write_text("window.QDX_RESULTS = " + json.dumps(out, separators=(",", ":")) + ";\n")

if not a.no_plots:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(range(0, a.steps, 4), hist[::4], lw=2); ax[0].set(title="Training loss", xlabel="epoch", ylabel="weighted MSE")
    ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}"); ax[1].plot([0, 1], [0, 1], "--", c="grey")
    ax[1].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate"); ax[1].legend()
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]]); ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[2].set(title="Confusion matrix (test)", xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Pneumonia"],
              yticklabels=["Normal", "Pneumonia"], xlabel="Predicted", ylabel="True")
    plt.tight_layout(); plt.savefig(ROOT / "results" / "training_results.png", dpi=130)

log(f"\nDone in {time.time() - t0:.0f}s. Wrote results/metrics.json, results/training_results.png, js/results.js")
(ROOT / "results" / "training_log.txt").write_text("\n".join(LOG) + "\n")
