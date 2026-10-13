"""OPTIONAL cross-check of the numpy simulator against PennyLane's default.qubit.

    pip install pennylane
    python python/pennylane_check.py

Builds the trained circuit in PennyLane and compares P(pneumonia) with our own
simulator on test images. Note: this script was written but NOT executed by the
author's environment (PennyLane was not installable offline) - run it yourself.
"""
import json, pathlib
import numpy as np
import pennylane as qml
from qsim import VQC, load_data, extract_features

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "js" / "results.js").read_text().split("=", 1)[1].strip().rstrip(";"))
n, L = R["nq"], R["layers"]
X, y = load_data(ROOT / "data" / "pneumoniamnist.npz")
F = extract_features(X["test"][:50])

m = VQC(n, L); m.w = np.array(R["w"]); m.lo = np.array(R["lo"]); m.hi = np.array(R["hi"])
ours = m.predict_proba(F); ang = m.angles(F)

dev = qml.device("default.qubit", wires=n)

@qml.qnode(dev)
def circuit(x, w):
    for q in range(n):
        qml.RY(x[n - 1 - q], wires=q)          # feature k sits on qubit n-1-k
    for l in range(L):
        for q in range(n): qml.RY(w[l][q], wires=q)
        for q in range(n): qml.CNOT(wires=[q, (q + 1) % n])
    return qml.expval(qml.PauliZ(0))

theirs = np.array([(1 - circuit(a, np.array(R["w"]))) / 2 for a in ang])
print("max |numpy - PennyLane| over 50 test images:", float(np.abs(ours - theirs).max()))
