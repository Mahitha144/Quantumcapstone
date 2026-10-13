"""Sanity tests for the quantum simulator. Run:  python tests/test_simulator.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import numpy as np
from qsim import VQC

def test_unitary_is_orthogonal():
    m = VQC(4, 2, seed=1); U = m.unitary(m.w)
    assert np.allclose(U @ U.T, np.eye(16), atol=1e-10)

def test_state_is_normalised():
    m = VQC(6, 3, seed=2); ang = np.random.default_rng(0).uniform(0, np.pi, (5, 6))
    out = m.product_state(ang) @ m.unitary(m.w).T
    assert np.allclose((out ** 2).sum(1), 1.0)

def test_single_qubit_ry_gives_cos_theta():
    m = VQC(1, 0)
    for th in (0.0, 0.7, 1.9, np.pi):
        S = m.product_state(np.array([[th]]))
        assert np.isclose(((S ** 2) @ m.zsign)[0], np.cos(th))

def test_cnot_ring_by_hand():
    # qubit0=|+>, qubit1=|0>. CNOT(0->1): (|00>+|11>)/sqrt2. CNOT(1->0): (|00>+|01>)/sqrt2
    # => qubit 0 is |0> with certainty, so <Z0> = +1
    m = VQC(2, 1); w = np.zeros((1, 2))
    S = m.product_state(np.array([[0.0, np.pi / 2]]))   # feature 1 is encoded on qubit 0
    assert np.isclose(m.expval_z(S, w)[0], 1.0)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
