/* Quantum circuit simulator + feature extraction (JavaScript port of python/qsim.py).
   Same circuit, same qubit convention (qubit 0 = most-significant bit; feature k on qubit n-1-k). */
(function (root) {
  const Q = {};
  const bit = (i, q, n) => (i >> (n - 1 - q)) & 1;

  /* a: 28x28 array of floats in [0,1]  ->  8 classical features */
  Q.features = function (a) {
    const f = a.flat(), m = f.reduce((x, y) => x + y) / 784;
    const sd = Math.sqrt(f.reduce((x, y) => x + (y - m) ** 2, 0) / 784);
    let gx = 0, gy = 0, dark = 0, bright = 0;
    for (let r = 0; r < 28; r++) for (let c = 0; c < 27; c++) gx += Math.abs(a[r][c + 1] - a[r][c]);
    for (let r = 0; r < 27; r++) for (let c = 0; c < 28; c++) gy += Math.abs(a[r + 1][c] - a[r][c]);
    for (const v of f) { if (v < 0.25) dark++; if (v > 0.6) bright++; }
    const mean = (r0, r1, c0, c1) => { let t = 0; for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) t += a[r][c]; return t / ((r1 - r0) * (c1 - c0)); };
    const up = mean(0, 14, 0, 28), lo = mean(14, 28, 0, 28), le = mean(0, 28, 0, 14), ri = mean(0, 28, 14, 28);
    return [m, sd, gx / 756 + gy / 756, lo / (up + 1e-6), le / (ri + 1e-6), mean(7, 21, 7, 21), dark / 784, bright / 784];
  };

  Q.angles = (f, R) => f.slice(0, R.nq).map((v, i) => Math.min(1, Math.max(0, (v - R.lo[i]) / (R.hi[i] - R.lo[i]))) * Math.PI);

  function ry(s, q, t, n) { const c = Math.cos(t / 2), sn = Math.sin(t / 2), m = 1 << (n - 1 - q), N = 1 << n;
    for (let i = 0; i < N; i++) if (!(i & m)) { const j = i | m, a = s[i], b = s[j]; s[i] = c * a - sn * b; s[j] = sn * a + c * b; } }
  function cx(s, c, t, n) { const m = 1 << (n - 1 - t), N = 1 << n;
    for (let i = 0; i < N; i++) if (bit(i, c, n) && !(i & m)) { const j = i | m; const x = s[i]; s[i] = s[j]; s[j] = x; } }

  /* run the full circuit for encoded angles; returns <Z0> and the 2^n amplitudes */
  Q.run = function (ang, R) {
    const n = R.nq, N = 1 << n, s = new Float64Array(N);
    for (let i = 0; i < N; i++) { let a = 1; for (let q = 0; q < n; q++) a *= bit(i, q, n) ? Math.sin(ang[n - 1 - q] / 2) : Math.cos(ang[n - 1 - q] / 2); s[i] = a; }
    for (let l = 0; l < R.layers; l++) {
      for (let q = 0; q < n; q++) ry(s, q, R.w[l][q], n);
      for (let q = 0; q < n; q++) cx(s, q, (q + 1) % n, n);
    }
    let z = 0; for (let i = 0; i < N; i++) z += (bit(i, 0, n) ? -1 : 1) * s[i] * s[i];
    return { z, s };
  };

  Q.predict = function (a, R) {
    const f = Q.features(a), ang = Q.angles(f, R), { z, s } = Q.run(ang, R);
    return { p: (1 - z) / 2, z, s, ang, f };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = Q; else root.QDX = Q;
})(typeof window !== 'undefined' ? window : this);
