(function () {
  const R = window.QDX_RESULTS, Q = window.QDX, $ = (id) => document.getElementById(id);
  const pc = (v, d = 1) => (v * 100).toFixed(d) + '%';
  const T = R.metrics.test, B = R.baseline, NQ = R.nq, N = 1 << NQ;
  const FDESC = ['Overall brightness of the image', 'Spread of pixel brightness (contrast)', 'Mean gradient magnitude (edges / opacities)', 'Lower half vs upper half brightness', 'Left half vs right half brightness', 'Brightness of the central lung region'];

  /* ---------- theme ---------- */
  const root = document.documentElement;
  try { const t = localStorage.getItem('qdx-theme'); if (t) root.dataset.theme = t; } catch (e) {}
  $('theme').onclick = () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('qdx-theme', root.dataset.theme); } catch (e) {}
  };

  /* ---------- router ---------- */
  const pages = [['home', 'Home'], ['dataset', 'Dataset'], ['model', 'Quantum model'], ['results', 'Results'], ['diagnose', 'Diagnose'], ['about', 'About']];
  $('links').innerHTML = pages.map((p) => `<a href="#${p[0]}" data-p="${p[0]}">${p[1]}</a>`).join('');
  function route() {
    const h = (location.hash || '#home').slice(1), id = pages.some((p) => p[0] === h) ? h : 'home';
    document.querySelectorAll('.page').forEach((e) => e.classList.toggle('on', e.id === id));
    document.querySelectorAll('#links a').forEach((e) => e.classList.toggle('on', e.dataset.p === id));
    scrollTo(0, 0); if (id === 'home') countUp(); if (id === 'results') animateBars();
  }
  addEventListener('hashchange', route);

  /* ---------- helpers ---------- */
  const stat = (v, l, attrs = '') => `<div class="card stat"><b ${attrs}>${v}</b><span>${l}</span></div>`;
  function countUp() {
    document.querySelectorAll('[data-count]').forEach((el) => {
      const to = +el.dataset.count, dec = +(el.dataset.dec || 0), suf = el.dataset.suf || '', t0 = performance.now();
      (function f(t) { const k = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(dec) + suf; if (k < 1) requestAnimationFrame(f); })(t0);
    });
  }
  function animateBars() { document.querySelectorAll('#results .bar i').forEach((i) => { const w = i.dataset.w; i.style.width = '0'; requestAnimationFrame(() => requestAnimationFrame(() => (i.style.width = w))); }); }
  function sampleCanvas(s) {
    const c = document.createElement('canvas'); c.width = c.height = 28; const x = c.getContext('2d'), im = x.createImageData(28, 28), b = atob(s.px);
    for (let i = 0; i < 784; i++) { const v = b.charCodeAt(i); im.data.set([v, v, v, 255], i * 4); } x.putImageData(im, 0, 0); return c;
  }
  function sampleArr(s) { const b = atob(s.px), a = []; for (let r = 0; r < 28; r++) { a.push([]); for (let c = 0; c < 28; c++) a[r].push(b.charCodeAt(r * 28 + c) / 255); } return a; }
  function gallery(el, click) {
    R.samples.forEach((s) => { const d = document.createElement('div'); d.className = 'gi'; d.title = s.label ? 'Pneumonia (ground truth)' : 'Normal (ground truth)';
      d.appendChild(sampleCanvas(s)); d.insertAdjacentHTML('beforeend', `<em class="${s.label ? 'p' : ''}">${s.label ? 'PNEU' : 'NORMAL'}</em>`); d.onclick = () => click(s); el.appendChild(d); });
  }

  /* ---------- home ---------- */
  $('hq').textContent = NQ; $('ml').textContent = R.layers;
  $('hstats').innerHTML = stat('', 'Test accuracy', `data-count="${T.acc * 100}" data-dec="1" data-suf="%"`) + stat('', 'ROC-AUC', `data-count="${R.auc}" data-dec="3"`) +
    stat('', 'Sensitivity (pneumonia caught)', `data-count="${T.sens * 100}" data-dec="1" data-suf="%"`) + stat(NQ + ' qubits', `${R.layers} layers · ${R.layers * NQ} trainable parameters`);

  /* ---------- dataset ---------- */
  $('dstats').innerHTML = stat((R.n.train + R.n.val + R.n.test).toLocaleString(), 'Chest X-ray images') + stat('28 × 28', 'Grayscale resolution') + stat('2', 'Classes: Normal, Pneumonia') + stat('3', 'Splits: train / val / test');
  $('dbars').innerHTML = ['train', 'val', 'test'].map((s) => { const d = R.dist[s], t = d[0] + d[1];
    return `<div class="fine">${s} · ${t} images</div><div class="dbar"><span class="n" style="width:${d[0] / t * 100}%">${pc(d[0] / t, 0)} normal</span><span class="p" style="width:${d[1] / t * 100}%">${pc(d[1] / t, 0)} pneumonia</span></div>`; }).join('');
  $('dtab').innerHTML = '<tr><th>Split</th><th>Images</th><th>Normal</th><th>Pneumonia</th></tr>' + ['train', 'val', 'test'].map((s) => `<tr><td>${s}</td><td>${R.n[s]}</td><td>${R.dist[s][0]}</td><td>${R.dist[s][1]}</td></tr>`).join('');

  /* ---------- model ---------- */
  (function circuit() {
    const gap = 46, cw = 34, xs0 = 90, colsPerLayer = 1 + NQ, W = xs0 + 60 + R.layers * (colsPerLayer * cw + 20) + 80, H = 40 + NQ * gap;
    let s = `<svg viewBox="0 0 ${W} ${H}" style="min-width:${Math.min(W, 900)}px;width:100%">`;
    for (let q = 0; q < NQ; q++) s += `<text x="8" y="${34 + q * gap + 4}">q${q} |0⟩</text><line x1="62" x2="${W - 20}" y1="${34 + q * gap}" y2="${34 + q * gap}" stroke="var(--bd)" stroke-width="2"/>`;
    const box = (x, q, t, col, tip) => `<g><title>${tip}</title><rect x="${x - 17}" y="${34 + q * gap - 14}" width="34" height="28" rx="7" fill="${col}" fill-opacity=".9"/><text x="${x}" y="${34 + q * gap + 4}" text-anchor="middle" style="fill:#fff;font-weight:600;font-size:10px">${t}</text></g>`;
    let x = xs0;
    for (let q = 0; q < NQ; q++) s += box(x, q, 'RY', 'var(--cyan)', `Encoding of feature ${NQ - 1 - q + 1}: ${R.feature_names[NQ - 1 - q]}`);
    x += 54;
    for (let l = 0; l < R.layers; l++) {
      for (let q = 0; q < NQ; q++) s += box(x, q, 'RY', 'var(--violet)', `Layer ${l + 1}, qubit ${q}: θ = ${R.w[l][q].toFixed(3)} rad`);
      x += cw + 10;
      for (let q = 0; q < NQ; q++) { const t = (q + 1) % NQ, y1 = 34 + q * gap, y2 = 34 + t * gap;
        s += `<line x1="${x}" x2="${x}" y1="${y1}" y2="${y2}" stroke="var(--pink)" stroke-width="2"/><circle cx="${x}" cy="${y1}" r="4.5" fill="var(--pink)"/><circle cx="${x}" cy="${y2}" r="9" fill="var(--bg)" stroke="var(--pink)" stroke-width="2"/><line x1="${x - 9}" x2="${x + 9}" y1="${y2}" y2="${y2}" stroke="var(--pink)" stroke-width="2"/><line x1="${x}" x2="${x}" y1="${y2 - 9}" y2="${y2 + 9}" stroke="var(--pink)" stroke-width="2"/>`; x += cw - 4; }
      x += 10;
    }
    s += `<rect x="${x + 4}" y="${34 - 16}" width="40" height="32" rx="8" fill="var(--good)"/><text x="${x + 24}" y="${34 + 4}" text-anchor="middle" style="fill:#fff;font-weight:700">⟨Z⟩</text></svg>`;
    $('circ').innerHTML = s + '<p class="fine">Cyan: feature encoding · Violet: trainable rotations · Pink: CNOT ring · Green: measurement of qubit 0</p>';
  })();
  $('ftab').innerHTML = '<tr><th>Qubit</th><th>Feature</th><th>Meaning</th></tr>' + Array.from({ length: NQ }, (_, q) => `<tr><td>q${q}</td><td>${R.feature_names[NQ - 1 - q]}</td><td class="fine">${FDESC[NQ - 1 - q]}</td></tr>`).join('');
  $('htab').innerHTML = [['Qubits / layers', `${NQ} / ${R.layers}`], ['Trainable parameters', R.layers * NQ], ['Optimizer', 'Adam, learning rate 0.08'], ['Gradients', 'Parameter-shift rule'], ['Loss', 'Class-weighted squared error'], ['Epochs', R.steps], ['Random starts', `4 (best seed ${R.seed} by validation)`], ['Decision threshold', `${R.th.toFixed(2)} (tuned on validation)`]].map((r) => `<tr><td>${r[0]}</td><td><b>${r[1]}</b></td></tr>`).join('');
  (function loss() {
    const ls = R.loss, mx = Math.max(...ls), mn = Math.min(...ls), pts = ls.map((v, i) => `${36 + i / (ls.length - 1) * 340},${120 - (v - mn) / (mx - mn) * 100}`).join(' ');
    $('lossc').innerHTML = `<svg viewBox="0 0 400 150"><polyline points="${pts}" fill="none" stroke="var(--violet)" stroke-width="3" stroke-linejoin="round"/><line x1="36" x2="376" y1="120" y2="120" stroke="var(--bd)"/><text x="2" y="24">${mx.toFixed(2)}</text><text x="2" y="124">${mn.toFixed(2)}</text><text x="36" y="142">epoch 0</text><text x="330" y="142">epoch ${R.steps}</text></svg>`;
  })();

  /* ---------- results ---------- */
  $('rlead').textContent = `Evaluated on ${R.n.test} held-out test X-rays that were never used for training or for choosing the threshold.`;
  $('rstats').innerHTML = stat('', 'Accuracy', `data-count="${T.acc * 100}" data-dec="1" data-suf="%"`) + stat('', 'Sensitivity', `data-count="${T.sens * 100}" data-dec="1" data-suf="%"`) + stat('', 'Specificity', `data-count="${T.spec * 100}" data-dec="1" data-suf="%"`) + stat('', 'ROC-AUC', `data-count="${R.auc}" data-dec="3"`);
  document.querySelectorAll('#rstats [data-count]').forEach((el) => { el.textContent = el.dataset.count * 1 > 2 ? (+el.dataset.count).toFixed(1) + '%' : (+el.dataset.count).toFixed(3); });
  const cell = (n, tip, good) => `<div class="c" style="background:${good ? 'var(--good)' : 'var(--bad)'}">${n}<small>${tip}</small></div>`;
  $('cm').innerHTML = `<div></div><div>Pred. normal</div><div>Pred. pneumonia</div><div>True normal</div>${cell(T.tn, 'correct', 1)}${cell(T.fp, 'false alarm', 0)}<div>True pneumonia</div>${cell(T.fn, 'missed', 0)}${cell(T.tp, 'correct', 1)}`;
  $('roc').innerHTML = `<svg viewBox="0 0 240 230"><rect x="34" y="10" width="190" height="190" fill="none" stroke="var(--bd)"/><line x1="34" y1="200" x2="224" y2="10" stroke="var(--bd)" stroke-dasharray="4"/><polyline fill="none" stroke="var(--cyan)" stroke-width="3" stroke-linejoin="round" points="${R.roc.map((p) => `${34 + p[0] * 190},${200 - p[1] * 190}`).join(' ')}"/><text x="80" y="224">False positive rate</text><text x="120" y="170" style="font-weight:700;font-size:13px">AUC ${R.auc.toFixed(3)}</text></svg>`;
  const cmpRows = [['Accuracy', T.acc, B.acc], ['Sensitivity', T.sens, B.sens], ['Specificity', T.spec, B.spec]];
  $('cmp').innerHTML = cmpRows.map((r) => `<div class="fine">${r[0]}</div><div class="bar"><i class="good" data-w="${r[1] * 100}%" title="Quantum"></i></div><div class="bar" style="margin-top:-6px"><i data-w="${r[2] * 100}%" style="background:var(--mut)"></i></div>`).join('') +
    `<div class="fine"><span style="color:var(--good)">■</span> Quantum circuit ${pc(T.acc)} · <span>■</span> Logistic regression ${pc(B.acc)} (AUC ${B.auc.toFixed(3)})</div>`;
  $('bnote').textContent = `Honest note: a simple classical model on the same ${NQ} features scores slightly higher. The quantum circuit is competitive with only ${R.layers * NQ} trainable parameters but shows no quantum advantage here.`;
  $('stab').innerHTML = '<tr><th>Split</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th></tr>' + ['train', 'val', 'test'].map((s) => { const m = R.metrics[s]; return `<tr><td>${s}</td><td>${pc(m.acc)}</td><td>${pc(m.sens)}</td><td>${pc(m.spec)}</td></tr>`; }).join('');

  /* ---------- diagnose ---------- */
  $('dstats'); const svg = $('res').querySelector('.gauge'); svg.insertAdjacentHTML('afterbegin', '<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="var(--good)"/><stop offset=".5" stop-color="#f5b83d"/><stop offset="1" stop-color="var(--bad)"/></linearGradient></defs>');
  $('nst').textContent = N;
  function show(a, truth, warn) {
    const r = Q.predict(a, R), p = r.p, pn = p > R.th;
    $('idle').hidden = true; $('res').hidden = false;
    $('gfill').style.strokeDashoffset = 283 * (1 - p); $('gp').textContent = pc(p);
    $('verdict').textContent = pn ? 'Pneumonia pattern detected' : 'No pneumonia pattern'; $('verdict').style.color = pn ? 'var(--bad)' : 'var(--good)';
    $('warn').hidden = !warn; $('warn').textContent = warn || '';
    $('b1').style.width = '0'; $('b0').style.width = '0'; requestAnimationFrame(() => { $('b1').style.width = pc(p); $('b0').style.width = pc(1 - p); });
    $('fbars').innerHTML = r.ang.map((v, i) => `<div class="frow"><span>${R.feature_names[i]}</span><div class="bar"><i style="width:${v / Math.PI * 100}%;background:var(--violet)"></i></div></div>`).join('');
    const A = $('amps'); A.innerHTML = ''; const mx = Math.max(...Array.from(r.s, (v) => v * v));
    for (let i = 0; i < N; i++) { const e = document.createElement('div'); e.title = '|' + i.toString(2).padStart(NQ, '0') + '⟩'; A.appendChild(e); requestAnimationFrame(() => requestAnimationFrame(() => (e.style.height = Math.max(2, r.s[i] ** 2 / mx * 100) + '%'))); }
    $('tech').textContent = `⟨Z₀⟩ = ${r.z.toFixed(3)} · decision threshold ${R.th.toFixed(2)} (tuned on validation data) · ${R.layers * NQ} trained parameters`;
    $('truth').textContent = truth === undefined ? '' : 'Ground-truth label: ' + (truth ? 'pneumonia' : 'normal') + ((truth == 1) === pn ? '  ✓ correct' : '  ✗ misclassified');
  }
  function withScan(fn) { const s = $('scan'); s.hidden = false; s.style.animation = 'none'; void s.offsetWidth; s.style.animation = ''; setTimeout(() => { s.hidden = true; fn(); }, 900); }
  const cv = document.createElement('canvas'); cv.width = cv.height = 28;
  function load(file) {
    if (!file) return; const u = URL.createObjectURL(file), img = new Image();
    img.onload = () => {
      const side = Math.min(img.width, img.height), sx = (img.width - side) / 2, sy = (img.height - side) / 2, x = cv.getContext('2d');
      x.fillStyle = '#000'; x.fillRect(0, 0, 28, 28); x.imageSmoothingQuality = 'high'; x.drawImage(img, sx, sy, side, side, 0, 0, 28, 28);
      const px = x.getImageData(0, 0, 28, 28).data, a = []; let sat = 0;
      for (let r = 0; r < 28; r++) { a.push([]); for (let c = 0; c < 28; c++) { const k = (r * 28 + c) * 4, R_ = px[k], G = px[k + 1], B_ = px[k + 2]; sat += Math.max(R_, G, B_) - Math.min(R_, G, B_); a[r].push((.299 * R_ + .587 * G + .114 * B_) / 255); } }
      $('ph').innerHTML = '<img src="' + u + '" alt="uploaded scan">';
      withScan(() => show(a, undefined, sat / 784 > 40 ? 'This image looks like a colour photo, not a grayscale X-ray. The result is unreliable.' : ''));
    }; img.src = u;
  }
  $('file').onchange = (e) => load(e.target.files[0]);
  const dr = $('drop'); dr.ondragover = (e) => { e.preventDefault(); dr.classList.add('on'); }; dr.ondragleave = () => dr.classList.remove('on');
  dr.ondrop = (e) => { e.preventDefault(); dr.classList.remove('on'); load(e.dataTransfer.files[0]); };
  function pick(s) { const c = sampleCanvas(s), ph = $('ph'); ph.innerHTML = ''; c.style.width = '220px'; c.style.height = '220px'; ph.appendChild(c); if (location.hash !== '#diagnose') location.hash = '#diagnose'; withScan(() => show(sampleArr(s), s.label)); }
  gallery($('try'), pick); gallery($('dgal'), pick);

  /* ---------- background particles ---------- */
  (function bg() {
    const c = $('bg'), x = c.getContext('2d'); let w, h, P = [];
    const size = () => { w = c.width = innerWidth; h = c.height = innerHeight; P = Array.from({ length: Math.min(70, Math.floor(w * h / 18000)) }, () => ({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * .3, vy: (Math.random() - .5) * .3 })); };
    size(); addEventListener('resize', size);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    (function f() {
      x.clearRect(0, 0, w, h); const col = getComputedStyle(root).getPropertyValue('--violet').trim() || '#8b74ff';
      P.forEach((p, i) => { p.x = (p.x + p.vx + w) % w; p.y = (p.y + p.vy + h) % h; x.fillStyle = col; x.globalAlpha = .7; x.beginPath(); x.arc(p.x, p.y, 1.6, 0, 7); x.fill();
        for (let j = i + 1; j < P.length; j++) { const q = P[j], d = Math.hypot(p.x - q.x, p.y - q.y); if (d < 120) { x.globalAlpha = (1 - d / 120) * .25; x.strokeStyle = col; x.beginPath(); x.moveTo(p.x, p.y); x.lineTo(q.x, q.y); x.stroke(); } } });
      requestAnimationFrame(f);
    })();
  })();

  route();
})();
