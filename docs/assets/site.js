/* KernelLens project page — interactions + hand-rolled SVG charts. */
(function () {
  "use strict";

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const cssvar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  document.documentElement.classList.add("js");

  /* ---------- scroll progress ---------- */
  const bar = $("#progress");
  const onScroll = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    bar.style.transform = "scaleX(" + (max > 0 ? h.scrollTop / max : 0) + ")";
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- pointer glow ---------- */
  const glow = $("#glow");
  let glowOn = false;
  addEventListener("pointermove", (e) => {
    document.documentElement.style.setProperty("--gx", e.clientX + "px");
    document.documentElement.style.setProperty("--gy", e.clientY + "px");
    if (!glowOn) { glow.classList.add("on"); glowOn = true; }
  });

  /* ---------- active nav link ---------- */
  const navLinks = $$("nav .links a[href^='#']");
  const sections = navLinks.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((s) => io.observe(s));
  }

  /* ---------- reveal ---------- */
  const reveals = $$(".rv");
  if ("IntersectionObserver" in window) {
    const ro = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); ro.unobserve(en.target); } });
    }, { rootMargin: "0px 0px -8% 0px" });
    reveals.forEach((el) => ro.observe(el));
  } else { reveals.forEach((el) => el.classList.add("in")); }

  /* ---------- count-up stats ---------- */
  const counters = $$("[data-count]");
  const runCount = (el) => {
    const to = parseFloat(el.dataset.count);
    const dec = (el.dataset.dec | 0);
    const t0 = performance.now(), dur = 900;
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = (to * e).toFixed(dec);
      if (p < 1) requestAnimationFrame(tick); else el.textContent = to.toFixed(dec);
    };
    requestAnimationFrame(tick);
  };
  if ("IntersectionObserver" in window) {
    const co = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { runCount(en.target); co.unobserve(en.target); } });
    }, { rootMargin: "0px 0px -10% 0px" });
    counters.forEach((el) => co.observe(el));
  } else { counters.forEach((el) => { el.textContent = parseFloat(el.dataset.count).toFixed(el.dataset.dec | 0); }); }

  /* ---------- tooltip ---------- */
  const tip = $("#tip");
  const showTip = (html, x, y) => {
    tip.innerHTML = html;
    tip.classList.add("on");
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y - r.height - 12;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top < 8) top = y + 16;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  };
  const hideTip = () => tip.classList.remove("on");

  /* ---------- SVG helpers ---------- */
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };

  const fmt = (v, d) => {
    if (v >= 1000) return v.toFixed(0);
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(d == null ? 1 : d);
    return v.toFixed(d == null ? 2 : d);
  };

  /* Vertical grouped bar chart. opts: {groups, series:[{name,color,values}], unit, log, dec} */
  function groupedBars(host, opts) {
    host.innerHTML = "";
    const W = 760, H = 360, P = { t: 18, r: 14, b: 74, l: 54 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const all = opts.series.flatMap((s) => s.values).filter((v) => v > 0);
    const log = !!opts.log;
    const yMin = log ? Math.min.apply(null, all) * 0.7 : 0;
    const yMax = Math.max.apply(null, all) * 1.08;
    const Y = (v) => {
      if (!log) return P.t + ih - (v / yMax) * ih;
      const a = Math.log10(Math.max(v, yMin)), b0 = Math.log10(yMin), b1 = Math.log10(yMax);
      return P.t + ih - ((a - b0) / (b1 - b0)) * ih;
    };
    // gridlines + y labels
    const ticks = log ? [0.01, 0.03, 0.1, 0.3, 1] : [0, yMax / 4, yMax / 2, (3 * yMax) / 4, yMax];
    ticks.forEach((tv) => {
      if (tv < yMin || tv > yMax) return;
      const y = Y(tv);
      svg.appendChild(el("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y, y2: y }));
      const t = el("text", { x: P.l - 8, y: y + 4, "text-anchor": "end" });
      t.textContent = tv >= 1 ? tv.toFixed(0) : tv.toFixed(2);
      svg.appendChild(t);
    });
    const nS = opts.series.length, nG = opts.groups.length;
    const gw = iw / nG, bw = Math.min(26, (gw * 0.72) / nS);
    opts.groups.forEach((g, gi) => {
      const gx = P.l + gi * gw + gw / 2;
      const total = bw * nS, start = gx - total / 2;
      opts.series.forEach((s, si) => {
        const v = s.values[gi];
        const x = start + si * bw + bw * 0.08;
        const y = Y(v), hgt = Math.max(1, P.t + ih - y);
        const r = el("rect", { class: "bar", x: x, y: y, width: bw * 0.84, height: hgt, rx: 3, fill: cssvar(s.color), style: `--d:${(gi * 60 + si * 25)}ms` });
        r.dataset.tip = `<div><i style="background:${cssvar(s.color)}"></i><b>${s.name}</b></div><div class="r"><span>${g}</span><span>${fmt(v, opts.dec)} ${opts.unit}</span></div>`;
        r.addEventListener("mousemove", (e) => showTip(r.dataset.tip, e.clientX, e.clientY));
        r.addEventListener("mouseleave", hideTip);
        svg.appendChild(r);
      });
      const lab = el("text", { class: "lab", x: gx, y: H - P.b + 22, "text-anchor": "middle" });
      lab.textContent = g;
      svg.appendChild(lab);
    });
    svg.appendChild(el("line", { class: "axis", x1: P.l, x2: W - P.r, y1: P.t + ih, y2: P.t + ih }));
    host.appendChild(svg);
  }

  /* Vertical single-series bars with per-bar colors. opts: {labels, values, colors, unit, dec} */
  function bars(host, opts) {
    host.innerHTML = "";
    const W = 760, H = 360, P = { t: 18, r: 14, b: 64, l: 54 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const vmax = Math.max.apply(null, opts.values) * 1.12;
    const Y = (v) => P.t + ih - (v / vmax) * ih;
    [0, vmax / 4, vmax / 2, (3 * vmax) / 4, vmax].forEach((tv) => {
      const y = Y(tv);
      svg.appendChild(el("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y, y2: y }));
      const t = el("text", { x: P.l - 8, y: y + 4, "text-anchor": "end" });
      t.textContent = tv >= 100 ? tv.toFixed(0) : (tv >= 10 ? tv.toFixed(0) : tv.toFixed(2));
      svg.appendChild(t);
    });
    const n = opts.values.length, gw = iw / n, bw = Math.min(64, gw * 0.5);
    opts.values.forEach((v, i) => {
      const gx = P.l + i * gw + gw / 2, x = gx - bw / 2;
      const y = Y(v), hgt = Math.max(1, P.t + ih - y);
      const col = cssvar(opts.colors[i]);
      const r = el("rect", { class: "bar", x: x, y: y, width: bw, height: hgt, rx: 4, fill: col, style: `--d:${i * 70}ms` });
      r.dataset.tip = `<div><i style="background:${col}"></i><b>${opts.labels[i]}</b></div><div class="r"><span>${opts.metric}</span><span>${fmt(v, opts.dec)} ${opts.unit}</span></div>`;
      r.addEventListener("mousemove", (e) => showTip(r.dataset.tip, e.clientX, e.clientY));
      r.addEventListener("mouseleave", hideTip);
      svg.appendChild(r);
      const val = el("text", { class: "lab w", x: gx, y: y - 6, "text-anchor": "middle" });
      val.textContent = fmt(v, opts.dec);
      svg.appendChild(val);
      const lab = el("text", { class: "lab", x: gx, y: H - P.b + 22, "text-anchor": "middle" });
      lab.textContent = opts.labels[i];
      svg.appendChild(lab);
    });
    svg.appendChild(el("line", { class: "axis", x1: P.l, x2: W - P.r, y1: P.t + ih, y2: P.t + ih }));
    host.appendChild(svg);
  }

  /* ---------- data ---------- */
  const MICRO = {
    groups: ["RMSNorm", "SwiGLU", "RoPE", "Fused CE"],
    series: [
      { name: "PyTorch Eager", color: "--c-eager", values: [0.517, 0.429, 0.673, 0.013] },
      { name: "torch.compile", color: "--c-compile", values: [0.173, 0.252, 0.174, 0.013] },
      { name: "Native Triton", color: "--c-triton", values: [0.160, 0.252, 0.159, 0.010] },
      { name: "KernelLens ORT", color: "--c-kl", values: [0.172, 0.269, 0.180, 0.025] },
    ],
  };
  const SPEED = {
    groups: ["Mamba SSM", "Squared-ReLU attn", "Residual+RMSNorm", "Fused CE (128k)"],
    series: [
      { name: "vs PyTorch Eager", color: "--c-triton", values: [20.73, 6.28, 2.42, 1.86] },
      { name: "vs torch.compile", color: "--c-compile", values: [0.65, 3.33, 1.00, 0.50] },
    ],
  };
  const E2E = {
    labels: ["PyTorch Eager", "torch.compile", "KernelLens ORT", "KernelLens TRT"],
    colors: ["--c-eager", "--c-compile", "--c-triton", "--c-kl"],
    metrics: {
      "TTFT · S=512": { values: [7.85, 7.92, 4.37, 4.14], unit: "ms", dec: 2 },
      "ITL · S=1": { values: [0.866, 0.866, 0.999, 0.905], unit: "ms", dec: 3 },
      "Total · 128 tok": { values: [0.118, 0.118, 0.131, 0.119], unit: "s", dec: 3 },
      "Peak VRAM": { values: [478.0, 473.0, 464.2, 466.2], unit: "MB", dec: 1 },
    },
  };

  const renderMicro = () => groupedBars($("#ch-micro"), Object.assign({ log: true, unit: "ms", dec: 3 }, MICRO));
  const renderSpeed = () => groupedBars($("#ch-speed"), Object.assign({ unit: "×", dec: 2 }, SPEED));

  let e2eMetric = Object.keys(E2E.metrics)[0];
  const renderE2E = () => {
    const m = E2E.metrics[e2eMetric];
    bars($("#ch-e2e"), { labels: E2E.labels, colors: E2E.colors, values: m.values, unit: m.unit, dec: m.dec, metric: e2eMetric });
  };

  /* controls */
  const wireSeg = (sel, onPick) => {
    const seg = $(sel);
    if (!seg) return;
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", seg).forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      onPick(b.dataset.v);
    });
  };
  wireSeg("#e2e-metric", (v) => { e2eMetric = v; renderE2E(); });

  /* install tabs */
  (function tabs() {
    const t = $("#inst-tabs");
    if (!t) return;
    t.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", t).forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      const pip = $("#inst-pip"), src = $("#inst-src");
      if (pip) pip.hidden = b.dataset.v !== "pip";
      if (src) src.hidden = b.dataset.v !== "src";
    });
  })();

  /* hero stage */
  (function stage() {
    const nodes = $$("#stage .node");
    if (!nodes.length) return;
    const tipEl = $("#stage-tip");
    nodes.forEach((n) => {
      const set = () => {
        nodes.forEach((m) => m.classList.toggle("on", m === n));
        tipEl.innerHTML = "<b>" + n.dataset.t + "</b> — " + n.dataset.d;
      };
      n.addEventListener("mouseenter", set);
      n.addEventListener("focus", set);
      n.addEventListener("click", set);
    });
  })();

  /* render all + theme re-render */
  const renderAll = () => { renderMicro(); renderSpeed(); renderE2E(); };
  renderAll();
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(renderAll);
  }
})();