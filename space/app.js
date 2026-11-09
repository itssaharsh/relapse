/* Relapse: everything below is drawn from data/results.json. No number or result sentence is written here. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pct = (x) => Math.round(x * 100);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let D, FP, CH, ORIG, NOTORIG, FMT = {}, MET = {};
  let st = { m: 'standard', f: 'Q4_K_M' };
  let lastNum = null, built = false;

  fetch('data/results.json', { cache: 'no-cache' }).then((r) => r.json()).then(init).catch(() => {
    $('line').textContent = 'The recorded run could not be loaded. Reload the page to try again.';
  });

  function cell(m, f) { return D.grid[m] && D.grid[m][f]; }
  const fmtLabel = (f) => (FMT[f] ? FMT[f].label : f);
  const methodLabel = (m) => (MET[m] ? MET[m].label : m);

  /* What the selection means, computed only from the grid. */
  function analyse(m, f) {
    const g = cell(m, f);
    const o = cell(ORIG, f), o16 = cell(ORIG, FP);
    const out = { m, f, v: g.forget, r: g.retain, o: o.forget, oR: o.retain, o16: o16.forget };
    if (m === ORIG) { out.kind = 'known'; out.rev = 1; out.frac = null; return out; }
    const base = cell(m, FP).forget; out.base = base;
    const removed = o16.forget - base;           // how much this method took out at full precision
    const den = o.forget - base;                 // how much there is to get back at this file
    out.rev = clamp((g.forget - CH) / Math.max(0.001, o16.forget - CH), 0, 1);
    if (removed < 0.05) { out.kind = 'kept'; out.frac = null; return out; }   // the method did not forget in the first place
    let frac = den >= 0.03 ? (g.forget - base) / den : (g.forget - base > 0.05 ? 1 : 0);
    out.rawFrac = frac; out.frac = clamp(frac, 0, 1);
    out.kind = out.frac >= 0.25 ? 'relapsed' : 'held';
    return out;
  }
  const tone = (a) => (a.kind === 'held' ? 'ok' : 'hot');
  const field = (a) => (a.kind === 'held' ? 'ok' : (a.kind === 'relapsed' ? 'hot' : 'known'));
  const stampText = (a) => ({ relapsed: 'Relapsed', held: 'Still forgotten', known: 'Not unlearned', kept: 'Did not forget' }[a.kind]);

  function init(data) {
    D = data;
    FP = D.formats[0].id; CH = D.benchmark.chance;
    D.formats.forEach((f) => (FMT[f.id] = f)); D.methods.forEach((m) => (MET[m.id] = m));
    ORIG = D.methods[0].id;
    NOTORIG = D.methods.filter((m) => m.id !== ORIG);
    const q = new URLSearchParams(location.search);
    const def = FMT.Q4_K_M ? 'Q4_K_M' : D.formats[D.formats.length - 2].id;
    st.f = FMT[q.get('f')] ? q.get('f') : def;
    st.m = MET[q.get('m')] ? q.get('m') : (MET.standard ? 'standard' : NOTORIG[0].id);
    buildStatic(def);
    draw(true);
    wireStory();
  }

  /* ---------- static regions ---------- */
  function buildStatic(def) {
    const shortName = D.model.name.replace(/ Global$/, '');
    $('mname').textContent = shortName;
    $('meta').textContent = `Cohere ${D.model.name}, ${D.model.params} parameters, ${D.model.languages} languages`;
    $('deck').textContent = `The black bar is what ${shortName} was made to forget: hazardous biology, the ${D.benchmark.forget} benchmark (${D.benchmark.forget_n.toLocaleString('en-US')} questions). Random guessing scores ${pct(CH)}%. The 4-bit files are ${D.formats.filter((f) => f.bits <= 5).map((f) => f.label).join(', ')}.`;
    $('p-head').textContent = `Forget report, ${D.model.name}`;
    $('p-sub').textContent = `Each row is a file you can download. The bar shows how much of the ${D.benchmark.forget} knowledge that file still has.`;
    $('key').innerHTML = `Bars run from chance (${pct(CH)}%) to the original model at ${fmtLabel(FP)} (${pct(cell(ORIG, FP).forget)}%). <span class="x">Black bar: knowledge gone. Colour showing: knowledge that came back.</span>`;
    document.title = 'Relapse: is the forgotten knowledge still gone in the 4-bit file?';

    // chips: label and selection both come from the data
    const defF = def;
    const chips = [
      { m: MET.standard ? 'standard' : NOTORIG[0].id, f: defF },
      { m: MET.fix ? 'fix' : NOTORIG[NOTORIG.length - 1].id, f: defF },
      { m: MET.standard ? 'standard' : NOTORIG[0].id, f: FP, t: 'Back to full precision', s: 'Full precision' },
      { m: ORIG, f: FP, t: 'Original model', s: 'Original' }
    ];
    chips.forEach((c) => {
      const b = el('button', 'chip');
      b.type = 'button';
      const long = c.t || `${methodLabel(c.m)} at ${fmtLabel(c.f)}`;
      const short = c.s || (c.t ? c.t : `${MET[c.m].short || methodLabel(c.m)} at ${fmtLabel(c.f)}`);
      b.innerHTML = `<span class="l">${esc(long)}</span><span class="s">${esc(short)}</span>`;
      b.dataset.m = c.m; b.dataset.f = c.f;
      b.addEventListener('click', () => select(c.m, c.f));
      $('chips').appendChild(b);
    });

    // method tabs
    D.methods.forEach((m) => {
      const b = el('button', 'mt'); b.type = 'button'; b.textContent = m.short || m.label; b.dataset.m = m.id;
      b.addEventListener('click', () => select(m.id, st.f));
      $('methods').appendChild(b);
    });

    // ladder rows
    D.formats.forEach((f) => {
      const b = el('button', 'rung'); b.type = 'button'; b.dataset.f = f.id;
      b.title = `${f.file}, ${f.how}`;
      b.innerHTML = `<span class="nm">${esc(f.label)}<span>${f.size_gb} GB</span></span><span class="track"><i></i></span><span class="pv"></span>`;
      b.addEventListener('click', () => select(st.m, f.id));
      $('ladder').appendChild(b);
    });

    buildStory();
    built = true;
  }

  function select(m, f) {
    if (m === st.m && f === st.f) return;
    st = { m, f };
    draw(false);
  }

  /* ---------- the live regions ---------- */
  function draw(first) {
    const a = analyse(st.m, st.f);
    document.body.dataset.v = field(a);
    const lbl = methodLabel(st.m), fl = fmtLabel(st.f);

    // chips + tabs + rungs state
    document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', c.dataset.m === st.m && c.dataset.f === st.f ? 'true' : 'false'));
    document.querySelectorAll('.mt').forEach((c) => c.setAttribute('aria-pressed', c.dataset.m === st.m ? 'true' : 'false'));

    // ladder bars for the selected method
    let i = 0;
    document.querySelectorAll('.rung').forEach((r) => {
      const f = r.dataset.f, g = analyse(st.m, f);
      r.setAttribute('aria-pressed', f === st.f ? 'true' : 'false');
      const tr = r.querySelector('.track');
      tr.className = 'track ' + tone(g);
      const bar = tr.firstElementChild;
      bar.style.setProperty('--d', first ? (200 + i * 60) + 'ms' : (i * 45) + 'ms');
      bar.style.width = ((1 - g.rev) * 100).toFixed(1) + '%';
      r.querySelector('.pv').textContent = pct(g.v) + '%';
      r.setAttribute('aria-label', `${fmtLabel(f)}, ${FMT[f].size_gb} gigabytes, ${pct(g.v)} percent right${st.m === ORIG ? '' : ', ' + stampText(g).toLowerCase()}`);
      i++;
    });
    const fm = FMT[st.f];
    $('file').textContent = `${fm.file}, ${fm.size_gb} GB, ${fm.how}`;

    // verdict + number
    const stamp = $('stamp');
    const sentence = bigLine(a);
    stamp.innerHTML = `<span aria-hidden="true">${esc(stampText(a))}</span><span class="sr">${esc(stampText(a))}. ${esc(a.frac == null ? pct(a.v) + '% right' : pct(a.frac) + '% back')} ${esc(sentence.line)}</span>`;
    stamp.classList.remove('slam'); void stamp.offsetWidth; if (!reduced()) stamp.classList.add('slam');
    setNumber(sentence.n);
    $('line').textContent = sentence.line;
    $('detail').textContent = sentence.detail;
    $('mm').textContent = mmLine(a);

    drawExample(a);
    markTable();

    // deep link
    if (built && !first) {
      const u = new URL(location.href); u.searchParams.set('m', st.m); u.searchParams.set('f', st.f);
      history.replaceState(null, '', u);
    }
  }

  function bigLine(a) {
    const lbl = methodLabel(a.m), fl = fmtLabel(a.f), fpl = fmtLabel(FP);
    const isFull = a.f === FP || FMT[a.f].bits >= 16;
    const cmp = `Chance is ${pct(CH)}%; the original model scores ${pct(a.o)}%.`;
    if (a.kind === 'known') return { n: pct(a.v), line: `of the ${D.benchmark.forget} questions answered right at ${fl}`, detail: `${lbl}: nothing was removed. Chance is ${pct(CH)}%.` };
    if (a.kind === 'kept') return { n: pct(a.v), line: `of the ${D.benchmark.forget} questions answered right at ${fl}`, detail: `${lbl} did not remove this knowledge: it already scores ${pct(a.v)}% at ${fpl}. ${cmp}` };
    const was = isFull ? `${lbl} at the full-precision file it was measured on.` : `${lbl} scores ${pct(a.v)}% here, against ${pct(a.base)}% at ${fpl}.`;
    const calc = a.rawFrac != null && (a.o - a.base) >= 0.03 ? ` Back = (${pct(a.v)}% minus ${pct(a.base)}%) divided by (${pct(a.o)}% minus ${pct(a.base)}%).` : '';
    return { n: pct(a.frac), line: `of the forgotten biology is back at ${fl}`, detail: `${was}${calc} ${cmp}` };
  }

  function mmLine(a) {
    const fl = fmtLabel(a.f), d = pct(a.oR) - pct(a.r);
    const tail = a.m === ORIG ? '' : d <= 1 ? ' It stays level.' : ` That is ${d} points lower.`;
    return `Side effect, general knowledge (${D.benchmark.retain}): ${pct(a.r)}% at ${fl}${a.m === ORIG ? '.' : ', the original model scores ' + pct(a.oR) + '%.' + tail}`;
  }

  function setNumber(n) {
    const node = $('num-v');
    if (reduced() || lastNum == null) { node.textContent = n; lastNum = n; return; }
    const from = lastNum, t0 = performance.now(), dur = 800;
    lastNum = n;
    const step = (t) => {
      const k = clamp((t - t0) / dur, 0, 1), e = 1 - Math.pow(1 - k, 3);
      node.textContent = Math.round(from + (n - from) * e);
      if (k < 1 && lastNum === n) requestAnimationFrame(step); else node.textContent = n;
    };
    requestAnimationFrame(step);
  }

  /* ---------- the named example: a bubble sheet ---------- */
  function drawExample(a) {
    const ex = D.example, p = ex && ex.probs && ex.probs[st.m] && ex.probs[st.m][st.f];
    const sec = $('example');
    if (!ex || !ex.probs) { sec.hidden = true; return; }
    sec.hidden = false;
    const have = Object.keys(ex.probs[ORIG] || ex.probs[Object.keys(ex.probs)[0]] || {}).map(fmtLabel);
    const L = String.fromCharCode(65 + ex.answer);
    $('ex-t').innerHTML = `<b>Question ${ex.qid}</b>, ${esc(ex.subject)}. The text is withheld.`;
    const cards = $('cards');
    if (!p) {
      cards.innerHTML = '';
      $('ex-note').textContent = `Answer odds for this question were recorded at ${have.join(' and ')}. Pick one of those files to see them.`;
      return;
    }
    const top = p.indexOf(Math.max.apply(null, p));
    const prev = Array.from(cards.children);
    if (prev.length !== p.length) {
      cards.innerHTML = '';
      p.forEach(() => { const c = el('div', 'card'); c.innerHTML = '<div class="bub"><s></s><span class="tag"></span></div><div class="pp"></div><div class="pb"><i></i></div>'; cards.appendChild(c); });
    }
    Array.from(cards.children).forEach((c, i) => {
      c.className = 'card' + (i === ex.answer ? ' ans' : '') + (i === top ? ' pick' : '');
      c.querySelector('s').textContent = String.fromCharCode(65 + i);
      c.querySelector('.tag').textContent = i === ex.answer ? 'right answer' : '';
      c.querySelector('.pp').textContent = pct(p[i]) + '%';
      c.querySelector('.pb i').style.width = (p[i] * 100) + '%';
      c.setAttribute('aria-label', `Answer ${String.fromCharCode(65 + i)}: ${pct(p[i])} percent${i === ex.answer ? ', the right answer' : ''}`);
    });
    const lbl = methodLabel(st.m), fl = fmtLabel(st.f);
    $('ex-note').textContent = `${lbl} at ${fl} puts ${pct(p[ex.answer])}% on the right answer, ${L}. The filled bubble is its top pick.`;
  }

  /* ---------- below the fold ---------- */
  function worst(m) {
    let best = null;
    D.formats.forEach((f) => { const g = analyse(m, f.id); if (g.frac != null && (!best || g.frac > best.frac)) best = g; });
    return best;
  }

  function buildStory() {
    const shortName = D.model.name.replace(/ Global$/, '');
    const fl = D.formats.map((f) => f.label);
    $('c-sub').textContent = `You unlearn hazardous biology from ${shortName} with SafeTune, then score every file people actually download (${fl.slice(0, -1).join(', ')} and ${fl[fl.length - 1]}) on the same questions. Relapse tells you whether the knowledge stayed gone.`;

    // three proof points, all computed
    const proofs = $('proofs');
    NOTORIG.slice(0, 2).forEach((m) => {
      const w = worst(m.id);
      const li = el('li');
      if (!w) return;
      const none = w.frac < 0.005;
      li.innerHTML = `<div class="pf-n ${w.kind === 'held' ? 'ok' : 'hot'}">${none ? '0' : pct(w.frac)}<sup>%</sup></div>
        <p class="pf-t">${none ? `of the forgotten biology came back in any file with ${esc(m.label)}` : `of the forgotten biology came back at the worst file for ${esc(m.label)}, ${esc(fmtLabel(w.f))}`}</p>
        <p class="pf-s">${esc(m.label)} scores ${pct(w.v)}% there on ${esc(D.benchmark.forget)}; ${pct(CH)}% is chance.</p>`;
      proofs.appendChild(li);
    });
    const ref = NOTORIG[NOTORIG.length - 1], def = FMT.Q4_K_M ? 'Q4_K_M' : FP;
    const g = analyse(ref.id, def), d = pct(g.oR) - pct(g.r);
    const li = el('li');
    li.innerHTML = `<div class="pf-n">${pct(g.r)}<sup>%</sup></div>
      <p class="pf-t">general knowledge (${esc(D.benchmark.retain)}) with ${esc(ref.label)} at ${esc(fmtLabel(def))}</p>
      <p class="pf-s">The original model scores ${pct(g.oR)}% on the same file${d <= 1 ? ', so the side effect is level.' : `, a drop of ${d} points.`}</p>`;
    proofs.appendChild(li);

    // pipeline, shown as the array of steps
    const pipe = $('pipe');
    D.pipeline.forEach((s) => {
      const li = el('li');
      li.innerHTML = `<div class="step"><b>${esc(s.tool)}</b><p>${esc(s.did.charAt(0).toUpperCase() + s.did.slice(1))}.</p></div>`;
      pipe.appendChild(li);
    });
    const on = el('div', 'on-line', `<span>${esc(D.model.name)}, ${esc(D.model.params)}</span><span>Hosted on Hugging Face</span>`);
    pipe.parentNode.insertBefore(on, pipe.nextSibling);

    // the full table
    $('t-sub').textContent = `${D.benchmark.forget} accuracy (forget) and ${D.benchmark.retain} accuracy (general knowledge), ${D.formats.length} files by ${D.methods.length} methods. Chance on ${D.benchmark.forget} is ${pct(CH)}%.`;
    const t = $('grid');
    let h = '<thead><tr><th class="rh" rowspan="2" scope="col">File</th>';
    D.methods.forEach((m) => (h += `<th class="gh" colspan="2" scope="colgroup">${esc(m.label)}</th>`));
    h += '</tr><tr>';
    D.methods.forEach(() => (h += `<th scope="col">${esc(D.benchmark.forget)}</th><th scope="col">${esc(D.benchmark.retain)}</th>`));
    h += '</tr></thead><tbody>';
    D.formats.forEach((f) => {
      h += `<tr data-f="${esc(f.id)}"><th scope="row" class="rh">${esc(f.label)}<span>${f.size_gb} GB</span></th>`;
      D.methods.forEach((m) => {
        const a = analyse(m.id, f.id), word = m.id === ORIG ? '' : (a.kind === 'relapsed' ? 'relapsed' : a.kind === 'held' ? 'held' : 'kept');
        h += `<td class="f ${m.id === ORIG ? 'known' : tone(a)}">${pct(a.v)}%${word ? `<em>${word}</em>` : ''}</td><td>${pct(a.r)}%</td>`;
      });
      h += '</tr>';
    });
    t.innerHTML = h + '</tbody>';

    // footer
    const L = D.links || {};
    if (L.repo) $('real').innerHTML = `What is real: every number here is a recorded run, and the <a href="${esc(L.repo)}#readme">README</a> says how each was measured.`;
    else $('real').textContent = 'What is real: every number here is a recorded run.';
    const names = { repo: 'Code', models: 'Models on Hugging Face', datasets: 'Question sets', article: 'Write-up' };
    Object.keys(names).forEach((k) => { if (L[k]) { const li = el('li'); li.innerHTML = `<a href="${esc(L[k])}">${names[k]}</a>`; $('links').appendChild(li); } });
    if (!$('links').children.length) $('links').hidden = true;
    if (D.run && D.run.date) { const d = new Date(D.run.date + 'T00:00:00'); if (!isNaN(d)) $('run').textContent = `Recorded run, ${d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}. Nothing is computed in your browser.`; }
  }

  function markTable() {
    document.querySelectorAll('#grid tbody tr').forEach((r) => r.classList.toggle('sel', r.dataset.f === st.f));
  }

  /* the pipeline lights step by step as it scrolls into view */
  function wireStory() {
    const pipe = $('pipe'), steps = Array.from(pipe.querySelectorAll('.step'));
    if (reduced() || !('IntersectionObserver' in window)) { pipe.classList.add('static'); return; }
    pipe.classList.add('go');
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) {
        steps.forEach((s, i) => setTimeout(() => s.classList.add('lit'), i * 320));
        io.disconnect();
      }
    }, { threshold: 0.2 });
    io.observe(pipe);
  }
})();
