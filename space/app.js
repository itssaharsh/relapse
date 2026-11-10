/* Relapse: everything below is drawn from data/results.json. No number or result sentence is written here. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pct = (x) => Math.round(x * 100);
  const p1 = (x) => (x * 100).toFixed(1);
  const sgn = (x) => (x < 0 ? '−' : '+') + Math.abs(x * 100).toFixed(1);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const RISE = 0.05, GREW = 0.02;           // a relapse is +5 points over fp16; the price "grew" when the gap is 2+ points wider than at fp16

  let D, FP, CH, ORIG, NOTORIG, FMT = {}, MET = {}, LANG = {}, STATS = {}, DEF = {};
  let st = { m: 'standard', f: 'NF4', lang: '' };
  let lastNum = null, built = false, topCache = {};

  fetch('data/results.json', { cache: 'no-cache' }).then((r) => r.json()).then(init).catch(() => {
    $('line').textContent = 'The recorded run could not be loaded. Reload the page to try again.';
  });

  const cell = (m, f) => (D.grid[m] && D.grid[m][f]) || null;
  const fmtLabel = (f) => (FMT[f] ? FMT[f].label : f);
  const methodLabel = (m) => (MET[m] ? MET[m].label : m);
  const methodShort = (m) => (MET[m] ? (MET[m].short || MET[m].label) : m);
  const forgetOf = (c, lang) => (!c ? null : lang ? (c.probes && c.probes[lang] != null ? c.probes[lang] : null) : c.forget);
  const langName = (l) => (LANG[l] || l);

  /* paired AuditKIT rows, matched to a method by their accuracies (the rows carry a model name, not a method id) */
  function indexStats() {
    const near = (a, b) => a != null && b != null && Math.abs(a - b) < 0.004;
    const toM = {};
    (D.stats || []).filter((s) => s.kind !== 'price' && s.a && s.b).forEach((s) => {
      const fA = s.a.slice(s.a.lastIndexOf('-') + 1), fB = s.b.slice(s.b.lastIndexOf('-') + 1);
      Object.keys(D.grid).forEach((m) => {
        const ca = cell(m, fA), cb = cell(m, fB);
        if (ca && cb && near(ca.forget, s.acc_a) && near(cb.forget, s.acc_b)) { STATS['f|' + m + '|' + fB] = s; if (s.model) toM[s.model] = m; }
      });
    });
    (D.stats || []).filter((s) => s.kind === 'price' && s.b).forEach((s) => {
      const fB = s.b.slice(s.b.lastIndexOf('-') + 1), m = s.method || toM[s.model];
      if (m) STATS['p|' + m + '|' + fB] = s;
    });
  }

  /* What the selection means, computed only from the grid. */
  function analyse(m, f, lang) {
    const c = cell(m, f), a = { m, f, lang };
    const v = forgetOf(c, lang);
    if (!c || v == null) { a.kind = 'missing'; a.field = 'known'; return a; }
    const o = cell(ORIG, f), oV = forgetOf(o, lang);
    a.v = v; a.r = c.retain; a.o = oV; a.oR = o ? o.retain : null; a.pc = c.forget_pcorrect;
    const top = topOf(lang);
    a.rev = clamp((v - CH) / Math.max(0.001, top - CH), 0, 1);
    if (m === ORIG) { a.fk = 'known'; a.field = 'known'; return (a.kind = 'known', a); }
    const b = cell(m, FP), bV = forgetOf(b, lang);
    a.base = bV;
    const s = lang ? null : STATS['f|' + m + '|' + f]; a.stat = s;
    if (bV == null) a.fk = 'unref';
    else a.fk = (v - bV >= RISE && (!s || s.p_value < 0.05)) ? 'relapsed' : 'still';
    a.price = a.oR != null ? a.oR - c.retain : null;
    const o16 = cell(ORIG, FP);
    a.price16 = o16 && b ? o16.retain - b.retain : null;
    a.priceStat = STATS['p|' + m + '|' + f];
    a.grew = a.price != null && a.price16 != null ? a.price - a.price16 >= GREW : null;
    a.kind = 'unlearned';
    a.field = a.grew === true ? 'hot' : a.grew === false ? 'ok' : 'known';
    return a;
  }
  function topOf(lang) {
    const k = lang || '_';
    if (topCache[k] == null) {
      let t = 0; D.formats.forEach((f) => { const v = forgetOf(cell(ORIG, f.id), lang); if (v != null && v > t) t = v; });
      topCache[k] = t || 0.6;
    }
    return topCache[k];
  }

  function init(data) {
    D = data;
    FP = D.formats[0].id; CH = D.benchmark.chance;
    D.formats.forEach((f) => (FMT[f.id] = f)); D.methods.forEach((m) => (MET[m.id] = m));
    LANG = (D.languages && D.languages.labels) || {};
    ORIG = D.methods[0].id; NOTORIG = D.methods.filter((m) => m.id !== ORIG);
    indexStats();
    const four = D.formats.filter((f) => f.bits <= 5);
    DEF.f4 = (FMT.NF4 ? 'NF4' : four.length ? four[four.length - 1].id : FP);
    DEF.m = (MET.standard && cell('standard', DEF.f4)) ? 'standard' : (NOTORIG.find((m) => cell(m.id, DEF.f4)) || NOTORIG[0]).id;
    const q = new URLSearchParams(location.search);
    st.f = FMT[q.get('f')] ? q.get('f') : DEF.f4;
    st.m = MET[q.get('m')] ? q.get('m') : DEF.m;
    st.lang = LANG[q.get('lang')] ? q.get('lang') : '';
    buildStatic();
    draw(true);
    wireStory();
  }

  /* ---------- static regions ---------- */
  function buildStatic() {
    const shortName = D.model.name.replace(/ Global$/, '');
    $('mname').textContent = shortName;
    $('meta').textContent = `Cohere ${D.model.name}, ${D.model.params} parameters, ${D.model.languages} languages`;
    $('deck').textContent = `The black bar is what ${shortName} was made to forget: hazardous biology, the ${D.benchmark.forget} benchmark (${D.benchmark.forget_n.toLocaleString('en-US')} questions).`;
    $('p-head').textContent = `Forget report, ${D.model.name}`;
    document.title = 'Relapse: is the forgotten knowledge still gone in the 4-bit file?';

    const four = DEF.f4, fl = fmtLabel(four);
    const m2 = (MET.circuit && 'circuit') || NOTORIG[NOTORIG.length - 1].id;
    const q4 = FMT.Q4_K_M ? 'Q4_K_M' : four;
    const mQ = [DEF.m].concat(NOTORIG.map((m) => m.id)).find((m) => cell(m, q4)) || DEF.m;
    const swL = LANG.sw ? 'sw' : Object.keys(LANG)[0];
    const chips = [
      { m: DEF.m, f: four, l: `${methodShort(DEF.m)} at 4-bit (${fl})`, s: `${methodShort(DEF.m)} at ${fl}` },
      { m: m2, f: four, l: `${methodShort(m2)} at 4-bit (${fl})`, s: `${methodShort(m2)} at ${fl}` },
      { m: mQ, f: q4, l: `Llama.cpp ${fmtLabel(q4)} file`, s: `${fmtLabel(q4)} file` }
    ];
    if (swL) chips.push({ m: DEF.m, f: four, lang: swL, l: `Ask in ${langName(swL)}`, s: langName(swL) });
    chips.push({ m: ORIG, f: FP, l: 'Original model', s: 'Original' });
    chips.forEach((c) => {
      const b = el('button', 'chip'); b.type = 'button';
      b.innerHTML = `<span class="l">${esc(c.l)}</span><span class="s">${esc(c.s)}</span>`;
      b.dataset.m = c.m; b.dataset.f = c.f; b.dataset.lang = c.lang || '';
      b.addEventListener('click', () => select(c.m, c.f, c.lang || ''));
      $('chips').appendChild(b);
    });

    D.methods.filter((m) => m.tab !== false).forEach((m) => {
      const b = el('button', 'mt'); b.type = 'button'; b.textContent = methodShort(m.id); b.dataset.m = m.id; b.title = m.label;
      b.addEventListener('click', () => select(m.id, st.f, st.lang));
      $('methods').appendChild(b);
    });

    D.formats.forEach((f) => {
      const b = el('button', 'rung'); b.type = 'button'; b.dataset.f = f.id; b.title = `${f.file}, ${f.how}`;
      b.innerHTML = `<span class="nm">${esc(f.label)}<span>${f.size_gb} GB</span></span>
        <span class="bars"><span class="track"><i></i></span><span class="gk"><b></b><u></u></span></span>
        <span class="pv"><b></b><b></b></span>`;
      b.addEventListener('click', () => select(st.m, f.id, st.lang));
      $('ladder').appendChild(b);
    });

    // language row: the benchmark itself, then each probe set
    const langs = $('langs');
    [''].concat(Object.keys(LANG)).forEach((l) => {
      const b = el('button', 'lg'); b.type = 'button'; b.dataset.l = l;
      b.innerHTML = `<span class="ln">${esc(l ? LANG[l] : D.benchmark.forget)}</span><span class="lv"></span>`;
      b.addEventListener('click', () => select(st.m, st.f, st.lang === l ? '' : l));
      langs.appendChild(b);
    });

    buildStory();
    built = true;
  }

  function select(m, f, lang) {
    if (m === st.m && f === st.f && lang === st.lang) return;
    st = { m, f, lang };
    draw(false);
  }

  /* ---------- draw ---------- */
  const stampText = (a) => (a.kind === 'unlearned' ? (a.grew === true ? 'Price grew' : a.grew === false ? 'Price held' : 'Price unknown') : a.kind === 'known' ? 'Not unlearned' : 'Not measured');
  const tagText = (a) => (a.kind !== 'unlearned' ? '' : ({ relapsed: 'Relapsed', still: 'Still forgotten', unref: 'No fp16 reference' }[a.fk]));

  function draw(first) {
    const a = analyse(st.m, st.f, st.lang);
    document.body.dataset.v = a.field;
    const lbl = methodLabel(st.m), fl = fmtLabel(st.f), fpl = fmtLabel(FP), bench = D.benchmark.forget, inLang = st.lang ? ` in ${langName(st.lang)}` : '';

    document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', c.dataset.m === st.m && c.dataset.f === st.f && c.dataset.lang === st.lang ? 'true' : 'false'));
    document.querySelectorAll('.mt').forEach((c) => c.setAttribute('aria-pressed', c.dataset.m === st.m ? 'true' : 'false'));
    document.querySelectorAll('.lg').forEach((c) => c.setAttribute('aria-pressed', c.dataset.l === st.lang ? 'true' : 'false'));

    // ladder: top bar hazardous biology (black = gone), lower bar general knowledge (hatched = the price)
    const maxR = Math.max.apply(null, D.formats.map((f) => (cell(ORIG, f.id) || { retain: 0 }).retain).concat([0.1]));
    let i = 0;
    document.querySelectorAll('.rung').forEach((r) => {
      const f = r.dataset.f, g = analyse(st.m, f, st.lang), c = cell(st.m, f);
      r.setAttribute('aria-pressed', f === st.f ? 'true' : 'false');
      const tr = r.querySelector('.track'), gk = r.querySelector('.gk'), pv = r.querySelectorAll('.pv b');
      const miss = g.kind === 'missing';
      r.classList.toggle('miss', miss);
      tr.className = 'track ' + (g.fk === 'relapsed' || g.kind === 'known' ? 'hot' : 'ok');
      const bar = tr.firstElementChild, d = first ? (200 + i * 60) : i * 45;
      bar.style.setProperty('--d', d + 'ms');
      bar.style.width = miss ? '100%' : ((1 - g.rev) * 100).toFixed(1) + '%';
      const b = gk.querySelector('b'), u = gk.querySelector('u');
      b.style.setProperty('--d', d + 'ms'); u.style.setProperty('--d', d + 'ms');
      if (c) {
        b.style.width = (c.retain / maxR * 100).toFixed(1) + '%';
        const gap = g.oR != null && st.m !== ORIG ? Math.max(0, g.oR - c.retain) : 0;
        u.style.left = (c.retain / maxR * 100).toFixed(1) + '%'; u.style.width = (gap / maxR * 100).toFixed(1) + '%';
        u.className = g.grew === true ? 'hot' : 'ok';
      } else { b.style.width = '0%'; u.style.width = '0%'; }
      pv[0].textContent = miss ? 'n/m' : pct(g.v) + '%';
      pv[1].textContent = c ? pct(c.retain) + '%' : 'n/m';
      r.setAttribute('aria-label', miss ? `${fmtLabel(f)}, not measured for this method` : `${fmtLabel(f)}, ${FMT[f].size_gb} gigabytes, ${bench} ${pct(g.v)} percent, general knowledge ${pct(c.retain)} percent`);
      i++;
    });
    const fm = FMT[st.f];
    $('file').textContent = `${fm.file}, ${fm.size_gb} GB, ${fm.how}`;

    // language row
    document.querySelectorAll('.lg').forEach((b) => {
      const l = b.dataset.l, c = cell(st.m, st.f), o = cell(ORIG, st.f);
      const v = forgetOf(c, l), ov = forgetOf(o, l);
      const lv = b.querySelector('.lv');
      lv.innerHTML = v == null ? '<span class="nm2">not measured</span>' : `<b>${pct(v)}%</b>${st.m !== ORIG && ov != null ? `<span>original ${pct(ov)}%</span>` : ''}`;
      b.setAttribute('aria-label', `${l ? LANG[l] : bench}: ${v == null ? 'not measured' : pct(v) + ' percent' + (st.m !== ORIG && ov != null ? ', original ' + pct(ov) + ' percent' : '')}`);
    });
    $('key').innerHTML = `Top bar: ${esc(st.lang ? langName(st.lang) + ' questions' : bench)}, from chance (${pct(CH)}%) to the original model. Black = knowledge gone. <span class="x">Lower bar: general knowledge (${esc(D.benchmark.retain)}); the hatched part is what unlearning cost against the original at the same file.</span>`;

    // stamps
    const stamps = $('stamps'), stamp = $('stamp'), tag = $('tag2');
    stamp.textContent = stampText(a); tag.textContent = tagText(a); tag.hidden = !tag.textContent;
    tag.dataset.k = a.fk || '';
    const sr = `${stampText(a)}. ${tagText(a)}.`;
    let sc = stamps.querySelector('.sr'); if (!sc) { sc = el('span', 'sr'); stamps.appendChild(sc); }
    sc.textContent = sr + ' ' + (a.kind === 'unlearned' && a.price != null ? `General knowledge price ${(a.price * 100).toFixed(1)} points at ${fl}.` : '');
    stamp.classList.remove('slam'); void stamp.offsetWidth; if (!reduced()) stamp.classList.add('slam');

    // the answer
    const ans = $('ans'), line = $('line'), detail = $('detail'), price = $('price');
    price.classList.remove('na');
    if (a.kind === 'missing') {
      ans.textContent = 'Not measured yet.';
      line.textContent = st.lang && cell(st.m, st.f) ? `${lbl} has no ${langName(st.lang)} questions scored at ${fl}.` : `${lbl} has not been scored at ${fl} yet.`;
      detail.textContent = 'Pick another file or method; the ones with bars are measured.';
      $('p-lab').textContent = ''; setNumber(null, ''); $('pline').textContent = ''; price.classList.add('na');
    } else if (a.kind === 'known') {
      ans.textContent = 'It never forgot.';
      line.textContent = `${lbl} scores ${p1(a.v)}% on the ${bench} questions${inLang} at ${fl}. Guessing scores ${pct(CH)}%.`;
      detail.textContent = '';
      $('p-lab').textContent = 'General knowledge';
      setNumber(a.r * 100, '%'); $('pline').textContent = `${D.benchmark.retain} accuracy of the original model at ${fl}. Every price on this page is measured against it.`;
    } else {
      ans.textContent = a.fk === 'relapsed' ? 'No, it came back.' : a.fk === 'still' ? 'Yes, still gone.' : 'Scored, no fp16 reference.';
      const against = a.f !== FP && a.base != null ? `, against ${p1(a.base)}% at ${fpl}` : '';
      line.textContent = `${methodShort(st.m)} scores ${p1(a.v)}% on ${bench}${inLang} at ${fl}${against}. The original scores ${a.o != null ? p1(a.o) + '%' : 'n/m'} there; guessing scores ${pct(CH)}%.`;
      detail.textContent = (a.stat ? auditLine(a.stat, a, 'forget accuracy') + ' ' : '') + `Relapse here means ${pct(RISE)} or more points back over ${fpl}.`;
      $('p-lab').textContent = 'The price';
      if (a.price == null) { setNumber(null, ''); $('pline').textContent = `General knowledge at ${fl} is not measured for ${lbl}.`; }
      else {
        setNumber(a.price * 100, ' pts');
        let t = `of general knowledge (${D.benchmark.retain}) lost at ${fl}: ${p1(a.r)}% where the original scores ${p1(a.oR)}%.`;
        if (a.f !== FP && a.price16 != null) {
          const dd = a.price - a.price16;
          t += ` At ${fpl} the gap was ${(a.price16 * 100).toFixed(1)} points, so the price ${a.grew ? 'grew by ' + (dd * 100).toFixed(1) : Math.abs(dd) < 0.005 ? 'held' : dd < 0 ? 'shrank by ' + (-dd * 100).toFixed(1) : 'held (' + (dd * 100).toFixed(1) + ' wider)'}.`;
        } else if (a.f === FP || a.price16 == null) t += ` This is the full-precision gap the 4-bit files are compared with.`;
        $('pline').textContent = t + (a.priceStat ? ' ' + auditLine(a.priceStat, a, 'the price') : '');
      }
    }

    markTable();
    drawExample(a);

    if (built && !first) {
      const u = new URL(location.href); u.searchParams.set('m', st.m); u.searchParams.set('f', st.f);
      if (st.lang) u.searchParams.set('lang', st.lang); else u.searchParams.delete('lang');
      history.replaceState(null, '', u);
    }
  }

  function auditLine(s, a, what) {
    const ci = s.ci95 || [];
    const fA = s.a.slice(s.a.lastIndexOf('-') + 1);
    const sig = s.p_value < 0.05 ? 'a real shift' : 'no detectable shift';
    return `AuditKIT, paired on ${s.n.toLocaleString('en-US')} questions: ${what} moves ${sgn(s.delta)} points from ${fmtLabel(fA)} to ${fmtLabel(a.f)} (95% interval ${sgn(ci[0])} to ${sgn(ci[1])}, p = ${s.p_value < 0.001 ? '<0.001' : s.p_value.toFixed(2)}): ${sig}.`;
  }

  function setNumber(n, unit) {
    const node = $('num-v'); $('num-u').textContent = unit || '';
    if (n == null) { node.textContent = '–'; lastNum = null; return; }
    const fix = (x) => (unit === '%' ? String(Math.round(x)) : x.toFixed(1).replace('-', '−'));
    if (reduced() || lastNum == null) { node.textContent = fix(n); lastNum = n; return; }
    const from = lastNum, t0 = performance.now(), dur = 800; lastNum = n;
    const step = (t) => {
      const k = clamp((t - t0) / dur, 0, 1), e = 1 - Math.pow(1 - k, 3);
      node.textContent = fix(from + (n - from) * e);
      if (k < 1 && lastNum === n) requestAnimationFrame(step); else node.textContent = fix(n);
    };
    requestAnimationFrame(step);
  }

  /* ---------- the named question: a bubble sheet ---------- */
  function drawExample(a) {
    const ex = D.example, sec = $('example');
    if (!ex || !ex.probs) { sec.hidden = true; return; }
    sec.hidden = false;
    let p = ex.probs[st.m] && ex.probs[st.m][st.f], ref = false;
    if (!p && ex.probs[ORIG] && ex.probs[ORIG][st.f]) { p = ex.probs[ORIG][st.f]; ref = true; }
    const L = String.fromCharCode(65 + ex.answer);
    $('ex-t').innerHTML = `<b>Question ${ex.qid}</b>, ${esc(ex.set || D.benchmark.forget)}. The text is withheld; the right answer is ${L}.`;
    const cards = $('cards');
    if (!p) {
      cards.innerHTML = '';
      const have = Object.keys(ex.probs[ORIG] || {}).map(fmtLabel);
      $('ex-note').textContent = `Answer odds for this question are not measured for ${methodShort(st.m)} at ${fmtLabel(st.f)}.`;
      return;
    }
    const top = p.indexOf(Math.max.apply(null, p));
    if (cards.children.length !== p.length) {
      cards.innerHTML = '';
      p.forEach(() => { const c = el('div', 'card'); c.innerHTML = '<div class="bub"><s></s><span class="tag"></span></div><div class="pp"></div><div class="pb"><i></i></div>'; cards.appendChild(c); });
    }
    Array.from(cards.children).forEach((c, i) => {
      c.className = 'card' + (i === ex.answer ? ' ans' : '') + (i === top ? ' pick' : '');
      c.querySelector('s').textContent = String.fromCharCode(65 + i);
      c.querySelector('.tag').textContent = i === ex.answer ? 'correct' : '';
      c.querySelector('.pp').textContent = pct(p[i]) + '%';
      c.querySelector('.pb i').style.width = (p[i] * 100) + '%';
      c.setAttribute('aria-label', `Answer ${String.fromCharCode(65 + i)}: ${pct(p[i])} percent${i === ex.answer ? ', the right answer' : ''}`);
    });
    if (ref) { $('ex-note').textContent = `Not scored for ${methodShort(st.m)} at ${fmtLabel(st.f)} yet. This is the original model on the same file: ${pct(p[ex.answer])}% on ${L}.`; return; }
    $('ex-note').textContent = `${methodShort(st.m)} at ${fmtLabel(st.f)} puts ${pct(p[ex.answer])}% on ${L}. The filled bubble is its top pick; the original puts ${(ex.probs[ORIG] && ex.probs[ORIG][st.f]) ? pct(ex.probs[ORIG][st.f][ex.answer]) + '%' : 'n/m'} on ${L}.`;
  }

  /* ---------- below the fold ---------- */
  function buildStory() {
    const shortName = D.model.name.replace(/ Global$/, '');
    $('c-sub').textContent = `You unlearn hazardous biology from ${shortName} with SafeTune, then score every file people actually download on two questions: is it still forgotten, and what did forgetting cost?`;

    // three proofs: all measured cells of the unlearned methods
    const cells = [];
    NOTORIG.forEach((m) => D.formats.forEach((f) => { const a = analyse(m.id, f.id, ''); if (a.kind === 'unlearned') cells.push(a); }));
    const still = cells.filter((a) => a.fk === 'still').length;
    const proofs = $('proofs');
    const add = (n, cls, t, s) => { const li = el('li'); li.innerHTML = `<div class="pf-n ${cls}">${n}</div><p class="pf-t">${t}</p><p class="pf-s">${s}</p>`; proofs.appendChild(li); };
    if (cells.length) {
      const maxF = cells.reduce((x, y) => (y.v > x.v ? y : x));
      add(`${still}<sup>of ${cells.length}</sup>`, still === cells.length ? 'ok' : 'hot', `measured unlearned files are still forgotten`, `${cells.length} method and file pairs scored on ${esc(D.benchmark.forget)}. The highest is ${p1(maxF.v)}% (${esc(methodShort(maxF.m))}, ${esc(fmtLabel(maxF.f))}), against ${pct(maxF.o)}% for the original and ${pct(CH)}% for guessing.`);
    }
    let lm = null;
    const keys = Object.keys(LANG);
    NOTORIG.forEach((m) => D.formats.forEach((f) => keys.forEach((l) => { const v = forgetOf(cell(m.id, f.id), l), o = forgetOf(cell(ORIG, f.id), l); if (v != null && o != null && (!lm || v > lm.v)) lm = { v, o, m: m.id, f: f.id, l }; })));
    if (lm) add(`${pct(lm.v)}<sup>%</sup>`, lm.v - CH < 0.2 ? 'ok' : 'hot', `is the most any unlearned file scores in ${keys.map((l) => esc(langName(l))).join(', ')}`, `That is ${esc(langName(lm.l))} with ${esc(methodShort(lm.m))} at ${esc(fmtLabel(lm.f))}; the original scores ${pct(lm.o)}% on the same questions.`);
    const four = DEF.f4, parts = [];
    [DEF.m].concat(NOTORIG.map((m) => m.id).filter((m) => m !== DEF.m && (m === 'circuit'))).forEach((m) => { const a = analyse(m, four, ''); if (a.kind === 'unlearned' && a.price != null) parts.push(a); });
    if (parts.length) {
      const a = parts[0];
      add(`${(a.price * 100).toFixed(1)}<sup>pts</sup>`, a.grew ? 'hot' : 'ok', `of general knowledge lost by ${esc(methodShort(a.m))} at ${esc(fmtLabel(four))}`, parts.map((b) => `${esc(methodShort(b.m))}: ${(b.price * 100).toFixed(1)} points at ${esc(fmtLabel(four))}, ${b.price16 != null ? (b.price16 * 100).toFixed(1) + ' at ' + esc(fmtLabel(FP)) : 'no fp16 reference'}`).join('. ') + '.');
    }

    // pipeline
    const pipe = $('pipe');
    D.pipeline.forEach((s) => {
      const li = el('li');
      li.innerHTML = `<div class="step"><b>${esc(s.tool)}</b><p>${esc(s.did.charAt(0).toUpperCase() + s.did.slice(1))}.</p></div>`;
      pipe.appendChild(li);
    });
    pipe.parentNode.insertBefore(el('div', 'on-line', `<span>${esc(D.model.name)}, ${esc(D.model.params)}</span><span>Hosted on Hugging Face</span>`), pipe.nextSibling);

    buildLayers();

    // table
    $('t-sub').textContent = `${D.benchmark.forget} accuracy (forget, chance ${pct(CH)}%) and ${D.benchmark.retain} accuracy (general knowledge). Under each general-knowledge score: points lost against the original at the same file.`;
    let h = '<thead><tr><th class="rh" rowspan="2" scope="col">File</th>';
    D.methods.forEach((m) => (h += `<th class="gh" colspan="2" scope="colgroup">${esc(methodShort(m.id))}</th>`));
    h += '</tr><tr>';
    D.methods.forEach(() => (h += `<th scope="col">${esc(D.benchmark.forget)}</th><th scope="col">${esc(D.benchmark.retain)}</th>`));
    h += '</tr></thead><tbody>';
    D.formats.forEach((f) => {
      h += `<tr data-f="${esc(f.id)}"><th scope="row" class="rh">${esc(f.label)}<span>${f.size_gb} GB</span></th>`;
      D.methods.forEach((m) => {
        const a = analyse(m.id, f.id, '');
        if (a.kind === 'missing') { h += '<td class="f na">not measured</td><td class="na">not measured</td>'; return; }
        const word = a.kind === 'unlearned' ? (a.fk === 'relapsed' ? 'relapsed' : a.fk === 'still' ? 'still gone' : '') : '';
        const tn = a.kind === 'known' ? 'known' : a.fk === 'relapsed' ? 'hot' : 'ok';
        const pr = a.kind === 'unlearned' && a.price != null ? `<em>price ${(a.price * 100).toFixed(1)}</em>` : '';
        const rt = a.kind === 'unlearned' ? (a.grew === true ? 'hot' : a.grew === false ? 'ok' : '') : '';
        h += `<td class="f ${tn}">${pct(a.v)}%${word ? `<em>${word}</em>` : ''}</td><td class="${rt}">${pct(a.r)}%${pr}</td>`;
      });
      h += '</tr>';
    });
    $('grid').innerHTML = h + '</tbody>';

    const L = D.links || {};
    $('real').innerHTML = L.repo ? `What is real: every number here is a recorded run, and the <a href="${esc(L.repo)}#readme">README</a> says how each was measured.` : 'What is real: every number here is a recorded run.';
    const names = { repo: 'Code', models: 'Models on Hugging Face', datasets: 'Question sets', article: 'Write-up' };
    Object.keys(names).forEach((k) => { if (L[k]) { const li = el('li'); li.innerHTML = `<a href="${esc(L[k])}">${names[k]}</a>`; $('links').appendChild(li); } });
    if (!$('links').children.length) $('links').hidden = true;
    if (D.run && D.run.date) { const d = new Date(D.run.date + 'T00:00:00'); if (!isNaN(d)) $('run').textContent = `Recorded run, ${d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}.`; }
  }

  /* CircuitKIT attribution per layer, default vs CircuitKIT layers highlighted */
  function buildLayers() {
    const C = D.circuit; if (!C || !C.mlp_abs || !C.mlp_abs.length) return;
    $('layer-wrap').hidden = false;
    const v = C.mlp_abs, mx = Math.sqrt(Math.max.apply(null, v)), dl = C.default_rmu_layers || [], cl = C.circuit_rmu_layers || [];
    const box = $('lbars');
    v.forEach((x, i) => {
      const inD = dl.indexOf(i) >= 0, inC = cl.indexOf(i) >= 0;
      const b = el('div', 'lb' + (inD ? ' d' : '') + (inC ? ' c' : ''));
      b.innerHTML = `<i style="height:${(Math.sqrt(x) / mx * 100).toFixed(1)}%"></i>`;
      b.title = `Layer ${i}: ${x.toFixed(2)}`;
      box.appendChild(b);
    });
    const sum = (ls) => ls.reduce((s, l) => s + v[l], 0), tot = v.reduce((s, x) => s + x, 0);
    const lr = (ls) => ls.length > 1 ? `${ls[0]} to ${ls[ls.length - 1]}` : String(ls[0]);
    $('l-head').textContent = `Where the biology answers live in ${D.model.name.replace(/ Global$/, '')}`;
    const ax = $('laxis'); [0, 5, 10, 15, 20, 25, 30, 35].filter((i) => i < v.length).forEach((i) => { const s = el('span', '', String(i)); s.style.left = (i / v.length * 100) + '%'; ax.appendChild(s); });
    box.setAttribute('aria-label', `Attribution per layer for ${v.length} layers. Default layers ${lr(dl)}, CircuitKIT layers ${lr(cl)}.`);
    $('lcap').innerHTML = `<span class="sw d"></span>SafeTune default layers ${lr(dl)} hold ${p1(sum(dl) / tot)}% of the attribution. <span class="sw c"></span>The layers CircuitKIT found, ${lr(cl)}, hold ${p1(sum(cl) / tot)}%. Bar height is the square root of each layer's attribution (${C.n_examples} ${esc(C.task)} examples), so small layers stay visible.`;
  }

  function markTable() {
    document.querySelectorAll('#grid tbody tr').forEach((r) => r.classList.toggle('sel', r.dataset.f === st.f));
  }

  function wireStory() {
    const pipe = $('pipe'), steps = Array.from(pipe.querySelectorAll('.step'));
    if (reduced() || !('IntersectionObserver' in window)) { pipe.classList.add('static'); return; }
    pipe.classList.add('go');
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { steps.forEach((s, i) => setTimeout(() => s.classList.add('lit'), i * 320)); io.disconnect(); }
    }, { threshold: 0.2 });
    io.observe(pipe);
  }
})();
