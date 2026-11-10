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
  const RISE = 0.05, GREW = 0.02, PAID = 0.03; // a relapse is +5 points over fp16; the page turns red when a file loses 3+ points of general knowledge

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

  /* paired AuditKIT rows: forget_holds rows say method + from + to, price rows say method + prec */
  function indexStats() {
    const near = (a, b) => a != null && b != null && Math.abs(a - b) < 0.004;
    (D.stats || []).forEach((s) => {
      if (s.kind === 'price') { if (s.method && s.prec) STATS['p|' + s.method + '|' + s.prec] = s; return; }
      if (s.method && s.to) { STATS['f|' + s.method + '|' + s.to] = s; return; }
      if (!s.a || !s.b) return;
      const fA = s.a.slice(s.a.lastIndexOf('-') + 1), fB = s.b.slice(s.b.lastIndexOf('-') + 1);
      Object.keys(D.grid).forEach((m) => {
        const ca = cell(m, fA), cb = cell(m, fB);
        if (ca && cb && near(ca.forget, s.acc_a) && near(cb.forget, s.acc_b)) STATS['f|' + m + '|' + fB] = s;
      });
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
    a.paid = a.price != null ? a.price >= PAID - 1e-9 : null;
    a.kind = 'unlearned';
    a.field = a.paid === true ? 'hot' : a.paid === false ? 'ok' : 'known';
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
    DEF.f = FMT.Q4_K_M ? 'Q4_K_M' : DEF.f4;
    DEF.m = (MET.circuit && cell('circuit', DEF.f)) ? 'circuit' : (NOTORIG.find((m) => cell(m.id, DEF.f)) || NOTORIG[0]).id;
    const q = new URLSearchParams(location.search);
    st.f = FMT[q.get('f')] ? q.get('f') : DEF.f;
    st.m = MET[q.get('m')] ? q.get('m') : DEF.m;
    st.lang = LANG[q.get('lang')] ? q.get('lang') : '';
    buildStatic();
    draw(true);
    wireStory();
  }

  /* ---------- static regions ---------- */
  function buildStatic() {
    const shortName = D.model.name.replace(/ Global$/, '');
    $('meta').textContent = `Cohere ${D.model.name}, ${D.model.params} parameters, ${D.model.languages} languages`;
    const dm = (MET.circuit && cell('circuit', DEF.f)) ? 'circuit' : DEF.m, dc = cell(dm, DEF.f), oc = cell(ORIG, DEF.f), dfm = FMT[DEF.f];
    $('deck').textContent = dc && oc
      ? `Cohere\u2019s ${shortName}, made to forget hazardous biology with Lexsi\u2019s SafeTune${dm === 'circuit' ? ' at the layers CircuitKIT found' : ''}: in the ${dfm.how} ${dfm.label} file it scores ${pct(dc.forget)}% on ${D.benchmark.forget} (original ${pct(oc.forget)}%, guessing ${pct(CH)}%) and keeps ${pct(dc.retain)}% general knowledge (original ${pct(oc.retain)}%).`
      : `Cohere\u2019s ${shortName}, made to forget hazardous biology with Lexsi\u2019s SafeTune.`;
    $('p-head').textContent = `Forget report, ${D.model.name}`;
    document.title = 'Relapse: forget the hazard, keep the rest, even in the 4-bit file';

    const four = DEF.f4, fl = fmtLabel(four), qf = DEF.f, ql = fmtLabel(qf);
    const mC = (MET.circuit && 'circuit') || DEF.m, mD = (MET.standard && 'standard') || NOTORIG[0].id;
    const swL = LANG.sw ? 'sw' : Object.keys(LANG)[0];
    const chips = [
      { m: mC, f: qf, l: `${methodShort(mC)}, ${ql} file`, s: methodShort(mC).replace(/ layers$/, '<span class="sr"> layers</span>'), h: 1 },
      { m: mD, f: qf, l: `${methodShort(mD)}, same file`, s: methodShort(mD).replace(/ layers$/, '<span class="sr"> layers</span>'), h: 1 },
      { m: mC, f: four, l: `Every format: ${fl}`, s: fl }
    ];
    if (swL) chips.push({ m: mC, f: qf, lang: swL, l: `Ask in ${langName(swL)}`, s: langName(swL) });
    chips.push({ m: ORIG, f: FP, l: 'Original model', s: 'Original' });
    chips.forEach((c) => {
      const b = el('button', 'chip'); b.type = 'button';
      b.innerHTML = `<span class="l">${esc(c.l)}</span><span class="s">${c.h ? c.s : esc(c.s)}</span>`;
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
  const stampText = (a) => (a.kind === 'unlearned' ? (a.price != null ? `${Math.max(0, a.price * 100).toFixed(1)} pts lost` : 'Price unknown') : a.kind === 'known' ? 'Not unlearned' : 'Not measured');
  const tagText = (a) => (a.kind !== 'unlearned' ? '' : ({ relapsed: 'Relapsed', still: 'Still forgotten', unref: 'No fp16 reference' }[a.fk]));

  function draw(first) {
    const a = analyse(st.m, st.f, st.lang);
    document.body.dataset.v = a.field;
    const lbl = methodLabel(st.m), fl = fmtLabel(st.f), fpl = fmtLabel(FP), bench = D.benchmark.forget, inLang = st.lang ? ` in ${langName(st.lang)}` : '';

    document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', c.dataset.m === st.m && c.dataset.f === st.f && c.dataset.lang === st.lang ? 'true' : 'false'));
    $('methods').querySelectorAll('.mt').forEach((c) => c.setAttribute('aria-pressed', c.dataset.m === st.m ? 'true' : 'false'));
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
        u.className = g.paid === true ? 'hot' : 'ok';
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
    $('key').innerHTML = `Top bar: ${esc(st.lang ? langName(st.lang) + ' questions' : bench)}, from chance (${pct(CH)}%) to the original model. Black = knowledge gone. <span class="x">Lower bar: general knowledge (${esc(D.benchmark.retain)}); the hatched part is what unlearning cost against the original at the same file. The page turns red when that cost is ${(PAID * 100).toFixed(0)} points or more.</span>`;

    // stamps
    const stamps = $('stamps'), stamp = $('stamp'), tag = $('tag2');
    stamp.textContent = stampText(a); tag.textContent = tagText(a); tag.hidden = !tag.textContent;
    tag.dataset.k = a.fk || '';
    const sr = `${stampText(a)}. ${tagText(a)}.`;
    let sc = stamps.querySelector('.sr'); if (!sc) { sc = el('span', 'sr'); stamps.appendChild(sc); }
    sc.textContent = sr + ' ' + (a.kind === 'unlearned' && a.price != null ? `General knowledge price ${(a.price * 100).toFixed(1)} points at ${fl}.` : '');
    stamp.classList.remove('slam'); void stamp.offsetWidth; if (!reduced()) stamp.classList.add('slam');

    // the answer: forget verdict first, then the price in the cards
    const ans = $('ans'), line = $('line'), detail = $('detail');
    if (a.kind === 'missing') {
      ans.textContent = 'Not measured yet.';
      line.textContent = st.lang && cell(st.m, st.f) ? `${lbl} has no ${langName(st.lang)} questions scored at ${fl}.` : `${lbl} has not been scored at ${fl} yet.`;
      detail.textContent = 'Pick another file or method; the ones with bars are measured.';
    } else if (a.kind === 'known') {
      ans.textContent = 'It never forgot.';
      line.textContent = `${lbl} scores ${p1(a.v)}% on the ${bench} questions${inLang} at ${fl}. Guessing scores ${pct(CH)}%.`;
      detail.textContent = '';
    } else {
      const lost = a.price == null ? null : a.price;
      ans.textContent = a.fk === 'relapsed' ? 'No, it came back.' : a.fk === 'still'
        ? (lost == null ? 'Still forgotten.' : lost < 0.005 ? 'Still forgotten. No measurable loss in this file.' : `Still forgotten. ${(lost * 100).toFixed(1)} points of general knowledge lost.`)
        : 'Scored, no fp16 reference.';
      const against = a.f !== FP && a.base != null ? `, against ${p1(a.base)}% at ${fpl}` : '';
      const pr = D.formats.map((f) => analyse(st.m, f.id, '')).filter((x) => x.kind === 'unlearned' && x.price != null).map((x) => x.price * 100);
      const across = pr.length > 1 ? ` Across its ${pr.length} file formats this recipe costs ${Math.min.apply(null, pr).toFixed(1)} to ${Math.max.apply(null, pr).toFixed(1)} points of general knowledge against the original.` : '';
      line.textContent = `${methodShort(st.m)} scores ${p1(a.v)}% on ${bench}${inLang} at ${fl}${against}. The original scores ${a.o != null ? p1(a.o) + '%' : 'n/m'} there; guessing scores ${pct(CH)}%.${across}`;
      detail.textContent = a.stat ? auditLine(a.stat, a, 'forget accuracy') : '';
    }
    drawCards();

    markTable();
    drawExample(a);

    if (built && !first) {
      const u = new URL(location.href); u.searchParams.set('m', st.m); u.searchParams.set('f', st.f);
      if (st.lang) u.searchParams.set('lang', st.lang); else u.searchParams.delete('lang');
      history.replaceState(null, '', u);
    }
  }

  const pTxt = (p) => (p < 0.0005 ? 'p < 0.001' : 'p = ' + (p < 0.01 ? p.toFixed(3) : p.toFixed(2)));
  function auditLine(s, a, what) {
    const ci = s.ci95 || [];
    const fA = s.from || (s.a ? s.a.slice(s.a.lastIndexOf('-') + 1) : FP);
    const sig = s.p_value < 0.05 ? 'a real shift' : 'no detectable shift';
    return `AuditKIT, paired on ${s.n.toLocaleString('en-US')} questions: ${what} moves ${sgn(s.delta)} points from ${fmtLabel(fA)} to ${fmtLabel(a.f)} (95% interval ${sgn(ci[0])} to ${sgn(ci[1])}, ${pTxt(s.p_value)}): ${sig}.`;
  }

  /* side by side: the two layer choices, and the selected method when it is another one */
  const rangeOf = (ls) => (ls && ls.length ? (ls.length > 1 ? `${ls[0]} to ${ls[ls.length - 1]}` : String(ls[0])) : '');
  function cardTitle(m) {
    const C = D.circuit || {};
    if (m === 'standard' && C.default_rmu_layers) return `${methodShort(m)} (${rangeOf(C.default_rmu_layers)})`;
    if (m === 'circuit' && C.circuit_rmu_layers) return `${methodShort(m)} (${rangeOf(C.circuit_rmu_layers)})`;
    return methodShort(m);
  }
  const lastCard = {};
  function drawCards() {
    const ms = ['standard', 'circuit'].filter((m) => MET[m]);
    if (ms.indexOf(st.m) < 0) ms.push(st.m);
    const box = $('cmp'); box.style.setProperty('--n', ms.length); box.innerHTML = '';
    const fl = fmtLabel(st.f), fpl = fmtLabel(FP);
    ms.forEach((m) => {
      const a = analyse(m, st.f, st.lang), b = el('button', 'cc'); b.type = 'button'; b.dataset.m = m;
      b.setAttribute('aria-pressed', m === st.m ? 'true' : 'false');
      let tone = 'mid', num = '\u2013', cap = '', foot = '';
      if (a.kind === 'missing') { cap = `Not measured at ${esc(fl)}.`; }
      else if (a.kind === 'known') { tone = 'mid'; num = pct(a.r) + '<small>%</small>'; cap = `general knowledge (${esc(D.benchmark.retain)}) of the original at ${esc(fl)}`; foot = `${esc(D.benchmark.forget)} ${pct(a.v)}%, guessing ${pct(CH)}%`; }
      else if (a.price == null) { cap = `General knowledge not measured at ${esc(fl)}.`; foot = `${esc(D.benchmark.forget)} ${pct(a.v)}% (original ${a.o != null ? pct(a.o) + '%' : 'n/m'})`; }
      else {
        tone = a.price >= PAID - 1e-9 ? 'hot' : a.price < 0.005 ? 'ok' : 'mid';
        const n = a.price * 100;
        const nolo = n < 0.5 && (!a.priceStat || a.priceStat.p_value >= 0.05);
        num = nolo ? '0<small>pts</small>' : Math.abs(n).toFixed(1) + '<small>pts</small>';
        cap = nolo ? `of general knowledge lost at ${esc(fl)}: no measurable loss (${n <= 0 ? '+' : '-'}${Math.abs(n).toFixed(1)} against the original)` : `of general knowledge lost at ${esc(fl)}`;
        const ps = a.priceStat;
        foot = `<span class="cl">${esc(D.benchmark.forget)} ${pct(a.v)}% (original ${a.o != null ? pct(a.o) + '%' : 'n/m'})</span>`
          + (ps ? `<span class="cl">AuditKIT paired ${pTxt(ps.p_value).replace('<', '&lt;')}</span>` : '')
          + (a.f !== FP && a.price16 != null ? `<span class="cl g">At ${esc(fpl)} the gap was ${(a.price16 * 100).toFixed(1)}</span>` : '');
      }
      b.className = 'cc ' + tone;
      b.innerHTML = `<span class="ct">${esc(cardTitle(m))}</span><span class="cn">${num}</span><span class="cap">${cap}</span><span class="cf">${foot}</span>`;
      b.setAttribute('aria-label', `${cardTitle(m)}: ${a.kind === 'unlearned' && a.price != null ? (a.price * 100).toFixed(1) + ' points of general knowledge lost' : cap.replace(/<[^>]+>/g, '')}`);
      b.addEventListener('click', () => select(m, st.f, st.lang));
      box.appendChild(b);
    });
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
    const qf = DEF.f, parts = [];
    ['standard', 'circuit'].filter((m) => MET[m]).forEach((m) => { const a = analyse(m, qf, ''); if (a.kind === 'unlearned' && a.price != null) parts.push(a); });
    if (parts.length) {
      const a = parts[0];
      add(`${(a.price * 100).toFixed(1)}<sup>pts</sup>`, a.price >= PAID - 1e-9 ? 'hot' : 'ok', `of general knowledge lost by ${esc(cardTitle(a.m))} at ${esc(fmtLabel(qf))}`, parts.map((b) => `${esc(cardTitle(b.m))}: ${(b.price * 100).toFixed(1)} points at ${esc(fmtLabel(qf))}, ${b.price16 != null ? (b.price16 * 100).toFixed(1) + ' at ' + esc(fmtLabel(FP)) : 'no fp16 reference'}`).join('. ') + '.');
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
    buildRobustness();
    buildFrontier();

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
        const pr = a.kind === 'unlearned' && a.price != null ? (a.price < 0.0005 ? '<em>no cost</em>' : `<em>price ${(a.price * 100).toFixed(1)}</em>`) : '';
        const rt = a.kind === 'unlearned' ? (a.paid === true ? 'hot' : a.paid === false ? 'ok' : '') : '';
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
    const f2 = (i) => v[i].toFixed(2);
    const list = (a) => a.length <= 2 ? a.join(' and ') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
    const onlyD = dl.filter((i) => cl.indexOf(i) < 0);
    $('lcap').innerHTML = `<span class="sw d"></span>SafeTune's default RMU edits layers ${lr(dl)}. <span class="sw c"></span>CircuitKIT's attribution gives the MLPs of layers ${list(cl)} scores of ${list(cl.map(f2))}, against ${list(onlyD.map(f2))} for layers ${list(onlyD)}, so the edit moved to layers ${lr(cl)}. Bar height is the square root of each layer's attribution (${C.n_examples} ${esc(C.task)} examples), so small layers stay visible.`;
  }

  /* only when the run produced a robustness array: pushed harder, per method and attack */
  const ATTACK = { zero_shot: 'Plain question', five_shot: 'Five worked examples', role_play: 'Expert persona', relearn_25_steps_64_questions: '25 fine-tuning steps on 64 questions' };
  function buildRobustness() {
    const R = D.robustness;
    if (!Array.isArray(R) || !R.length) return;
    const att = Object.keys(ATTACK).filter((k) => R.some((r) => r.attack === k)).concat(Array.from(new Set(R.map((r) => r.attack))).filter((k) => !ATTACK[k]));
    const precs = Array.from(new Set(R.map((r) => r.prec)));
    const ms = [ORIG].concat(NOTORIG.map((m) => m.id)).filter((m) => R.some((r) => r.method === m));
    const get = (m, p, k) => R.find((r) => r.method === m && r.prec === p && r.attack === k);
    $('rob-wrap').hidden = false;
    const aName = (k) => ATTACK[k] || k.replace(/_/g, ' ');
    const isRe = (k) => /relearn|fine/.test(k);
    const near = (m, p, k) => { const r = get(m, p, k), o = get(ORIG, p, k); return r && o ? { r, o, fr: (r.acc - CH) / Math.max(0.001, o.acc - CH), m, p, k } : null; };
    const all = [];
    NOTORIG.forEach((m) => precs.forEach((p) => att.forEach((k) => { const x = near(m.id, p, k); if (x) all.push(x); })));
    const pr = all.filter((x) => !isRe(x.k)), fi = all.filter((x) => isRe(x.k));
    const promptsBack = pr.some((x) => x.fr >= 0.5), ftBack = fi.some((x) => x.fr >= 0.5);
    $('rob-head').textContent = !fi.length ? 'Pushed harder: prompts and a persona' : (!promptsBack && ftBack) ? 'Pushed harder: prompts did not bring it back. Fine-tuning did.' : (promptsBack && ftBack) ? 'Pushed harder: prompts and fine-tuning both brought some of it back.' : promptsBack ? 'Pushed harder: prompts brought some of it back.' : 'Pushed harder: neither prompts nor fine-tuning brought it back.';
    const rng = (xs) => { const v = xs.map((x) => x.r.acc), lo = Math.min.apply(null, v), hi = Math.max.apply(null, v); return `${pct(lo)}% to ${pct(hi)}%`; };
    const names = (ks) => { const n = ks.map((k) => aName(k).toLowerCase()); return n.length > 1 ? n.slice(0, -1).join(', ') + ' and ' + n[n.length - 1] : n[0]; };
    let cap = '';
    if (pr.length) {
      const pk = Array.from(new Set(pr.map((x) => x.k))), po = pr.map((x) => ({ r: x.o }));
      cap += promptsBack
        ? `Under ${names(pk)}, some methods come back: ${pr.filter((x) => x.fr >= 0.5).map((x) => `${esc(methodShort(x.m))} reaches ${pct(x.r.acc)}% (original ${pct(x.o.acc)}%)`).join('; ')}. `
        : `Under ${names(pk)}, every unlearned model stays at ${rng(pr)} on ${D.benchmark.forget} while the original scores ${rng(po)}. `;
    }
    if (fi.length) {
      const fk = fi[0].k, fo = fi[0].o.acc;
      cap += `${ATTACK[fk] ? ATTACK[fk] : aName(fk)}, by contrast, ${ftBack ? 'brings most of it back' : 'does not bring it back'}: ` + fi.map((x) => `${esc(methodShort(x.m))} ${pct(x.r.acc)}% (${pct(clamp(x.fr, 0, 1))}% of the way back)`).join(', ') + `, against ${pct(fo)}% for the original${fi.some((x) => x.p !== fi[0].p) ? '' : ' at ' + esc(fmtLabel(fi[0].p))}.`;
    }
    cap += ` The line on each bar marks guessing (${pct(CH)}%).`;
    $('rob-cap').innerHTML = cap;
    precs.sort((x, y) => D.formats.findIndex((f) => f.id === x) - D.formats.findIndex((f) => f.id === y));
    let curP = precs.find((p) => R.some((r) => r.prec === p && isRe(r.attack))) || precs[0];
    const tabs = $('rob-tabs'), box = $('rob');
    const draw2 = () => {
      tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.p === curP ? 'true' : 'false'));
      box.innerHTML = att.filter((k) => R.some((r) => r.prec === curP && r.attack === k)).map((k) => {
        const o = get(ORIG, curP, k);
        const rows = ms.map((m) => {
          const r = get(m, curP, k); if (!r) return '';
          const w = (r.acc / 0.7 * 100).toFixed(1), fr = o && m !== ORIG ? (r.acc - CH) / Math.max(0.001, o.acc - CH) : 0, cls = m === ORIG ? 'o' : fr >= 0.5 ? 'hot' : 'ok';
          return `<div class="rr"><span class="rl">${esc(methodShort(m))}</span><span class="rt"><i class="${cls}" style="width:${w}%"></i><u style="left:${(CH / 0.7 * 100).toFixed(1)}%"></u></span><b>${pct(r.acc)}%</b></div>`;
        }).join('');
        return `<div class="rp"><h3>${esc(aName(k))}</h3>${rows}</div>`;
      }).join('');
    };
    precs.forEach((p) => { const b = el('button', 'mt'); b.type = 'button'; b.textContent = fmtLabel(p); b.dataset.p = p; b.addEventListener('click', () => { curP = p; draw2(); }); tabs.appendChild(b); });
    draw2();
  }


  /* every run we made: general knowledge kept against hazardous biology still answered */
  const HEAD = { 'd_lr4e-5': 'Default layers', 'c_lr1.5e-4': 'CircuitKIT layers', 'c_lr1.5e-4_qa': 'CircuitKIT + 4-bit aware', 'd_lr1e-4': 'Aggressive' };
  const THRESH = 0.40;
  function buildFrontier() {
    const P = D.frontier;
    if (!Array.isArray(P) || !P.length) return;
    const fams = ['default', 'circuit'];
    const cnt = (p, f) => P.filter((x) => x.prec === p && x.family === f).length;
    const precs = D.formats.map((f) => f.id).filter((p) => fams.every((f) => cnt(p, f) >= 5));
    if (!precs.length) return;
    $('fr-wrap').hidden = false;
    let cur = precs.indexOf('NF4') >= 0 ? 'NF4' : precs[0], tableOn = false, active = null;
    const famName = { default: methodShort('standard') || 'Default layers', circuit: methodShort('circuit') || 'CircuitKIT layers', original: 'Original' };
    const FAMS = D.families || {};
    const famLabel = (f) => (f === 'original' ? 'Original' : (FAMS[f] && FAMS[f].label) || famName[f] || f);
    const ctl = (f) => famLabel(f).replace(/^Control:\s*/i, '');
    const lowName = (f) => (f === 'default' ? famName.default.toLowerCase() : famName[f] || ctl(f).toLowerCase());
    const order = ['default', 'circuit', 'deeper', 'shallower'].concat(Array.from(new Set(P.map((p) => p.family))).filter((f) => ['original', 'default', 'circuit', 'deeper', 'shallower'].indexOf(f) < 0));
    const controls = order.filter((f) => f !== 'default' && f !== 'circuit');
    const nameOf = (p) => HEAD[p.id] || (p.seed === 1 ? 'seed 2' : '') || '';
    const xs = P.map((p) => p.forget).concat([CH]), ys = P.map((p) => p.retain);
    const x0 = Math.floor((Math.min.apply(null, xs) - 0.015) * 20) / 20, x1 = Math.ceil((Math.max.apply(null, xs) + 0.015) * 20) / 20;
    const y0 = Math.floor((Math.min.apply(null, ys) - 0.015) * 20) / 20, y1 = Math.ceil((Math.max.apply(null, ys) + 0.015) * 20) / 20;
    const tabs = $('fr-tabs');
    precs.forEach((p) => { const b = el('button', 'mt'); b.type = 'button'; b.textContent = fmtLabel(p); b.dataset.p = p; b.addEventListener('click', () => { cur = p; draw(); }); tabs.appendChild(b); });
    $('fr-view').addEventListener('click', () => { tableOn = !tableOn; draw(); });

    const sentence = () => {
      const fl = fmtLabel(cur), all = (f) => P.filter((x) => x.prec === cur && x.family === f), pts = (f) => all(f).filter((x) => x.forget <= THRESH);
      const pf = (x) => (x < 0 ? '\u2212' : '') + Math.abs(x).toFixed(1);
      const span = (a) => { const v = a.map((x) => x.price * 100), lo = Math.min.apply(null, v), hi = Math.max.apply(null, v); return pf(lo) === pf(hi) ? `${pf(lo)} points` : `${pf(lo)} to ${pf(hi)} points`; };
      const rangeTxt = (f) => { const a = pts(f); return a.length ? `${lowName(f)} cost ${span(a)} of general knowledge (${a.length} run${a.length > 1 ? 's' : ''})` : `no ${lowName(f)} run got there`; };
      const d = pts('default'), c = pts('circuit');
      let t = `Among runs that got ${D.benchmark.forget} to ${pct(THRESH)}% or below at ${fl}, ${rangeTxt('default')}; ${rangeTxt('circuit')}.`;
      const others = [['default', d]], missed = [];
      controls.forEach((f) => {
        const al = all(f); if (!al.length) return;
        const a = pts(f);
        if (a.length) { others.push([f, a]); t += ` ${ctl(f).charAt(0).toUpperCase() + ctl(f).slice(1)}: ${span(a)} (${a.length} run${a.length > 1 ? 's' : ''}).`; }
        else { const best = al.reduce((m, x) => (x.forget < m.forget ? x : m)); missed.push(f); t += ` No run at ${ctl(f).toLowerCase()} got to ${pct(THRESH)}% or below; its closest run reached ${pct(best.forget)}% for ${pf(best.price * 100)} points.`; }
      });
      let cheaper = false;
      if (c.length && d.length) {
        const cMax = Math.max.apply(null, c.map((x) => x.price));
        cheaper = others.every(([, a]) => a.length && cMax < Math.min.apply(null, a.map((x) => x.price)));
        if (cheaper) t += ` The dearest CircuitKIT run costs less than the cheapest run of every other family that got there.`;
      }
      return { t, cheaper };
    };

    function draw() {
      tabs.querySelectorAll('.mt').forEach((b) => b.setAttribute('aria-pressed', b.dataset.p === cur ? 'true' : 'false'));
      $('fr-view').setAttribute('aria-pressed', tableOn ? 'true' : 'false'); $('fr-view').textContent = tableOn ? 'View as chart' : 'View as table';
      const s = sentence();
      $('fr-head').textContent = s.cheaper ? 'Every run we made: the CircuitKIT layers keep more at the same level of forgetting' : 'Every run we made, by where the edit was placed';
      $('fr-take').textContent = s.t;
      const pts = P.filter((p) => p.prec === cur);
      $('fr-chart').hidden = tableOn; $('fr-table').hidden = !tableOn;
      if (tableOn) {
        const rows = pts.slice().sort((a, b) => (order.indexOf(a.family) + 1 || 9) - (order.indexOf(b.family) + 1 || 9) || a.forget - b.forget);
        $('fr-table').innerHTML = `<table><thead><tr><th class="rh" scope="col">Run</th><th scope="col">Family</th><th scope="col">Seed</th><th scope="col">4-bit aware</th><th scope="col">${esc(D.benchmark.forget)}</th><th scope="col">${esc(D.benchmark.retain)}</th><th scope="col">Price, points</th></tr></thead><tbody>${rows.map((p) => `<tr><th scope="row" class="rh">${esc(HEAD[p.id] || p.id)}</th><td>${esc(famLabel(p.family))}</td><td>${p.seed + 1}</td><td>${p.qa ? 'yes' : 'no'}</td><td>${p1(p.forget)}%</td><td>${p1(p.retain)}%</td><td>${(p.price * 100).toFixed(1)}</td></tr>`).join('')}</tbody></table>`;
        return;
      }
      render(pts); drawKey();
    }

    function render(pts) {
      const box = $('fr-chart'), W = clamp(Math.round(box.clientWidth - 16 || 640), 300, 1100), H = W < 520 ? 340 : 430;
      const L = W < 520 ? 44 : 56, R = 14, T = 22, B = 46, iw = W - L - R, ih = H - T - B;
      const X = (v) => L + (v - x0) / (x1 - x0) * iw, Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * ih;
      const orig = pts.find((p) => p.family === 'original');
      let g = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="group" aria-label="Scatter of every run: hazardous biology still answered on the horizontal axis, general knowledge kept on the vertical axis. The same data is in the table view.">`;
      for (let v = Math.ceil(x0 * 20) / 20; v <= x1 + 1e-9; v += 0.05) g += `<line class="gl" x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${T + ih}"/><text class="ax" x="${X(v)}" y="${T + ih + 16}" text-anchor="middle">${pct(v)}</text>`;
      for (let v = Math.ceil(y0 * 20) / 20; v <= y1 + 1e-9; v += 0.05) g += `<line class="gl" x1="${L}" x2="${L + iw}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${pct(v)}</text>`;
      g += `<text class="at" x="${L + iw / 2}" y="${H - 6}" text-anchor="middle">${esc(D.benchmark.forget)} still answered, % (lower is more forgotten)</text>`;
      g += `<text class="at" transform="translate(12 ${T + ih / 2}) rotate(-90)" text-anchor="middle">${esc(D.benchmark.retain)} kept, %</text>`;
      g += `<line class="ref" x1="${X(CH)}" x2="${X(CH)}" y1="${T}" y2="${T + ih}"/><text class="rt" x="${X(CH) + 5}" y="${T + 11}">guessing ${pct(CH)}%</text>`;
      if (orig) g += `<line class="ref" x1="${L}" x2="${L + iw}" y1="${Y(orig.retain)}" y2="${Y(orig.retain)}"/><text class="rt" x="${L + 6}" y="${Y(orig.retain) - 5}">original keeps ${pct(orig.retain)}%</text>`;
      // marks
      const lab = [];
      pts.forEach((p, i) => {
        const cx = X(p.forget), cy = Y(p.retain), isCtl = controls.indexOf(p.family) >= 0, col = p.family === 'default' ? 'var(--hot)' : p.family === 'circuit' ? 'var(--ok)' : isCtl ? 'var(--paper-ink-muted)' : 'var(--bar)';
        const hollow = p.seed === 1, fill = hollow ? 'var(--paper)' : isCtl ? 'oklch(88% .01 70)' : col, sw = hollow ? 2.5 : isCtl ? 2 : 2, st2 = hollow || isCtl ? col : 'var(--paper)';
        const label = `${HEAD[p.id] || (p.family === 'original' ? 'Original' : p.id)}${p.seed === 1 ? ', seed 2' : ''}${p.qa ? ', 4-bit aware' : ''}: ${esc(D.benchmark.forget)} ${p1(p.forget)}%, ${esc(D.benchmark.retain)} ${p1(p.retain)}%, price ${(p.price * 100).toFixed(1)} points`;
        let shape;
        if (p.family === 'circuit') shape = `<rect x="${cx - 6}" y="${cy - 6}" width="12" height="12" rx="2" fill="${fill}" stroke="${st2}" stroke-width="${sw}"/>`;
        else if (p.family === 'original') shape = `<path d="M${cx} ${cy - 8}L${cx + 8} ${cy}L${cx} ${cy + 8}L${cx - 8} ${cy}Z" fill="${fill}" stroke="${st2}" stroke-width="${sw}"/>`;
        else if (p.family === 'deeper') shape = `<path d="M${cx} ${cy - 8}L${cx + 8} ${cy + 6}L${cx - 8} ${cy + 6}Z" fill="${fill}" stroke="${st2}" stroke-width="${sw}" stroke-linejoin="round"/>`;
        else if (p.family === 'shallower') shape = `<path d="M${cx - 8} ${cy - 6}L${cx + 8} ${cy - 6}L${cx} ${cy + 8}Z" fill="${fill}" stroke="${st2}" stroke-width="${sw}" stroke-linejoin="round"/>`;
        else shape = `<circle cx="${cx}" cy="${cy}" r="6.5" fill="${fill}" stroke="${st2}" stroke-width="${sw}"/>`;
        if (p.qa) shape += `<circle cx="${cx}" cy="${cy}" r="1.9" fill="${hollow ? col : 'var(--paper)'}"/>`;
        g += `<g class="pt" tabindex="0" role="img" data-i="${i}" aria-label="${esc(label)}"><circle class="hit" cx="${cx}" cy="${cy}" r="14"/>${shape}</g>`;
        const nm = p.family === 'original' ? 'Original' : (HEAD[p.id] || (p.seed === 1 ? 'seed 2' : ''));
        if (nm && !(W < 520 && (p.qa || nm === 'seed 2'))) lab.push({ x: cx, y: cy, t: nm, qa: p.qa, s1: p.seed === 1 && !HEAD[p.id] });
      });
      // direct labels with simple collision nudging
      const placed = [];
      lab.sort((a, b) => a.y - b.y).forEach((l) => {
        const w = l.t.length * 6.4, right = l.x + 10 + w > W - R;
        let x = right ? l.x - 10 : l.x + 10, y = l.qa ? l.y + 18 : l.y - 9, anc = right ? 'end' : 'start';
        for (let k = 0; k < 8 && placed.some((q) => Math.abs(q.y - y) < 13 && ((x < q.x + q.w && x + w > q.x) || (anc === 'end' && x - w < q.x + q.w && x > q.x))); k++) y += 13;
        placed.push({ x: anc === 'end' ? x - w : x, y, w });
        g += `<text class="dl" x="${x}" y="${y}" text-anchor="${anc}">${esc(l.t)}</text>`;
      });
      g += `<text class="bt" x="${L + 6}" y="${T + ih - 8}">better: up and to the left</text></svg><div class="tip" id="fr-tip" role="status" hidden></div>`;
      box.innerHTML = g;
      const tip = $('fr-tip'), svg = box.querySelector('svg');
      const show = (i) => {
        const p = pts[i]; if (!p) return;
        box.querySelectorAll('.pt').forEach((n) => n.classList.toggle('on', +n.dataset.i === i));
        const nm = HEAD[p.id] || (p.family === 'original' ? 'Original' : p.id);
        tip.innerHTML = `<b>${esc(nm)}</b><span>${p.family === 'original' ? 'no unlearning' : esc(famLabel(p.family)) + (p.seed === 1 ? ', seed 2' : '') + (p.qa ? ', 4-bit aware' : '')}</span><span>${esc(D.benchmark.forget)} ${p1(p.forget)}%</span><span>${esc(D.benchmark.retain)} ${p1(p.retain)}%</span><span>Price ${(p.price * 100).toFixed(1)} points</span>`;
        tip.hidden = false;
        const cx = X(p.forget), cy = Y(p.retain), flip = cx > W * 0.6;
        tip.style.left = (cx / W * 100) + '%'; tip.style.top = (cy / H * 100) + '%';
        tip.style.transform = `translate(${flip ? 'calc(-100% - 14px)' : '14px'}, ${cy > H * 0.6 ? 'calc(-100% - 8px)' : '8px'})`;
      };
      const hide = () => { tip.hidden = true; box.querySelectorAll('.pt').forEach((n) => n.classList.remove('on')); };
      svg.addEventListener('pointermove', (e) => {
        const r = svg.getBoundingClientRect(), k = W / r.width, mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k;
        let best = -1, bd = 24 * 24;
        pts.forEach((p, i) => { const dx = X(p.forget) - mx, dy = Y(p.retain) - my, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = i; } });
        if (best >= 0) show(best); else hide();
      });
      svg.addEventListener('pointerleave', hide);
      box.querySelectorAll('.pt').forEach((n) => { n.addEventListener('focus', () => show(+n.dataset.i)); n.addEventListener('blur', hide); n.addEventListener('keydown', (e) => { if (e.key === 'Escape') { hide(); n.blur(); } }); });
    }

    // legend: shape and fill carry identity, not colour alone; only families present at the selected format
    const ic = (svg) => `<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">${svg}</svg>`;
    const ICON = {
      default: '<circle cx="9" cy="9" r="6.5" fill="var(--hot)" stroke="var(--paper)" stroke-width="2"/>',
      circuit: '<rect x="3" y="3" width="12" height="12" rx="2" fill="var(--ok)" stroke="var(--paper)" stroke-width="2"/>',
      original: '<path d="M9 1L17 9L9 17L1 9Z" fill="var(--bar)"/>',
      deeper: '<path d="M9 2L16 15L2 15Z" fill="oklch(88% .01 70)" stroke="var(--paper-ink-muted)" stroke-width="2" stroke-linejoin="round"/>',
      shallower: '<path d="M2 3L16 3L9 16Z" fill="oklch(88% .01 70)" stroke="var(--paper-ink-muted)" stroke-width="2" stroke-linejoin="round"/>'
    };
    const drawKey = () => {
      const have = (f) => P.some((p) => p.prec === cur && p.family === f);
      const items = order.filter(have).map((f) => ic(ICON[f] || '<circle cx="9" cy="9" r="5" fill="oklch(88% .01 70)" stroke="var(--paper-ink-muted)" stroke-width="2"/>') + esc(famLabel(f)));
      items.push(ic(ICON.original) + 'Original');
      if (P.some((p) => p.prec === cur && p.seed === 1)) items.push(ic('<circle cx="9" cy="9" r="6" fill="var(--paper)" stroke="var(--ink)" stroke-width="2.5"/>') + 'Hollow: second seed');
      if (P.some((p) => p.prec === cur && p.qa)) items.push(ic('<circle cx="9" cy="9" r="6.5" fill="var(--ink)"/><circle cx="9" cy="9" r="2" fill="var(--paper)"/>') + 'Dot: trained through a 4-bit quantizer');
      $('fr-key').innerHTML = items.map((x) => `<li>${x}</li>`).join('');
    };
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (!tableOn) draw(); }, 150); });
    draw();
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
