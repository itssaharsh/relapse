/* Relapse, Stage 2. Every number on the page is read from data/stage2.json (written by scripts/build_space.py); none is typed here. */
(function () {
  'use strict';
  var MODELS = ['global', 'fire', 'earth', 'water', 'base'];
  var SCALE = 70;            // bars run from 0% to 70%, above every score on the page
  var FT_LOW = 25;           // the fine-tuning chart starts at guessing level
  var WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
  var D = null, current = 'global';

  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  var cap = function (s) { return s.charAt(0).toUpperCase() + s.slice(1); };
  var name = function (v) { return 'Tiny Aya ' + cap(v); };
  var f1 = function (x) { return (Math.abs(x) < 0.05 ? 0 : x).toFixed(1); };
  var pct = function (x) { return f1(x) + '%'; };
  var sgn = function (x) { var r = Math.round(x * 10) / 10; return (r > 0 ? '+' : r < 0 ? '−' : '±') + Math.abs(r).toFixed(1); };
  var pos = function (x) { return Math.max(0, Math.min(100, (x / SCALE) * 100)); };
  var tag = function (back) { return back ? '<span class="tag back">Came back</span>' : '<span class="tag held">Held</span>'; };

  var FORMATS = {
    pct: pct, pt: f1, sgn: sgn,
    pt0: function (x) { return x.toFixed(0); },
    words: function (x) { return WORDS[x] || String(x); },
    Words: function (x) { return cap(WORDS[x] || String(x)); }
  };

  function get(path) {
    return path.split('.').reduce(function (o, k) { return o == null ? undefined : o[k]; }, D);
  }

  /* <span data-t="path|format"> placeholders in the HTML */
  function fill() {
    document.querySelectorAll('[data-t]').forEach(function (el) {
      var parts = el.getAttribute('data-t').split('|'), v = get(parts[0]);
      if (v === undefined) { console.warn('missing value', parts[0]); return; }
      el.textContent = parts[1] ? FORMATS[parts[1]](v) : String(v);
    });
  }

  function dots() {
    ['default', 'released'].forEach(function (k) {
      var b = D.brief[k], box = document.querySelector('[data-dots="' + k + '"]'), html = '';
      for (var i = 0; i < b.tok_runs; i++) html += '<i class="' + (i < b.tok_relapses ? 'on' : 'off') + '"></i>';
      box.innerHTML = html;
    });
  }

  /* check a released model */
  function picker() {
    var box = $('picker');
    MODELS.forEach(function (v) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = cap(v); b.dataset.v = v;
      b.setAttribute('aria-pressed', String(v === current));
      b.setAttribute('aria-label', name(v));
      b.addEventListener('click', function () { show(v); });
      box.appendChild(b);
    });
  }

  function hazardRow(o) {
    var refs = '<i class="ref" style="left:' + pos(25) + '%" title="25%: guessing"></i>' +
               (o.orig != null ? '<i class="ref orig" style="left:' + pos(o.orig) + '%" title="the original model"></i>' : '');
    return '<li class="row ' + o.cls + '"><div class="r-label"><b>' + o.label + '</b>' + (o.sub ? '<span>' + o.sub + '</span>' : '') + '</div>' +
      '<div class="r-bar" role="img" aria-label="' + esc(o.label + ': ' + pct(o.v)) + '"><i class="fill" style="width:' + pos(o.v) + '%"></i>' + refs + '</div>' +
      '<div class="r-val">' + pct(o.v) + '</div><div class="r-state">' + o.state + '</div></li>';
  }

  function show(v) {
    current = v;
    var m = D.models[v];
    document.querySelectorAll('#picker button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.v === v)); });
    $('m-name').textContent = name(v) + ', unlearned';
    $('m-sub').textContent = 'Edited at layers ' + m.layers + ', seed ' + m.seed + ', the copy we publish.';
    var repo = $('m-repo'); repo.href = m.repo; repo.textContent = 'Model files on Hugging Face';
    repo.setAttribute('aria-label', name(v) + ' unlearned model files on Hugging Face');

    var deployed = [['file_q4_0', 'in the real Q4_0 file'], ['file_q4_k_m', 'in the real Q4_K_M file'], ['nf4', 'in NF4'], ['outdated_tokenizer', 'through the outdated tokenizer setting']];
    var cb = deployed.filter(function (d) { return m[d[0]].came_back; });
    var held = deployed.filter(function (d) { return !m[d[0]].came_back; });
    var list = function (xs) { xs = xs.map(function (d) { return d[1]; }); return xs.length > 1 ? xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1] : xs[0]; };
    var s = 'Unlearning took its hazard score ' + f1(m.forgot) + ' points below the original. ';
    s += cb.length ? 'The knowledge came back ' + list(cb) + (held.length ? ', and the forgetting held ' + list(held) + '. ' : '. ')
                   : 'The forgetting held ' + list(held) + '. ';
    s += m.five_examples.came_back ? 'Five worked examples in the prompt brought it back. ' : 'Five worked examples in the prompt did not bring it back. ';
    s += m.fine_tuning.came_back ? 'A short fine-tune did: ' + sgn(m.fine_tuning.change) + ' points.' : 'A short fine-tune did not either (' + sgn(m.fine_tuning.change) + ' points).';
    $('m-summary').textContent = s;

    var vs = function (o, what) { return tag(o.came_back) + '<span>' + sgn(o.change) + ' points ' + what + '</span>'; };
    var rows = [
      { cls: 'base', label: 'Original model', sub: 'Before unlearning', v: m.original, state: '<span class="tag plain">Before</span>' },
      { cls: 'held', label: 'Unlearned, full precision', sub: 'The weights as trained', v: m.unlearned, orig: m.original, state: '<span class="tag held">Unlearned</span><span>' + f1(m.forgot) + ' points lower</span>' },
      { cls: m.file_q4_0.came_back ? 'back' : 'held', label: 'Real Q4_0 file', sub: '4-bit file for llama.cpp', v: m.file_q4_0.score, orig: m.original, state: vs(m.file_q4_0, 'vs full precision') },
      { cls: m.file_q4_k_m.came_back ? 'back' : 'held', label: 'Real Q4_K_M file', sub: '4-bit file for llama.cpp', v: m.file_q4_k_m.score, orig: m.original, state: vs(m.file_q4_k_m, 'vs full precision') },
      { cls: m.nf4.came_back ? 'back' : 'held', label: 'NF4', sub: '4-bit, as bitsandbytes stores it', v: m.nf4.score, orig: m.original, state: vs(m.nf4, 'vs full precision') },
      { cls: m.outdated_tokenizer.came_back ? 'back' : 'held', label: 'Through the outdated tokenizer', sub: 'The older tokenizer setting', v: m.outdated_tokenizer.score, orig: m.original, state: vs(m.outdated_tokenizer, 'vs the official one') },
      { cls: m.five_examples.came_back ? 'back' : 'held', label: 'With five worked examples in the prompt', sub: 'Asked plainly, the same questions score ' + pct(m.five_examples.plain), v: m.five_examples.score, orig: m.original, state: vs(m.five_examples, 'vs asked plainly') },
      { cls: m.fine_tuning.came_back ? 'back' : 'held', label: 'After a short fine-tuning attack', sub: '25 steps on 64 hazard questions; ' + pct(m.fine_tuning.before) + ' before, on the same questions', v: m.fine_tuning.score, orig: m.original, state: vs(m.fine_tuning, 'vs before') }
    ];
    $('m-rows').innerHTML = rows.map(hazardRow).join('');

    $('m-orig-mark').textContent = pct(m.original);

    var keep = function (label, sub, o) {
      return '<li class="row keep-row"><div class="r-label"><b>' + label + '</b><span>' + sub + '</span></div>' +
        '<div class="r-bar" role="img" aria-label="' + esc(label + ': ' + pct(o.unlearned) + ', original ' + pct(o.original)) + '"><i class="fill" style="width:' + pos(o.unlearned) + '%"></i><i class="ref orig" style="left:' + pos(o.original) + '%" title="the original model"></i></div>' +
        '<div class="r-val">' + pct(o.unlearned) + '</div><div class="r-state"><span>from ' + pct(o.original) + ', ' + f1(o.lost) + ' points lost</span></div></li>';
    };
    $('m-keep').innerHTML = keep('General knowledge (MMLU)', 'Broad multiple-choice test', m.mmlu) + keep('Benign biology', 'Ordinary biology questions from MMLU', m.benign);
  }

  /* when one file relapses */
  function relapses() {
    var R = D.relapses, cols = [['q4_0', 'Q4_0, exact rounding'], ['nf4', 'NF4'], ['file_q4_0', 'Real Q4_0 file'], ['file_q4_k_m', 'Real Q4_K_M file']];
    $('rl-body').innerHTML = R.cases.map(function (c) {
      var cells = cols.map(function (k) {
        var x = c.cells[k[0]];
        return '<td data-label="' + k[1] + '"' + (x.came_back ? ' class="back"' : '') + '><span class="v">' + pct(x.score) + '</span><span class="d">' + sgn(x.change) + ' points</span>' + tag(x.came_back) + '</td>';
      }).join('');
      return '<tr><th scope="row">' + name(c.model) + '<span>layers ' + c.layers + ', seed ' + c.seed + '</span></th><td class="fp" data-label="Full precision"><span class="v">' + pct(c.fp16) + '</span></td>' + cells + '</tr>';
    }).join('');
    $('rl-held-note').textContent = (R.another_format_held ? 'In every run at least one 4-bit format of the same weights held' : 'In some runs every 4-bit format relapsed') +
      (R.any_q4_k_m_file_relapsed ? ', though a real Q4_K_M file did relapse' : ', and no real Q4_K_M file relapsed');
  }

  /* fine-tuning attack, every final run */
  function finetune() {
    var x = function (v) { return Math.max(0, Math.min(100, ((v - FT_LOW) / (SCALE - FT_LOW)) * 100)); };
    var rows = D.costs.fine_tuning.map(function (r) {
      var a = Math.min(r.before, r.after), b = Math.max(r.before, r.after);
      return '<li class="' + (r.came_back ? 'back' : 'held') + '"><div class="f-label">' + cap(r.model) + ' <span>seed ' + r.seed + '</span></div>' +
        '<div class="f-track" role="img" aria-label="' + esc(name(r.model) + ', seed ' + r.seed + ': ' + pct(r.before) + ' before, ' + pct(r.after) + ' after, original ' + pct(r.original)) + '">' +
        '<i class="f-orig" style="left:' + x(r.original) + '%"></i><i class="f-line" style="left:' + x(a) + '%;width:' + (x(b) - x(a)) + '%"></i>' +
        '<i class="f-dot before" style="left:' + x(r.before) + '%"></i><i class="f-dot after" style="left:' + x(r.after) + '%"></i></div>' +
        '<div class="f-state">' + tag(r.came_back) + '<span>' + pct(r.before) + ' → ' + pct(r.after) + '</span></div></li>';
    });
    rows.push('<li class="axis" aria-hidden="true"><div class="f-label"></div><div class="f-track f-axis"><span style="left:0">' + FT_LOW + '% guessing</span><span style="right:0">' + SCALE + '%</span></div><div class="f-state"></div></li>');
    $('ft').innerHTML = rows.join('');
  }

  function init(data) {
    D = data;
    fill(); dots(); picker(); show(current); relapses(); finetune();
  }

  fetch('data/stage2.json', { cache: 'no-cache' })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(init)
    .catch(function (e) {
      console.error(e);
      var p = document.createElement('p'); p.className = 'err';
      p.textContent = 'The results could not be loaded. Reload the page, or read them in the README on GitHub.';
      $('brief').insertBefore(p, $('brief').children[1]);
    });
})();
