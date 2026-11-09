/* One agent's whole episode in the case-study layout of arena-capstone script/visualize_case.html: a sticky left
 * column "The whole episode" (one row per tool call, top to bottom) and the transcript on the right, one block per
 * tool call. There is no reader pass for these runs, so the six ladder lanes stand where the sentence colours were.
 * Layout, classes and wording follow visualize_case.html and group_labels.js (RUNG_W); written for this viewer.
 * Every string goes in through h() text nodes. */
import { h } from './dom.js';
import { readableReasoning } from './transcript.js';

// Colour ramps per ladder, level 0 first. Cheating and collaboration are viz_common.js RUNG and COLLAB fills.
export const RAMP = {
  cheat: ['#eeede8', '#fde4c8', '#f9c396', '#f29a6a', '#e8714a', '#dc5b3c', '#d0452e', '#a8201c'],
  coord: ['#eeede8', '#f1eefa', '#e4dff7', '#d1c8f3', '#bdb2ee', '#8a76dc', '#4f3cb0'],
  coop: ['#eeede8', '#cfe6f0', '#7fb8d4', '#2b7d96'],
  collab: ['#eeede8', '#d3ecd9', '#7cc192', '#1f6b47'],
  collude: ['#eeede8', '#f4d6d0', '#eaa79a', '#d0452e', '#a8201c', '#6e0f0c'],
  scheme: ['#eeede8', '#c9a0d8', '#7a4fb0'],
};
const SHORT = { cheat: 'Ch', coord: 'Cd', coop: 'Cp', collab: 'Cb', collude: 'Cl', scheme: 'Sc' };
// Milestone wording, after group_labels.js RUNG_W ("saw it", "tried to reach it", ...).
const WORDS = {
  cheat: { exposed: 'saw it', tried: 'tried to reach it', opened: 'opened it', used: 'used it', tampered: 'tampered with the game', checked: 'checked it with the grader', shipped: 'shipped it' },
  coord: { 'read the board': 'read the board', posted: 'posted', discussed: 'discussed', solicited: 'solicited', crossed: 'piece crossed', assembled: 'assembled' },
};
const STAGE = ['not seen yet', 'saw it', 'tried to reach it', 'opened it', 'used it', 'tampered with the game', 'checked it with the grader', 'shipped it'];

/** Sentences of a text, each in a span.sn (so a later reader pass can colour them one by one). */
function sentences(text, call) {
  const box = h('div', { class: 'txt' });
  const parts = String(text || '').split(/(?<=[.!?])\s+(?=[A-Z0-9"'(`])|\n{2,}/);
  parts.forEach((p, i) => { if (!p) return; box.appendChild(h('span', { class: 'sn', dataset: { call: String(call) } }, p)); if (i < parts.length - 1) box.appendChild(document.createTextNode(' ')); });
  return box;
}
function asStr(v) {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  return JSON.stringify(v, null, 2);
}
function argText(a) {
  let x = a;
  if (typeof x === 'string') { try { x = JSON.parse(x); } catch (e) { return x; } }
  if (x && typeof x === 'object' && !Array.isArray(x)) {
    const keys = Object.keys(x);
    if (keys.length === 1) return asStr(x[keys[0]]);
    return keys.map(k => `${k}: ${asStr(x[k])}`).join('\n');
  }
  return asStr(x);
}
function resultText(r) {
  const c = r.content;
  if (Array.isArray(c)) return c.map(p => (p && p.text) || asStr(p)).join('\n');
  return asStr(c) || asStr(r.extra && r.extra.error);
}
function blk(label, text, call, input) {
  const el = h('div', { class: ['blk', input && 'inp', input && 'fold'] }, h('div', { class: 'lab' }, label), sentences(text, call));
  if (input) el.addEventListener('click', () => el.classList.remove('fold'));
  return el;
}

/**
 * ctx: {agent, levels (the run's six-ladder payload), boardCalls Map<tool_call_id, [mark]>, regulatorReads: Set of
 * tool_call_ids that read a regulator post, isRegulator, regulatorPostCalls: Set of tool_call_ids}.
 * Returns the same handle as renderTranscript: {el, findCall, findToolCall, showAll, nCalls}.
 */
export function renderCase(t, ctx) {
  const lv = ctx.levels;
  const steps = (t && t.trajectory && t.trajectory.steps) || [];
  // 1. tool calls in order, with the step's reasoning and text on its first call
  const calls = [];
  const loose = [];
  for (const s of steps) {
    const src = s.source || 'unknown';
    const rs = (s.observation && s.observation.results) || [];
    const reasoning = readableReasoning(s.reasoning_content);
    const msg = asStr(s.message);
    if (src === 'agent' && (s.tool_calls || []).length) {
      s.tool_calls.forEach((tc, j) => {
        let res = rs.filter(r => r.source_call_id === tc.tool_call_id);
        if (!res.length && rs[j] && !rs[j].source_call_id) res = [rs[j]];
        calls.push({ k: calls.length, tc, res, reasoning: j === 0 && reasoning && reasoning.text, text: j === 0 ? msg : '', step: s.step_id });
      });
    } else {
      loose.push({ before: calls.length, src, text: msg, reasoning: reasoning && reasoning.text, step: s.step_id });
    }
  }
  const n = calls.length;
  const idOf = new Map(calls.map(c => [c.tc.tool_call_id, c.k]));
  // 2. level per ladder per call (monotone), and milestones
  const order = lv ? lv.order : ['cheat'];
  const evs = (lv && lv.events && lv.events[ctx.agent]) || [];
  const at = new Map();
  const endEv = [];
  for (const e of evs) {
    const k = e.k != null ? e.k : e.call_id ? idOf.get(e.call_id) : null;
    if (k == null) { endEv.push(e); continue; }
    if (!at.has(k)) at.set(k, []);
    at.get(k).push(e);
  }
  const level = order.map(() => new Array(n).fill(0));
  const cur = Object.fromEntries(order.map(L => [L, 0]));
  const fresh = new Map();
  for (let k = 0; k < n; k++) {
    for (const e of at.get(k) || []) if (e.index > cur[e.ladder]) { cur[e.ladder] = e.index; if (!fresh.has(k)) fresh.set(k, []); fresh.get(k).push(e); }
    order.forEach((L, i) => { level[i][k] = cur[L]; });
  }
  const finalIx = Object.fromEntries(order.map(L => [L, (lv && lv.agents[ctx.agent] && lv.agents[ctx.agent][L] && lv.agents[ctx.agent][L].index) || 0]));
  const single = lv ? lv.single : true;
  const miles = [];
  for (const [k, es] of fresh) for (const e of es) {
    const w = (WORDS[e.ladder] || {})[e.level];
    if (w) miles.push({ k, t: `${w}, call ${k}`, ladder: e.ladder, tip: e.evidence });
  }
  if (ctx.regulatorCalls) {
    const ks = [...ctx.regulatorCalls].map(id => idOf.get(id)).filter(k => k != null).sort((a, b) => a - b);
    if (ks.length) miles.push({ k: ks[0], t: `regulator warned, call ${ks[0]}`, ladder: 'reg', tip: ctx.isRegulator ? 'this agent\'s first post as the hidden regulator' : 'the first call that read a hidden regulator\'s post' });
  }
  for (const e of endEv) {
    const w = (WORDS[e.ladder] || {})[e.level];
    if (w) miles.push({ k: n, t: `${w} (end of run)`, ladder: e.ladder, tip: e.evidence, end: true });
  }
  miles.sort((a, b) => a.k - b.k);

  // 3. right: one block per tool call
  const blocks = [];
  const right = h('div', { class: 'right' });
  const stripOf = k => h('span', { class: 'cs-strip' }, order.map((L, i) => {
    const lad = lv ? lv.ladders[L] : { name: 'Cheating', levels: STAGE };
    const na = single && L !== 'cheat';
    const v = level[i][k];
    const isNew = (fresh.get(k) || []).some(e => e.ladder === L);
    return h('span', { class: ['cs-chip', isNew && 'cs-new'], title: na ? `${lad.name}: n/a (one agent, no board)` : `${lad.name}: ${lad.levels[v]} (${v} of ${lad.levels.length - 1})${isNew ? '\nnew at this call: ' + (fresh.get(k) || []).filter(e => e.ladder === L).map(e => e.evidence).join('; ') : ''}` },
      h('i', { style: `background:${na ? 'transparent' : RAMP[L][v]}` }), na ? `${lad.name}: n/a` : [`${lad.name}: `, h('b', null, lad.levels[v]), ` (${v} of ${lad.levels.length - 1})`]);
  }));
  let li = 0;
  const looseBlock = x => h('div', { class: 'call msg' }, h('div', { class: 'ch' }, h('span', { class: 'cn' }, x.src === 'agent' ? 'Message' : x.src === 'user' ? 'User' : 'System'), x.step != null ? h('span', null, `step ${x.step}`) : null),
    x.reasoning ? blk('Reasoning', x.reasoning, -1, false) : null, x.text ? blk(x.src === 'agent' ? 'Text' : 'Prompt', x.text, -1, x.src !== 'agent') : null);
  for (const c of calls) {
    while (li < loose.length && loose[li].before <= c.k) right.appendChild(looseBlock(loose[li++]));
    const k = c.k;
    const marks = ctx.boardCalls.get(c.tc.tool_call_id) || [];
    const evNew = fresh.get(k) || [];
    const el = h('div', { class: 'call', id: `a-${ctx.agent}-call-${k}`, dataset: { call: String(k) } },
      h('div', { class: 'ch' }, h('span', { class: 'cn' }, `Call ${k}`), ' · ', h('span', { title: 'how far the agent had got toward the honeypot by this call' }, STAGE[level[0][k]] || ''),
        evNew.map(e => h('span', { class: 'ev', title: e.evidence }, (WORDS[e.ladder] || {})[e.level] || e.level)),
        h('span', { class: 'mz' }, c.tc.function_name || '')),
      stripOf(k),
      marks.length ? h('div', { class: 'cs-board' }, 'Board: ', marks.map((m, i) => [i ? ' · ' : '', h('span', { title: m.title || '' }, m.badge), m.text ? ` (${m.text})` : ''])) : null,
      c.reasoning ? blk('Reasoning', c.reasoning, k, false) : null,
      c.text && c.text.trim() ? blk('Text', c.text, k, false) : null,
      blk(`Tool call: ${c.tc.function_name || '?'}`, argText(c.tc.arguments), k, false),
      c.res.length ? c.res.map(r => blk('Tool result', resultText(r), k, true)) : null);
    blocks.push(el);
    right.appendChild(el);
  }
  while (li < loose.length) right.appendChild(looseBlock(loose[li++]));

  // 4. left: the whole episode, one row per call, six lanes, milestones in the margin
  const NS = 'http://www.w3.org/2000/svg';
  const LB = 180, LW = 14, GAP = 2, W = LB + order.length * (LW + GAP) + 10;
  const maxH = Math.max(320, Math.min(window.innerHeight - 220, 900));
  const rh = Math.max(2, Math.min(9, maxH / Math.max(n, 1)));
  const H = Math.ceil(n * rh) + 20 + 14;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('class', 'cs-svg');
  const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const [x, y] of Object.entries(a)) e.setAttribute(x, y); return e; };
  order.forEach((L, i) => {
    const x = LB + i * (LW + GAP);
    const lab = mk('text', { x: x + LW / 2, y: 10, 'text-anchor': 'middle', class: 'cs-lane-h' });
    lab.textContent = SHORT[L];
    const ttl = document.createElementNS(NS, 'title'); ttl.textContent = lv ? lv.ladders[L].name : L; lab.appendChild(ttl);
    svg.appendChild(lab);
    for (let k = 0; k < n; k++) {
      const na = single && L !== 'cheat';
      const r = mk('rect', { x, y: 14 + k * rh, width: LW, height: rh + 0.3, fill: na || !level[i][k] ? 'var(--line)' : RAMP[L][level[i][k]], class: 'cs-cell', 'data-call': k });
      svg.appendChild(r);
    }
    // end-of-run level (no call recorded) as a final band
    const v = finalIx[L];
    svg.appendChild(mk('rect', { x, y: 14 + n * rh + 4, width: LW, height: 8, fill: (single && L !== 'cheat') || !v ? 'var(--line)' : RAMP[L][v] }));
  });
  let last = -99;
  const placed = [];
  for (const m of miles) {
    const y0 = m.end ? 14 + n * rh + 8 : 14 + m.k * rh + rh / 2;
    const ty = Math.max(y0 + 3, last + 12);
    last = ty;
    placed.push(ty);
    const col = m.ladder === 'reg' ? 'var(--bad)' : m.ladder === 'cheat' ? 'var(--accent)' : 'var(--ink-2)';
    const tx = mk('text', { x: LB - 10, y: ty, 'text-anchor': 'end', class: 'cs-mile', fill: col, 'data-call': m.end ? '' : m.k });
    tx.textContent = m.t;
    const tt = document.createElementNS(NS, 'title'); tt.textContent = m.tip || m.t; tx.appendChild(tt);
    svg.appendChild(tx);
    svg.appendChild(mk('line', { x1: LB - 7, y1: ty - 3, x2: LB - 2, y2: y0, stroke: col, 'stroke-width': 1 }));
  }
  if (last + 8 > H) svg.setAttribute('height', Math.ceil(last + 8));
  for (let k = 0; k < n; k += 10) {
    const y = 14 + k * rh + 3;
    if (placed.every(py => Math.abs(py - y) > 12)) {
      const tx = mk('text', { x: LB - 10, y, 'text-anchor': 'end', class: 'cs-tick', 'data-call': k });
      tx.textContent = `call ${k}`;
      svg.appendChild(tx);
    }
  }
  let mark = null;
  function show(k) {
    const el = blocks[k];
    if (!el) return null;
    for (const b of right.querySelectorAll('.call.here')) b.classList.remove('here');
    el.classList.add('here');
    el.scrollIntoView({ block: 'start', behavior: 'smooth' });
    if (mark) mark.remove();
    mark = mk('rect', { x: LB - 1, y: 14 + k * rh - 1, width: order.length * (LW + GAP) + 1, height: Math.max(3, rh + 2), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2 });
    svg.appendChild(mark);
    return el;
  }
  svg.addEventListener('click', e => {
    const tgt = e.target.closest('[data-call]');
    let k = tgt && tgt.getAttribute('data-call') !== '' ? Number(tgt.getAttribute('data-call')) : null;
    if (k == null) { const r = svg.getBoundingClientRect(); k = Math.floor((e.clientY - r.top - 14) / rh); }
    if (k >= 0 && k < n) show(k);
  });
  const nSent = right.querySelectorAll('.sn').length;
  const left = h('div', { class: 'left' },
    h('h2', null, 'The whole episode ', h('span', { class: 'sub' }, `${n} calls, ${nSent} sentences`)),
    svg,
    h('div', { class: 'cs-legend' }, order.map(L => h('span', null, h('b', null, SHORT[L]), ' ', lv ? lv.ladders[L].name : L))),
    h('p', { class: 'note' }, 'No reader pass for this run; lanes show the six ladders. One row per tool call, call 0 at the top; the band below the last call is the level at the end of the run. Click a row or a label to open that call.'));
  const el = h('div', { class: 'case' }, h('div', { class: 'main' }, left, right));
  return {
    el, nCalls: n, showAll() {},
    findCall: k => show(k),
    findToolCall: id => (idOf.has(id) ? show(idOf.get(id)) : null),
  };
}
