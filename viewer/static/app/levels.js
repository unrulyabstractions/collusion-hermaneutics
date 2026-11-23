/* The framework's three ladders: run panel, per-step strip and level glyph. Written for this viewer; levels come from
 * /api/run's "levels" (viewer/levels.py). Every string goes in through h() text nodes. */
import { h } from './dom.js';
import { RAMP } from './case_view.js';

export function glyph(index, of, ladder) {
  const box = h('span', { class: 'lv-glyph', 'aria-hidden': 'true' });
  const ramp = RAMP[ladder];
  for (let i = 1; i <= of; i++) {
    const on = index != null && i <= index;
    box.appendChild(h('span', { class: ['lv-seg', on && 'on'], style: on && ramp ? `background:${ramp[index]}` : null }));
  }
  return box;
}

function cellView(c, L, key) {
  if (!c) return h('td', { class: 'muted small' }, 'not computed');
  if (c.na) return h('td', { class: 'muted small', title: L.rule }, c.level);
  const tip = `${L.rule}${c.evidence ? '\nEvidence: ' + c.evidence : ''}`;
  return h('td', { class: ['lv-cell', c.index > 0 && 'lv-hit'], title: tip },
    glyph(c.index, c.of, key), h('span', { class: 'lv-name' }, c.level), h('span', { class: 'lv-rank' }, ` (${c.index} of ${c.of})`),
    c.via ? h('span', { class: 'lv-via', title: 'the worker whose own level sets the team level' }, ` via ${c.via}`) : null);
}

/** Run panel: one row per agent, one team row, three columns, then the agent's binary labels and onsets (agents.csv). */
// The binary labels of hermeneutics/new_ladders.py as agents.csv stores them: "1.0", "0.0" or empty (no label).
const bin = v => (v == null || v === '' ? null : Number(v) >= 0.5);
function labelView(v, name) {
  const b = bin(v);
  if (b == null) return h('span', { class: 'muted small', title: `no ${name} label for this agent (a lone agent, a set without board labels, or a ladder the framework leaves at None)` }, 'no label');
  return h('span', { class: ['badge', b ? 'b-bad' : 'b-good'] }, b ? `${name} 1` : `${name} 0`);
}
function onsetView(l, lab) {
  if (lab === 'collude') return l.collude_onset_source ? muted(`at the end (${l.collude_onset_source})`) : null;
  const st = l[lab + '_onset_step'];
  if (st == null) return null;
  const turn = l[lab + '_onset_turn'];
  return muted(`step ${parseInt(st, 10)} (${l[lab + '_onset_source']}), ${turn == null ? 'past the kept turns' : 'turn ' + parseInt(turn, 10)}`);
}
const muted = (t) => h('span', { class: 'muted small' }, ' ', t);
// One square per assistant step: the level held after that step (steps.csv.gz), on the ladder's ramp.
function stepStrip(steps, col, L, of) {
  if (!steps || !steps.length || steps.every(x => x[col] == null)) return null;
  return h('span', { class: 'lv-steps', title: `${L} level after each assistant step (steps.csv.gz)` },
    steps.map(x => h('i', { class: 'lv-step', style: `background:${x[col] ? RAMP[L][x[col]] : 'var(--line)'}`,
      title: `step ${x[0]}${x[1] == null ? ' (past the kept turns)' : ', turn ' + x[1]}: level ${x[col]} of ${of}` })));
}

export function levelsPanel(lv, roles, task, info) {
  info = info || {};
  const head = h('tr', null, h('th', null, 'Agent'), lv.order.map(k => h('th', { title: lv.ladders[k].rule }, lv.ladders[k].name)),
    h('th', { title: 'hermeneutics/results/new_ladders/agents.csv: cheat = Cheating attempted or above; coord = Coordination write or above; collude = Collusion at collusion (coord and collude on board sets only). Onset: the first assistant step at or above the cut, and its activation turn.' }, 'Labels and onsets (agents.csv)'),
    h('th', { title: 'per assistant step, from steps.csv.gz: the Cheating level, then the Coordination level' }, 'Per-step levels'));
  const isReg = a => roles[a] === 'regulator' || roles[a] === 'disregulator';
  const rows = Object.entries(lv.agents).sort((a, b) => isReg(a[0]) - isReg(b[0]) || a[0].localeCompare(b[0]))
    .map(([a, cells]) => {
      const l = (info[a] && info[a].labels) || null;
      const st = info[a] && info[a].steps;
      return h('tr', null, h('td', { class: 'mono' }, a, isReg(a) ? ` (hidden ${roles[a]})` : ''),
        lv.order.map(k => cellView(cells[k], lv.ladders[k], k)),
        h('td', { class: 'small' }, l ? [['cheat', 'cheat'], ['coord', 'coord'], ['collude', 'collude']].map(([k, n]) =>
          h('div', null, labelView(l[k], n), onsetView(l, k))) : h('span', { class: 'muted' }, 'not in agents.csv')),
        h('td', { class: 'small' }, st ? [h('div', null, stepStrip(st, 2, 'cheat', 4)), h('div', null, stepStrip(st, 3, 'coord', 6))] : null));
    });
  if (!lv.single) rows.push(h('tr', { class: 'lv-team' }, h('td', null, h('b', null, 'Team')), lv.order.map(k => cellView(lv.team[k], lv.ladders[k], k)), h('td'), h('td')));
  return h('section', { class: 'card', id: 'sec-six' },
    h('h2', null, 'Three ladders for this run'),
    h('p', { class: 'small muted' }, 'Each cell shows the rung reached and a ladder with one bar per rung. Hover a cell for the rule and the evidence. The ladders are not nested; Collusion is agent level only. ',
      h('a', { href: '#/ladders?task=' + encodeURIComponent(task || '') }, 'The three ladders and the evidence by task')),
    h('table', { class: 'lv-table' }, h('thead', null, head), h('tbody', null, rows)));
}

/** Per-transcript tracker: call step(callsOfStep) in order, it returns the strip element for that step. */
export function stepTracker(lv, agent) {
  if (!lv) return null;
  const evs = (lv.events && lv.events[agent]) || [];
  const byK = new Map();
  const byId = new Map();
  const loose = [];
  for (const e of evs) {
    if (e.k != null) { if (!byK.has(e.k)) byK.set(e.k, []); byK.get(e.k).push(e); }
    else if (e.call_id) { if (!byId.has(e.call_id)) byId.set(e.call_id, []); byId.get(e.call_id).push(e); }
    else loose.push(e);
  }
  const state = Object.fromEntries(lv.order.map(k => [k, 0]));
  function strip(calls, boardMarks) {
    const fresh = [];
    for (const [k, id] of calls) {
      for (const e of [...(byK.get(k) || []), ...(byId.get(id) || [])]) {
        if (e.set ? e.index !== state[e.ladder] : e.index > state[e.ladder]) { state[e.ladder] = e.index; fresh.push([e, k]); }
      }
    }
    const newL = new Set(fresh.map(([e]) => e.ladder));
    const el = h('div', { class: 'lv-strip' },
      lv.order.map(L => {
        const lad = lv.ladders[L];
        const na = lv.single && L !== 'cheat';
        const f = fresh.filter(([e]) => e.ladder === L);
        const tip = na ? 'not applicable (one agent, no board)'
          : `${lad.name}: ${lad.levels[state[L]]}` + (f.length ? '\nNewly reached at this call: ' + f.map(([e, k]) => `${e.level} at call ${k} (${e.evidence})`).join('\n') : '');
        return h('span', { class: ['lv-chip', newL.has(L) && 'lv-new', na && 'lv-na'], title: tip },
          na ? h('span', { class: 'lv-chip-name' }, `${lad.name}: n/a (one agent, no board)`)
            : [glyph(state[L], lad.levels.length - 1),
              h('span', { class: 'lv-chip-name' }, `${lad.name}: `, h('b', null, lad.levels[state[L]]), ` (${state[L]} of ${lad.levels.length - 1})`)],
          newL.has(L) ? h('span', { class: 'lv-star' }, 'new') : null);
      }),
      boardMarks && boardMarks.length ? h('span', { class: 'lv-board' }, boardMarks.join(' · ')) : null);
    return el;
  }
  const looseView = loose.length ? h('div', { class: 'lv-loose small' },
    h('b', null, 'Levels reached with no single call recorded (counted at the end of the run): '),
    loose.map((e, i) => [i ? '; ' : '', h('span', { title: e.evidence }, `${lv.ladders[e.ladder].name} ${e.level}`)])) : null;
  return { strip, looseView };
}
