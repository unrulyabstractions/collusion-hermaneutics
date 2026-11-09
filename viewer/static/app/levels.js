/* The six ladders: run panel, per-step strip and level glyph. Written for this viewer; levels come from
 * /api/run's "levels" (viewer/levels.py). Every string goes in through h() text nodes. */
import { h } from './dom.js';

export function glyph(index, of) {
  const box = h('span', { class: 'lv-glyph', 'aria-hidden': 'true' });
  for (let i = 1; i <= of; i++) box.appendChild(h('span', { class: ['lv-seg', index != null && i <= index && 'on'] }));
  return box;
}

function cellView(c, L) {
  if (!c) return h('td', { class: 'muted small' }, 'not computed');
  if (c.na) return h('td', { class: 'muted small', title: L.rule }, c.level);
  const tip = `${L.kind === 'proxy' ? 'Proxy' : 'Direct'} rule. ${L.rule}${c.evidence ? '\nEvidence: ' + c.evidence : ''}`;
  return h('td', { class: ['lv-cell', c.index > 0 && 'lv-hit'], title: tip },
    glyph(c.index, c.of), h('span', { class: 'lv-name' }, c.level),
    L.kind === 'proxy' ? h('span', { class: 'lv-proxy' }, 'proxy') : null);
}

/** Run panel: one row per agent, one team row, six columns. */
export function levelsPanel(lv, roles) {
  const head = h('tr', null, h('th', null, 'Agent'), lv.order.map(k => h('th', { title: lv.ladders[k].rule },
    lv.ladders[k].name, lv.ladders[k].kind === 'proxy' ? ' (proxy)' : '')));
  const rows = Object.entries(lv.agents).sort((a, b) => (roles[a[0]] === 'regulator') - (roles[b[0]] === 'regulator') || a[0].localeCompare(b[0]))
    .map(([a, cells]) => h('tr', null, h('td', { class: 'mono' }, a, roles[a] === 'regulator' ? ' (hidden regulator)' : ''),
      lv.order.map(k => cellView(cells[k], lv.ladders[k]))));
  if (!lv.single) rows.push(h('tr', { class: 'lv-team' }, h('td', null, h('b', null, 'Team')), lv.order.map(k => cellView(lv.team[k], lv.ladders[k]))));
  return h('section', { class: 'card', id: 'sec-six' },
    h('h2', null, 'Six ladders for this run'),
    h('p', { class: 'small muted' }, 'Each cell shows the level reached and a ladder with one bar per level. Hover a cell for the rule, whether it is direct or a proxy, and the evidence. ',
      h('a', { href: '#/ladders' }, 'All ladders and the level by task tables')),
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
        if (e.index > state[e.ladder]) { state[e.ladder] = e.index; fresh.push([e, k]); }
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
          h('span', { class: 'lv-chip-name' }, lad.name), na ? h('span', { class: 'muted' }, 'n/a') : glyph(state[L], lad.levels.length - 1),
          newL.has(L) ? h('span', { class: 'lv-star' }, 'new: ' + f.map(([e]) => e.level).join(', ')) : null);
      }),
      boardMarks && boardMarks.length ? h('span', { class: 'lv-board' }, boardMarks.join(' · ')) : null);
    return el;
  }
  const looseView = loose.length ? h('div', { class: 'lv-loose small' },
    h('b', null, 'Levels reached with no single call recorded (counted at the end of the run): '),
    loose.map((e, i) => [i ? '; ' : '', h('span', { title: e.evidence }, `${lv.ladders[e.ladder].name} ${e.level}`)])) : null;
  return { strip, looseView };
}
