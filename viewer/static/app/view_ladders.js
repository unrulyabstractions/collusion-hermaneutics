/* View: the ladders (#/ladders?task=<code>). The three ladders' rungs and rules are the framework's (served from
 * viewer/levels.py, rungs from hermeneutics/new_ladders.py); the evidence by task table is SGA's own
 * (scripts/paper/tables.py, served from viewer/ladders_data.py). */
import { h } from './dom.js';
import { glyph } from './levels.js';
import { fullLegend } from './case_view.js';

function byTask(title, label, notes, levels, d, hl, about) {
  return h('section', { class: 'card' }, h('h2', null, title), about || null,
    h('div', { class: 'table-wrap' }, h('table', { class: 'lv-ladder-table' },
      h('thead', null, h('tr', null, h('th', null, label),
        d.domains.map((dom, j) => h('th', { class: j === hl && 'lv-hl' }, dom, h('div', { class: 'small muted' }, notes[j]))))),
      h('tbody', null, levels.map(([level, gloss, cells]) => h('tr', null,
        h('td', null, h('b', null, level), h('div', { class: 'small muted' }, gloss)),
        cells.map((c, j) => h('td', { class: j === hl && 'lv-hl' }, c))))))));
}

export async function viewLadders(q) {
  const r = await fetch('/api/ladders');
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
  const task = q.get('task') || '';
  const hl = task in d.task_column ? d.task_column[task] : -1;
  const three = h('section', { class: 'card' }, h('h2', null, 'The three ladders'),
    h('p', { class: 'small muted' }, 'Each ladder is read per agent and per team by the framework\'s code, with no judge. The ladders are not nested: each is read on its own. The binary label of each ladder is its rung at or above the cut.'),
    d.order.map(k => {
      const L = d.ladders[k];
      return h('div', { class: 'lv-def' }, h('h3', null, L.name, ' ', h('span', { class: 'lv-proxy' }, `label at ${L.levels[L.cut]} or above`)),
        h('ol', { start: '0' }, L.levels.map((lv, i) => h('li', null, glyph(i, L.levels.length - 1, k), lv, i === L.cut ? h('b', null, '  (cut)') : null))),
        h('p', { class: 'small' }, L.rule));
    }));
  return h('div', { class: 'page wide' },
    h('div', { class: 'page-head' }, h('h1', null, 'Ladders'),
      hl >= 0 ? h('p', { class: 'muted' }, `The column of this run's task (${d.domains[hl]}) is highlighted.`) : null),
    three,
    h('section', { class: 'card' }, h('h2', null, 'Colour legend'), fullLegend(d.ladders, d.order)),
    byTask('The code\'s evidence by task', 'evidence', d.rung_notes, d.evidence, d, hl,
      h('p', { class: 'small lv-about' }, 'The names every run records in signals.jsonl (src/measures/rungs.py EVIDENCE). They were the old cheat ladder\'s rungs; the Cheating ladder\'s clauses now read them.')));
}
