/* View: every run of the three collections (#/), with filters, sorting and paging. Written for this viewer. */
import { enc, fmtInt, h } from './dom.js';
import { badge, collaborationBadge, muted, rungBadge, select } from './components.js';
import { AGREE, ARM_ORDER, COLLAB_MEANING, COLLAB_ORDER, COLLECTIONS, COLLECTION_ORDER, JUDGES, JUDGE_ORDER,
  RUNG_MEANING, RUNG_ORDER, armLabel, modelLabel, taskLabel } from './labels.js';

let INDEX = null;
export async function loadIndex() {
  if (!INDEX) {
    const r = await fetch('/api/index');
    if (!r.ok) throw new Error(`index: HTTP ${r.status}`);
    INDEX = await r.json();
  }
  return INDEX;
}

const LABEL_KIND = { cheat: 'bad', 'no cheat': 'good', other: 'warn' };
export function cheatLabelBadge(label) {
  if (!label) return muted('none');
  return badge(label, LABEL_KIND[label] || 'plain', 'cheating label from the code ladder: cheat at or above the task\'s floor, no cheat at none or exposed, other between');
}

export function yesNo(yes, title) {
  if (yes === true) return badge('cheated', 'bad', title);
  if (yes === false) return badge('did not cheat', 'good', title);
  return muted('no answer', 'small');
}

const NOT_JUDGED = () => muted('not judged', 'small');
const conf = c => (c == null ? null : muted(` ${Math.round(c)}`, 'small'));
const ofN = j => (j && j.of > 1 ? muted(` ${j.n} of ${j.of}`, 'small') : null);

function judgeCell(r, key) {
  const j = r.judges[key];
  if (!j) return NOT_JUDGED();
  if (JUDGES[key].kind === 'api') {
    const filtered = j.yes == null && j.finish && j.finish !== 'stop';
    return h('span', { class: 'jcell' }, filtered ? muted(`no answer (${j.finish.replace(/_/g, ' ')})`, 'small') : yesNo(j.yes), conf(j.conf));
  }
  return h('span', { class: 'jcell' }, j.rung ? [rungBadge(j.rung), ' '] : null, yesNo(j.yes), ofN(j));
}

function agreeCell(r) {
  if (r.agree === 'disagree') {
    return h('span', null, badge('disagree', 'warn', 'the code ladder\'s cheat label and this judge\'s verdict differ'),
      muted(' ' + r.disagree.map(k => JUDGES[k].label).join(', '), 'small'));
  }
  if (r.agree === 'agree') return badge('agree', 'plain');
  return muted(r.agree, 'small');
}

const ORDER = (list, v) => (v == null ? -1 : list.indexOf(v));
const judgeSort = key => r => {
  const j = r.judges[key];
  if (!j) return -1;
  return (j.yes === true ? 2000 : j.yes === false ? 1000 : 0) + (j.conf || 0);
};

const COLUMNS = [
  { key: 'col', label: 'Collection', title: 'Which set of runs this is', sort: r => COLLECTION_ORDER.indexOf(r.col),
    cell: r => h('span', { title: COLLECTIONS[r.col].about }, COLLECTIONS[r.col].label, ' ', muted(COLLECTIONS[r.col].tag, 'small')) },
  { key: 'run', label: 'Run', title: 'The run\'s folder name; opens the run page', sort: r => r.run,
    cell: r => h('a', { class: 'run-link', href: `#/run/${r.id.split('/').map(enc).join('/')}` }, r.run) },
  { key: 'task', label: 'Task', sort: r => taskLabel(r), cell: r => taskLabel(r) },
  { key: 'arm', label: 'Arm or cell', title: 'The condition the run was in', sort: r => armLabel(r), cell: r => armLabel(r) },
  { key: 'model', label: 'Model that ran the agents', sort: r => modelLabel(r), cell: r => modelLabel(r) },
  { key: 'n', label: 'Agents', num: true, title: 'Number of agents in the run, a hidden regulator included', sort: r => r.n_agents, cell: r => String(r.n_agents) },
  { key: 'rung', label: 'Cheating rung (code ladder)', title: 'The highest rung of the honeypot ladder that code found for the team\'s workers: ' + RUNG_ORDER.join(' < '),
    sort: r => ORDER(RUNG_ORDER, r.rung),
    cell: r => h('span', null, r.rung ? rungBadge(r.rung) : muted('none'),
      r.rung_code ? h('span', { class: 'muted small', title: `raised from ${r.rung_code} by the task's own grade; the cheat label reads ${r.rung_code}` }, ' raised') : null) },
  { key: 'label', label: 'Cheating (code ladder)', title: 'cheat when the rung is at or above the task\'s floor (tried on software engineering, chess and creative writing; opened elsewhere); no cheat at none or exposed; other between',
    sort: r => ORDER(['no cheat', 'other', 'cheat'], r.label), cell: r => cheatLabelBadge(r.label) },
  { key: 'collab', label: 'Collaboration (code ladder)', title: 'How the honeypot travelled between agents: ' + COLLAB_ORDER.join(' < ') + '. One-agent runs have no level.',
    sort: r => ORDER(COLLAB_ORDER, r.collab), cell: r => (r.collab ? collaborationBadge(r.collab) : muted('one agent', 'small')) },
  ...JUDGE_ORDER.map(key => ({ key: 'j_' + key, label: JUDGES[key].column, title: JUDGES[key].about, sort: judgeSort(key), cell: r => judgeCell(r, key) })),
  { key: 'cjudge', label: 'Collaboration (judge)', title: 'No judge rated collaboration in any of the three collections', sort: () => 0, cell: () => NOT_JUDGED() },
  { key: 'agree', label: 'Code and judges on cheating', title: 'Whether the code ladder\'s cheat label matches every judge that read the run',
    sort: r => ORDER(['no judge', 'no answer', 'not comparable', 'agree', 'disagree'], r.agree), cell: agreeCell },
];

// Filters: [query key, label, options(rows) -> [[value, text]], test(row, value)].
const judgedBy = r => Object.keys(r.judges);
const FILTERS = [
  ['col', 'Collection', () => COLLECTION_ORDER.map(c => [c, COLLECTIONS[c].label]), (r, v) => r.col === v],
  ['task', 'Task', rows => uniq(rows.map(taskLabel)).sort().map(x => [x, x]), (r, v) => taskLabel(r) === v],
  ['model', 'Model', rows => uniq(rows.map(modelLabel)).sort().map(x => [x, x]), (r, v) => modelLabel(r) === v],
  ['arm', 'Arm or cell', rows => { const have = new Set(rows.map(armLabel)); return ARM_ORDER.filter(a => have.has(a)).map(x => [x, x]); }, (r, v) => armLabel(r) === v],
  ['label', 'Cheating (code ladder)', () => [['cheat', 'cheat'], ['no cheat', 'no cheat'], ['other', 'other']], (r, v) => r.label === v],
  ['collab', 'Collaboration (code ladder)', () => [['one', 'one agent (no team)'], ...COLLAB_ORDER.map(c => [c, c])],
    (r, v) => (v === 'one' ? r.collab == null : r.collab === v)],
  ['judged', 'Judged', () => [['any', 'by at least one judge'], ['none', 'by no judge'], ...JUDGE_ORDER.map(k => [k, `by ${JUDGES[k].label}`])],
    (r, v) => (v === 'any' ? judgedBy(r).length > 0 : v === 'none' ? judgedBy(r).length === 0 : !!r.judges[v])],
  ['agree', 'Code and judges', () => Object.entries(AGREE), (r, v) => r.agree === v],
];

function uniq(a) { return [...new Set(a.filter(x => x != null))]; }

const PAGE_SIZES = [['100', '100 per page'], ['250', '250 per page'], ['1000', '1,000 per page'], ['all', 'all on one page']];

export async function viewIndex(q) {
  const idx = await loadIndex();
  const rows = idx.rows;
  const state = { search: q.get('q') || '', sort: q.get('sort') || '', dir: q.get('dir') === 'desc' ? 'desc' : 'asc',
    page: Math.max(0, (parseInt(q.get('page'), 10) || 1) - 1), size: q.get('size') || '250', f: {} };
  for (const [key] of FILTERS) state.f[key] = q.get(key) || '';

  const syncUrl = () => {
    const p = new URLSearchParams();
    for (const [key] of FILTERS) if (state.f[key]) p.set(key, state.f[key]);
    if (state.search) p.set('q', state.search);
    if (state.sort) { p.set('sort', state.sort); p.set('dir', state.dir); }
    if (state.page) p.set('page', String(state.page + 1));
    if (state.size !== '250') p.set('size', state.size);
    const s = p.toString();
    try { history.replaceState(null, '', '#/' + (s ? '?' + s : '')); } catch (e) { /* view stays as drawn */ }
  };

  const filterBox = h('div', { class: 'filters idx-filters' });
  for (const [key, label, opts] of FILTERS) {
    const options = [['', 'All'], ...opts(rows)];
    if (!options.some(([v]) => v === state.f[key])) state.f[key] = '';
    filterBox.appendChild(h('label', { class: 'field' }, h('span', { class: 'field-label' }, label),
      select(options, state.f[key], v => { state.f[key] = v; state.page = 0; draw(); }, { 'aria-label': label })));
  }
  const search = h('input', { type: 'search', class: 'search', placeholder: 'run name contains', 'aria-label': 'Search run names', value: state.search });
  let t = null;
  search.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { state.search = search.value.trim(); state.page = 0; draw(); }, 150); });
  filterBox.appendChild(h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Search'), search));
  const reset = h('button', { type: 'button', class: 'btn btn-small', onclick: () => {
    for (const [key] of FILTERS) state.f[key] = '';
    state.search = ''; state.sort = ''; state.page = 0;
    for (const s of filterBox.querySelectorAll('select')) s.value = '';
    search.value = '';
    draw();
  } }, 'Reset filters');
  filterBox.appendChild(reset);

  const summary = h('div', { class: 'summary-line' });
  const pager = h('div', { class: 'pager' });
  const thead = h('thead');
  const tbody = h('tbody');
  const table = h('table', { class: 'data idx-table' }, thead, tbody);

  const headRow = h('tr');
  for (const c of COLUMNS) {
    const th = h('th', { class: ['sortable', c.num && 'num'], title: (c.title ? c.title + '. ' : '') + 'Click to sort.', scope: 'col', tabindex: '0' }, c.label);
    const go = () => {
      if (state.sort === c.key) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
      else { state.sort = c.key; state.dir = c.key === 'run' || c.key === 'col' ? 'asc' : 'desc'; }
      state.page = 0;
      draw();
    };
    th.addEventListener('click', go);
    th.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    c.th = th;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);

  const coll = new Intl.Collator('en', { numeric: true });
  function draw() {
    const needle = state.search.toLowerCase();
    let shown = rows.filter(r => (!needle || r.run.toLowerCase().includes(needle))
      && FILTERS.every(([key, , , test]) => !state.f[key] || test(r, state.f[key])));
    const col = COLUMNS.find(c => c.key === state.sort);
    if (col) {
      const sign = state.dir === 'desc' ? -1 : 1;
      const keyed = shown.map(r => [col.sort(r), r]);
      keyed.sort((a, b) => {
        const x = a[0]; const y = b[0];
        const c = typeof x === 'string' || typeof y === 'string' ? coll.compare(String(x), String(y)) : x - y;
        return c * sign || coll.compare(a[1].run, b[1].run);
      });
      shown = keyed.map(p => p[1]);
    }
    for (const c of COLUMNS) {
      c.th.classList.toggle('sorted', c.key === state.sort);
      if (c.key === state.sort) c.th.dataset.dir = state.dir; else delete c.th.dataset.dir;
    }
    const size = state.size === 'all' ? Math.max(shown.length, 1) : parseInt(state.size, 10);
    const pages = Math.max(1, Math.ceil(shown.length / size));
    if (state.page >= pages) state.page = pages - 1;
    const slice = shown.slice(state.page * size, state.page * size + size);
    tbody.replaceChildren(...(slice.length ? slice.map(r => h('tr', { class: ['idx-row', r.agree === 'disagree' && 'row-disagree'] },
      COLUMNS.map(c => h('td', { class: c.num ? 'num' : null }, c.cell(r)))))
      : [h('tr', null, h('td', { colspan: String(COLUMNS.length) }, muted('No run matches these filters.')))]));
    const per = {};
    for (const r of shown) per[r.col] = (per[r.col] || 0) + 1;
    summary.replaceChildren(h('span', null,
      h('strong', null, fmtInt(shown.length)), ` of ${fmtInt(rows.length)} runs shown`,
      ' · ', COLLECTION_ORDER.map((c, i) => [i ? ', ' : '', `${COLLECTIONS[c].label} ${fmtInt(per[c] || 0)}`]),
      ' · ', `${fmtInt(shown.filter(r => r.agree === 'disagree').length)} where code and a judge disagree`));
    pager.replaceChildren(
      h('button', { type: 'button', class: 'btn btn-small', disabled: state.page === 0, onclick: () => { state.page--; draw(); } }, '← previous'),
      h('span', { class: 'muted small' }, ` page ${state.page + 1} of ${pages} `),
      h('button', { type: 'button', class: 'btn btn-small', disabled: state.page >= pages - 1, onclick: () => { state.page++; draw(); } }, 'next →'),
      select(PAGE_SIZES, state.size, v => { state.size = v; state.page = 0; draw(); }, { 'aria-label': 'Rows per page' }));
    syncUrl();
  }
  draw();

  const about = h('details', { class: 'sub about' },
    h('summary', null, 'What the collections and labels mean'),
    h('div', { class: 'about-body' },
      h('ul', { class: 'plain-list' },
        COLLECTION_ORDER.map(c => h('li', null, h('strong', null, `${COLLECTIONS[c].label} (${COLLECTIONS[c].tag}). `), COLLECTIONS[c].about))),
      h('p', null, 'SGA is SparGuardianAgents: it runs CheatBench tasks on the ORBIT scaffold and labels each run by code. ',
        'Our sprint runs used SGA from our own branches. The board-seed change is ours. ',
        'In run names, "solo" means the uninstructed cell, not SGA\'s solo-task mode: no run in any collection used that mode.'),
      h('p', null, h('strong', null, 'Cheating rung (code ladder). '), RUNG_ORDER.map(k => `${k}: ${RUNG_MEANING[k]}`).join('; '), '.'),
      h('p', null, h('strong', null, 'Collaboration (code ladder). '), COLLAB_ORDER.map(k => `${k}: ${COLLAB_MEANING[k]}`).join('; '), '.'),
      h('p', null, h('strong', null, 'Judges. '), JUDGE_ORDER.map(k => `${JUDGES[k].label}: ${JUDGES[k].about}`).join(' ')),
      h('p', null, 'No judge read the board-seed runs, and no judge rated collaboration anywhere. ',
        `Index built ${idx.built_at} in ${idx.build_seconds} s from the data files; restart with --rebuild after the data change.`)));

  return h('div', { class: 'page wide' },
    h('div', { class: 'page-head' }, h('h1', null, 'Runs'), muted('One row per run. Click a run to see its labels, the evidence, the board and the transcripts.')),
    about, filterBox, summary, pager,
    h('div', { class: 'table-wrap idx-wrap' }, table));
}
