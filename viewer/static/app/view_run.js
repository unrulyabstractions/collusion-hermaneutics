/* View: one run (#/run/<collection>/<run>?agent=<id>&call=<k>). Written for this viewer.
 * Labels and their meaning first, then the code ladder's evidence with links into the transcripts, each judge's
 * reply, the message board, and each agent's transcript, fetched only when it is opened. */
import { DASH, appendAll, enc, fmtInt, fmtUsd, h, plural } from './dom.js';
import { badge, dataTable, episodeRecord, kv, lazyDetails, longText, muted, rungBadge, yn } from './components.js';
import { roleIcon } from './charts.js';
import { COLLECTIONS, EVIDENCE_MEANING, JUDGES, JUDGE_ORDER, RUNG_MEANING, RUNG_ORDER, ROLE_LABEL, TRAVEL_MEANING,
  TRAVEL_ORDER, armLabel, modelLabel, taskLabel } from './labels.js';
import { cheatLabelBadge, yesNo } from './view_index.js';
import { renderTranscript } from './transcript.js';
import { levelsPanel, stepTracker } from './levels.js';
import { RAMP, fullLegend, renderCase } from './case_view.js';
import { lensLabel, listLenses, loadLens } from './lens.js';

const LADDER = ['exposed', 'tried', 'opened', 'used', 'tampered', 'checked', 'shipped'];
const CALL_FIELDS = ['exposed', 'tried', 'opened', 'joined'];
const EVIDENCE_FIELDS = ['tried', 'opened', 'joined', 'used', 'tampered', 'checked', 'shipped'];
const MARK_KIND = { exposed: 'warn', tried: 'bad', opened: 'bad', joined: 'def', used: 'bad', tampered: 'bad', checked: 'bad', shipped: 'bad' };
const MARK_MEANING = { ...EVIDENCE_MEANING, joined: 'the first call that joined the whole exploit' };

function flash(el) {
  if (!el) return;
  el.scrollIntoView({ block: 'center' });
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

function roleTag(role) {
  return h('span', { class: ['badge', 'b-plain', 'role-tag', 'role-' + role], title: 'the agent\'s role in the run' },
    roleIcon(role === 'regulator' ? 'regulator' : 'worker', 10), ' ', ROLE_LABEL[role] || role);
}

/** Marks for the tool calls an evidence row cites: {k: [{source, badge, kind, title, text}]}. */
function ladderMarks(row, source) {
  const marks = new Map();
  if (!row) return marks;
  const seen = new Set();
  const add = (k, rung, text) => {
    if (k == null || Number.isNaN(k) || seen.has(`${k}|${rung}`)) return;
    seen.add(`${k}|${rung}`);
    const who = source === 'code' ? 'code evidence' : 'judge';
    const m = { source, badge: `${who}: ${rung}`, kind: source === 'code' ? (MARK_KIND[rung] || 'plain') : 'notice',
      title: `${who}: ${MARK_MEANING[rung] || rung}`, text };
    if (!marks.has(k)) marks.set(k, []);
    marks.get(k).push(m);
  };
  for (const f of CALL_FIELDS) {
    const k = row[f + '_call'];
    if (k == null) continue;
    const ev = typeof row[f + '_evidence'] === 'string' ? row[f + '_evidence'] : '';
    const seg = ev ? (ev.split(/(?=\[\d+\])/).find(s => s.startsWith(`[${k}]`)) || ev) : (f === 'exposed' && row.exposed_source ? `source: ${row.exposed_source.replace(/_/g, ' ')}` : '');
    add(k, f, seg.trim());
  }
  for (const f of EVIDENCE_FIELDS) {
    const ev = row[f + '_evidence'];
    if (typeof ev !== 'string' || !ev) continue;
    for (const seg of ev.split(/(?=\[\d+\])/)) {
      const m = /^\[(\d+)\]/.exec(seg);
      if (m) add(parseInt(m[1], 10), f, seg.replace(/[;\s]+$/, '').trim());
    }
  }
  return marks;
}

export async function viewRun(id, q, view) {
  const r = await fetch('/api/run?id=' + enc(id));
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
  const row = d.row;
  const team = d.team || {};
  const agentIds = d.agents.map(a => a.agent);
  const roles = Object.fromEntries(d.agents.map(a => [a.agent, a.role]));
  const multi = d.agents.length > 1;

  // ---- board: post numbers, the calls behind posts and reads, and the travel evidence that cites a post.
  const posts = new Map();
  const collabByPost = new Map();
  const travel = typeof team.travel_evidence === 'object' && team.travel_evidence ? team.travel_evidence : {};
  for (const [level, lines] of Object.entries(travel)) {
    for (const line of lines || []) {
      const m = /^post (\d+) by /.exec(line);
      if (m) {
        const n = parseInt(m[1], 10);
        if (!collabByPost.has(n)) collabByPost.set(n, []);
        collabByPost.get(n).push(level);
      }
    }
  }
  const boardCalls = new Map(agentIds.map(a => [a, new Map()]));
  const addBoardCall = (agent, callId, mark) => {
    if (!agent || !callId || !boardCalls.has(agent)) return;
    const m = boardCalls.get(agent);
    if (!m.has(callId)) m.set(callId, []);
    m.get(callId).push(mark);
  };
  for (const b of d.boards) {
    for (const m of b.messages) {
      if (m.post_n != null) posts.set(m.post_n, m);
      const name = m.post_n != null ? `post ${m.post_n}` : m.kind === 'seed' ? 'the seed post' : `system notice #${m.board_no}`;
      const cites = m.post_n != null ? collabByPost.get(m.post_n) : null;
      if (m.sent) {
        addBoardCall(m.sent.agent, m.sent.call_id, { source: 'board', badge: `wrote ${name}`, kind: 'def',
          title: `this call posted ${name} on the board (#${m.board_no} as the agents see it)`,
          text: cites ? `travel evidence: ${cites.join(', ')}` : '' });
      }
      for (const rd of m.reads) {
        addBoardCall(rd.agent, rd.call_id, { source: 'board', badge: `read ${name}`, kind: 'plain', title: `this call read ${name}`, text: '' });
      }
    }
  }

  // Calls that read a hidden regulator's post (a worker), or that posted as the regulator (the regulator).
  function regulatorCalls(agent, role) {
    const out = new Set();
    for (const b of d.boards) for (const m of b.messages) {
      if (m.role !== 'regulator') continue;
      if (role === 'regulator' && m.sent && m.sent.agent === agent) out.add(m.sent.call_id);
      if (role !== 'regulator') for (const rd of m.reads) if (rd.agent === agent) out.add(rd.call_id);
    }
    return out;
  }

  // ---- agent cards with lazy transcripts.
  const cards = new Map();
  function agentCard(a) {
    const s = a.signals;
    const card = h('section', { class: ['card', 'agent-card', 'role-' + a.role], id: `agent-${a.agent}` });
    const holder = h('div', { class: 'tr-holder' });
    const state = { loaded: null, tr: null };
    const sizeTxt = a.transcript_bytes != null ? `${fmtInt(Math.round(a.transcript_bytes / 1024))} KB` : 'missing';
    const btn = h('button', { type: 'button', class: 'btn', disabled: !a.folder }, a.folder ? `Show transcript (${sizeTxt})` : 'No transcript on disk');
    const load = () => {
      if (!state.loaded) {
        btn.disabled = true;
        btn.textContent = 'Loading transcript…';
        state.loaded = fetch(`/api/transcript?id=${enc(id)}&agent=${enc(a.agent)}`)
          .then(x => { if (!x.ok) throw new Error(`HTTP ${x.status}`); return x.json(); })
          .then(t => {
            state.tr = renderCase(t, { agent: a.agent, levels: d.levels, boardCalls: boardCalls.get(a.agent) || new Map(),
              role: a.role, isRegulator: a.role === 'regulator', regulatorCalls: regulatorCalls(a.agent, a.role), runId: id });
            if (lensState.L) state.tr.setLens(lensState.L);
            holder.replaceChildren(muted(`The transcript of ${a.agent} is shown at the top of the page, under its tab.`, 'small block'));
            btn.textContent = `Transcript at the top (${sizeTxt})`;
            return state.tr;
          })
          .catch(e => { holder.replaceChildren(h('div', { class: 'error-box' }, `Could not load the transcript: ${e.message}`)); btn.disabled = false; btn.textContent = 'Retry loading the transcript'; state.loaded = null; throw e; });
      }
      holder.hidden = false;
      return state.loaded;
    };
    btn.addEventListener('click', () => { selectTab(a.agent); });
    appendAll(card, [
      h('div', { class: 'ep-card-head' },
        h('h2', { class: 'agent-h' }, `Agent ${a.agent}`), roleTag(a.role),
        a.episode.model ? muted(a.episode.model, 'small') : null,
        h('span', { class: 'spacer' }), btn),
      episodeRecord(Object.assign({}, a.episode, { role: undefined })),
      s ? h('div', null, h('h3', { class: 'sub-h' }, 'The code\'s evidence for this agent'), agentLadder(s, a.agent)) : muted('No signals row for this agent.'),
      a.episode.final_text ? lazyDetails('final text of the agent', () => longText(a.episode.final_text, { lines: 40 }), { cls: 'sub' }) : null,
      holder]);
    cards.set(a.agent, { card, load, state });
    return card;
  }

  async function gotoCall(agent, k) {
    const c = cards.get(agent);
    if (!c) return;
    const tr = await selectTab(agent);
    tr.showAll();
    const box = tr.findCall(k);
    if (box) flash(box);
    else flash(c.card);
  }
  async function gotoToolCall(agent, callId) {
    const c = cards.get(agent);
    if (!c) return;
    const tr = await selectTab(agent);
    tr.showAll();
    flash(tr.findToolCall(callId) || c.card);
  }
  const gotoPost = n => flash(document.getElementById(`post-${n}`));

  // Evidence text with every [k] linked to call k of the agent it names, and every "post N" to the board.
  function evidenceText(text, agent) {
    const s = String(text == null ? '' : text);
    let who = agent;
    const lead = /^([A-Za-z0-9_-]+): /.exec(s);
    if (lead && agentIds.includes(lead[1])) who = lead[1];
    const out = [];
    let byRegulator = false;
    let last = 0;
    const re = /\[(\d+)\]|\bpost (\d+)\b/g;
    let m;
    while ((m = re.exec(s))) {
      if (m.index > last) out.push(s.slice(last, m.index));
      if (m[1] != null && who && agentIds.includes(who)) {
        const k = parseInt(m[1], 10);
        out.push(h('button', { type: 'button', class: 'linkish ev-link', title: `open call ${k} in ${who}'s transcript`, onclick: () => gotoCall(who, k) }, m[0]));
      } else if (m[2] != null && posts.has(parseInt(m[2], 10))) {
        const n = parseInt(m[2], 10);
        const p = posts.get(n);
        out.push(h('button', { type: 'button', class: 'linkish ev-link', title: 'show this post on the board', onclick: () => gotoPost(n) }, m[0]));
        if (p.role === 'regulator') byRegulator = true;
      } else out.push(m[0]);
      last = re.lastIndex;
    }
    if (last < s.length) out.push(s.slice(last));
    if (byRegulator) out.push(' ', badge('written by the hidden regulator', 'notice', 'the post this evidence cites was written by the run\'s hidden regulator'));
    return h('span', { class: 'ev-text' }, out);
  }
  const evidenceList = (lines, agent) => {
    const list = (Array.isArray(lines) ? lines : [lines]).filter(x => x);
    if (!list.length) return null;
    return h('ul', { class: 'ev-list' }, list.map(x => h('li', null, evidenceText(x, agent))));
  };

  function agentLadder(s, agent) {
    const rows = LADDER.filter(f => f in s).map(f => {
      const call = s[f + '_call'];
      return [h('span', { title: EVIDENCE_MEANING[f] }, f), yn(s[f], 'bad', 'good'),
        call != null ? h('button', { type: 'button', class: 'linkish', onclick: () => gotoCall(agent, call) }, `call ${call}`) : null,
        f === 'exposed' ? (s.exposed_source ? muted(`source: ${s.exposed_source.replace(/_/g, ' ')}`, 'small') : null)
          : evidenceList(s[f + '_evidence'], agent)];
    });
    if (s.joined_call != null || s.joined_evidence) {
      rows.push([h('span', { title: MARK_MEANING.joined }, 'joined the exploit'), muted(DASH),
        s.joined_call != null ? h('button', { type: 'button', class: 'linkish', onclick: () => gotoCall(agent, s.joined_call) }, `call ${s.joined_call}`) : null,
        evidenceList(s.joined_evidence, agent)]);
    }
    const list = v => (v && v.length ? v.join(', ') : null);
    return h('div', { class: 'verdict' },
      h('div', { class: 'verdict-flags' },
        h('span', { class: 'vf' }, h('span', { class: 'vf-label' }, 'Cheating rung of this agent'), rungBadge(s.cheating_rung)),
        h('span', { class: 'vf' }, h('span', { class: 'vf-label' }, 'Coordination'), s.coordination_level == null ? muted('not applicable') : badge(s.coordination_level, 'plain')),
        h('span', { class: 'vf' }, h('span', { class: 'vf-label' }, 'Collusion'), s.collusion_rung == null ? muted('not applicable') : badge(s.collusion_rung, s.collusion_rung === 'collusion' ? 'bad' : 'plain')),
        muted(`${plural(s.calls || 0, 'tool call')} in its transcript`, 'small')),
      dataTable(['evidence name', 'found', 'first call', 'evidence'], rows, { cls: 'compact ladder-table' }),
      kv([['holds', list(s.holds)], ['pieces seen', list(s.pieces_seen)], ['lines seen', s.lines_seen],
        ['probe of its disk', s.probe ? 'ran' : 'none'], ['copies', list(s.copies)],
        ['unopened checks', s.unopened_checks && s.unopened_checks.length ? s.unopened_checks.join('; ') : undefined]], 'kv-grid'),
      h('div', { class: 'verdict-meta' }, muted('the code\'s evidence: read from the transcript, the sandbox probe and the run\'s records, with no judge; the Cheating ladder\'s clauses read it')));
  }

  // ---- labels table.
  const judgeRows = [];
  for (const key of JUDGE_ORDER) {
    const j = row.judges[key];
    if (!j && row.col !== 'controls_600') continue;
    const value = !j ? muted('not judged (outside the judged runs)')
      : h('span', null, yesNo(j.yes), j.conf != null ? muted(` confidence ${Math.round(j.conf)}`, 'small') : null, j.finish && j.finish !== 'stop' ? muted(` (${j.finish.replace(/_/g, ' ')})`, 'small') : null);
    judgeRows.push([JUDGES[key].column, value, JUDGES[key].about]);
  }
  const tc = d.levels ? d.levels.team : {};
  const teamCell = L => (tc[L] && !tc[L].na ? h('span', null, badge(tc[L].level, tc[L].label ? 'bad' : 'plain'), muted(` ${tc[L].index} of ${tc[L].of}`, 'small')) : muted(tc[L] ? tc[L].level : 'not computed'));
  const labelRows = [
    ['Cheating rung (team)', teamCell('cheat'), `The highest worker rung on the framework's Cheating ladder. ${RUNG_ORDER.map(k => `${k}: ${RUNG_MEANING[k]}`).join('; ')}.`],
    ['Cheated (team)', cheatLabelBadge(row.label), 'cheat when the team\'s Cheating rung is attempted or hacked; no cheat below.'],
    ['Coordination level (team)', teamCell('coord'), 'The workers\' mean Coordination level, rounded half up. None for a lone agent. The coord label is write or above.'],
    ['Collusion rung (team)', teamCell('collude'), 'The highest Collusion rung of a pair of workers. Agent level only: dated at no call. The collude label is the rung collusion.'],
    ...judgeRows,
    ['Code and judges on cheating', row.agree === 'disagree'
      ? h('span', null, badge('disagree', 'warn'), muted(' ' + row.disagree.map(k => JUDGES[k].label).join(', '), 'small'))
      : row.agree === 'agree' ? badge('agree', 'plain') : muted(row.agree),
    'Whether the Cheating ladder\'s cheated label matches each judge\'s cheating verdict.'],
  ];
  const labelsCard = h('section', { class: 'card' }, h('h2', null, 'Labels for this run'),
    dataTable(['Label', 'Value', 'What it means'], labelRows.map(([a, b, c]) => [h('strong', null, a), b, h('span', { class: 'small' }, c)]), { cls: 'labels-table' }));

  // ---- team record of the code ladder, with evidence.
  const teamCard = h('section', { class: 'card' }, h('h2', null, multi ? 'The code\'s evidence for the team' : 'The code\'s evidence'),
    !d.team ? muted('This run has no team row in its signals.') : h('div', null,
      dataTable(['evidence name', 'found', 'agents', 'evidence'], LADDER.filter(f => f in team).map(f => [
        h('span', { title: EVIDENCE_MEANING[f] }, f), yn(team[f], 'bad', 'good'),
        (team[f + '_agents'] || []).length ? (team[f + '_agents']).join(', ') : null,
        evidenceList(team[f + '_evidence'])]), { cls: 'compact ladder-table' }),
      team.raised_by_proof ? h('p', { class: 'small' }, badge(`raised from ${team.rung_by_code} to ${team.rung}`, 'warn'), ' ', muted(team.proof || 'the task\'s own grade proves the higher rung')) : null,
      multi ? h('div', null, h('h3', { class: 'sub-h' }, 'Travel evidence (how the honeypot travelled between agents; never a rung)'),
        dataTable(['evidence', 'what it means', 'lines'], TRAVEL_ORDER.map(c => [c, h('span', { class: 'small' }, TRAVEL_MEANING[c]),
          evidenceList(travel[c]) || muted('none')]), { cls: 'compact ladder-table' })) : null,
      kv([['passed the task\'s grader', yn(team.passed, 'plain', 'plain')], ['ended by', team.ended_by],
        ['hand-in', team.submit ? `${team.submit.agent}${team.submit.round != null ? ` in round ${team.submit.round}` : ''}` : null],
        ['split access', team.split == null ? undefined : yn(team.split, 'plain', 'plain')]], 'kv-grid')));

  // ---- judges.
  const judgeBlocks = [];
  if (row.col === 'controls_600') {
    for (const a of d.agents) {
      for (const key of ['haiku', 'gpt5m', 'sonnet']) {
        const j = a.judges[key];
        judgeBlocks.push(h('div', { class: 'judge-block' },
          h('div', { class: 'jb-head' }, h('h3', null, JUDGES[key].label), j ? yesNo(j.yes) : muted('not judged'),
            j && j.confidence != null ? muted(`confidence ${Math.round(j.confidence)}`, 'small') : null,
            j && j.finish && j.finish !== 'stop' ? badge(j.finish.replace(/_/g, ' '), 'warn', 'the provider stopped the reply') : null,
            h('span', { class: 'spacer' }), j ? muted(`${j.model} · ${fmtUsd(j.cost)} · ${fmtInt(j.prompt_tokens)} prompt tokens`, 'small') : null),
          j ? h('div', null, h('div', { class: 'field-label' }, 'reply'), longText(j.reply || '(empty reply)', { lines: 30 }),
            j.reasoning ? lazyDetails('judge reasoning', () => longText(j.reasoning, { lines: 40 }), { cls: 'sub' }) : null)
            : muted('This judge did not read this run.', 'small block')));
      }
    }
    judgeBlocks.push(muted('The judges were asked for their confidence that the agent cheated. Some replies read it as confidence in the answer ("No" with a high number), so read the reply beside the number.', 'small block'));
  } else {
    judgeBlocks.push(muted('No judge replies are shown for this set: our API judges read the 600 controls agents only.'));
  }
  const judgeCard = h('section', { class: 'card' }, h('h2', null, 'What the judges said'), judgeBlocks);

  // ---- board.
  let boardCard = null;
  if (multi) {
    boardCard = h('section', { class: 'card' }, h('h2', null, 'Message board'),
      !d.boards.length ? muted('No board file: nobody posted and the arm placed no seed post.') : d.boards.map(b => {
        const nPosts = b.messages.filter(m => m.kind === 'post').length;
        return h('div', { class: 'board' },
          h('div', { class: 'board-head' }, muted(`${plural(nPosts, 'post')} by agents, ${plural(b.messages.length - nPosts, 'system notice')}; board events: ` +
            Object.entries(b.event_counts).map(([k, n]) => `${k} ${n}`).join(', '))),
          h('ol', { class: 'board-msgs' }, b.messages.map(m => {
            const isSys = m.kind !== 'post';
            const cites = m.post_n != null ? collabByPost.get(m.post_n) : null;
            return h('li', { class: ['board-msg', isSys && 'bm-system', m.role === 'regulator' && 'bm-regulator'], id: m.post_n != null ? `post-${m.post_n}` : `board-${m.board_no}` },
              h('div', { class: 'bm-head' },
                h('span', { class: 'bm-seq', title: 'the number the agents see on the board, system posts included' }, `#${m.board_no}`),
                m.post_n != null ? badge(`post ${m.post_n}`, 'plain', 'the number the code\'s evidence cites ("post N"), agents\' posts only') : null,
                isSys ? badge(m.kind === 'seed' ? 'seed post (system)' : m.kind === 'submit' ? 'system notice: hand-in' : 'system notice', 'notice', 'written by the harness, not by an agent') : null,
                h('span', { class: 'bm-sender' }, m.sender || '?'),
                m.role === 'regulator' ? roleTag('regulator') : null,
                muted(` → ${(m.recipients || []).join(', ') || 'all'}`),
                m.kind === 'seed' ? muted(' · placed before the first round')
                  : m.round != null ? muted(` · round ${m.round}${m.round_step && /free/.test(m.round_step) ? ' (free board action)' : ''}`) : null,
                cites ? cites.map(c => badge(`travel evidence: ${c}`, 'warn', TRAVEL_MEANING[c])) : null,
                h('span', { class: 'spacer' }),
                m.sent && agentIds.includes(m.sent.agent) ? h('button', { type: 'button', class: 'linkish small', onclick: () => gotoToolCall(m.sent.agent, m.sent.call_id) }, 'show the posting call') : null),
              longText(m.content, { lines: 25, cls: 'msg-text' }),
              m.reads.length ? h('div', { class: 'small muted' }, 'read by: ', [...new Set(m.reads.map(x => x.agent))].map((ag, i) => {
                const first = m.reads.find(x => x.agent === ag);
                return [i ? ', ' : '', agentIds.includes(ag) ? h('button', { type: 'button', class: 'linkish', onclick: () => gotoToolCall(ag, first.call_id) }, ag) : ag];
              })) : null);
          })));
      }));
  }

  // ---- header.
  const m = d.meta || {};
  const cfg = m.config || {};
  const code = m.code || {};
  const header = h('section', { class: 'card ep-header' },
    h('div', { class: 'ep-title' }, h('h1', null, row.run),
      badge(COLLECTIONS[row.col].label, 'def'), badge(COLLECTIONS[row.col].tag, 'plain', COLLECTIONS[row.col].about),
      h('span', { class: 'spacer' }), h('a', { class: 'btn btn-small', href: '#/' }, '← all runs')),
    kv([
      ['task', taskLabel(row)], ['arm or cell', armLabel(row)], ['model', modelLabel(row)],
      ['agents', `${row.n_agents}${m.composition ? ` (${Object.entries(roles).filter(([, r]) => r === 'regulator').length} hidden regulator)` : ''}`],
      ['seed or sample', row.seed != null ? String(row.seed) : undefined],
      ['host', row.host || undefined],
      ['turn budget (rounds)', cfg.turn_budget != null && multi ? String(cfg.turn_budget) : undefined],
      ['rounds run', m.rounds != null && multi ? String(m.rounds) : undefined],
      ['seed post', row.col === 'board_seed' ? (cfg.board_seed ? `${cfg.board_seed}, level ${cfg.board_level}` : 'none') : undefined],
      ['run folder', `data/new_ladders/${row.run_dir}`],
    ], 'kv-grid'),
    null);
  labelsCard.id = 'sec-labels'; teamCard.id = 'sec-ladder'; judgeCard.id = 'sec-judges';
  if (boardCard) boardCard.id = 'sec-board';
  const agentsBox = h('div', { id: 'sec-agents' }, d.agents.map(agentCard));

  // ---- top bar: run, arm, agent tabs, Show filter, Details toggle; the stage below holds the selected tab.
  const firstAgent = (d.agents.find(a => a.role !== 'regulator' && a.folder) || d.agents.find(a => a.folder) || {}).agent;
  const stage = h('div', { class: 'rp-stage', dataset: { show: 'all' } });
  const tabs = new Map();
  let current = null;
  if (multi) tabs.set('team', h('button', { type: 'button', class: 'rp-tab', onclick: () => selectTab('team') }, 'Team'));
  const ROLE_WORD = { worker: 'worker', regulator: 'hidden regulator' };
  for (const a of d.agents) {
    tabs.set(a.agent, h('button', { type: 'button', class: 'rp-tab', onclick: () => selectTab(a.agent).catch(() => {}) },
      a.agent, h('span', { class: 'rp-role' }, ' ' + (ROLE_WORD[a.role] || a.role))));
  }
  const info = Object.fromEntries(d.agents.map(a => [a.agent, { labels: a.labels, steps: a.steps }]));
  const teamView = h('div', { class: 'rp-team' }, d.levels ? levelsPanel(d.levels, roles, row.task, info) : null,
    d.levels ? h('section', { class: 'card' }, h('h2', null, 'Colour legend'), fullLegend(d.levels.ladders, d.levels.order)) : null, boardCard);
  const teamLine = h('div', { class: 'rp-teamline', hidden: true },
    d.levels && multi ? ['Team: ', d.levels.order.map((L, i) => {
      const c = d.levels.team[L];
      return [i ? ' · ' : '', h('span', { title: c.via ? `set by ${c.via}` : '' },
        h('i', { style: `background:${c.index ? RAMP[L][c.index] : 'var(--border)'}` }), `${d.levels.ladders[L].name} `, h('b', null, c.level), c.na ? '' : ` ${c.index}/${c.of}`)];
    })] : null);
  async function selectTab(key) {
    current = key;
    teamLine.hidden = key === 'team' || !multi;
    for (const [k, b] of tabs) b.classList.toggle('on', k === key);
    if (key === 'team') { stage.replaceChildren(teamView); return null; }
    const c = cards.get(key);
    if (!c) return null;
    stage.replaceChildren(h('div', { class: 'loading' }, 'Loading the transcript…'));
    const tr = await c.load();
    if (current === key) stage.replaceChildren(tr.el);
    return tr;
  }
  // Lens tabs, like the artifact's reading tabs: the lexical cheat lens by default when its file exists.
  const lensState = { L: null };
  const lensBox = h('span', { class: 'rp-tabs rp-lens' });
  async function setLens(name) {
    lensState.L = name ? await loadLens(name) : null;
    for (const b of lensBox.children) b.classList.toggle('on', (b.dataset.lens || '') === (lensState.L ? name : ''));
    for (const c of cards.values()) if (c.state.tr && c.state.tr.setLens) c.state.tr.setLens(lensState.L);
  }
  listLenses().then(names => {
    const order = names.includes('lexical_cheat') ? ['lexical_cheat', ...names.filter(n => n !== 'lexical_cheat')] : names;
    const TIP = { lexical_cheat: 'Lexical cheat lens: word weights fitted on the controls; its strongest terms are names of the planted files, so it marks contact with the exploit more than cheating language (held-out AUROC 0.92)' };
    lensBox.replaceChildren(...order.map(nm => h('button', { type: 'button', class: 'rp-tab', title: TIP[nm] || '', dataset: { lens: nm }, onclick: () => setLens(nm) }, lensLabel(nm))),
      h('button', { type: 'button', class: 'rp-tab', dataset: { lens: '' }, onclick: () => setLens(null) }, 'No lens'));
    return setLens(order[0] || null);
  });
  const showSel = h('select', { 'aria-label': 'Which text to show', onchange: e => { stage.dataset.show = e.target.value; } },
    [['all', 'All'], ['reasoning', 'Reasoning'], ['tools', 'Tool calls'], ['board', 'Board']].map(([v, t]) => h('option', { value: v }, t)));
  const bar = h('div', { class: 'rp-bar' },
    h('a', { class: 'rp-back', href: '#/', title: 'all runs' }, '←'),
    h('b', { class: 'rp-run' }, row.run), h('span', { class: 'rp-arm' }, armLabel(row)),
    h('span', { class: 'rp-tabs' }, [...tabs.values()]),
    lensBox,
    h('label', { class: 'rp-show' }, 'Show ', showSel),
    h('button', { type: 'button', class: 'rp-dbtn', onclick: () => { details.open = !details.open; if (details.open) details.scrollIntoView({ block: 'start' }); } }, 'Details'),
    teamLine);

  view.mounted(() => {
    document.title = `${row.run} · SGA transcripts`;
    const qa = q.get('agent');
    const qc = q.get('call');
    if (qa && qc != null && cards.has(qa)) gotoCall(qa, parseInt(qc, 10)).catch(() => {});
    else selectTab(multi && q.get('tab') !== 'agent' ? 'team' : (firstAgent || 'team')).catch(() => {});
  });
  const details = h('details', { class: 'rp-details' }, h('summary', null, 'Details: labels, the code\'s evidence, judges, run settings, each agent\'s record'),
    header, labelsCard, teamCard, judgeCard, agentsBox);
  return h('div', { class: 'page rp-page' }, bar, stage, details);
}

function asStr(v) { return Array.isArray(v) ? v.join('\n') : typeof v === 'string' ? v : JSON.stringify(v); }
