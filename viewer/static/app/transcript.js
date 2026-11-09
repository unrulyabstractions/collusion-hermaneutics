/* One agent's transcript (ATIF transcript.json) as readable turns.
 *
 * Adapted from SparGuardianAgents (unrulyabstractions/spar-guardian-agents) @ 26b218e,
 * src/ui/static/app/view_transcript.js lines 12-145 (STEP_KINDS, FLAG_BADGE, stepPreview, resultView, renderStep,
 * stepMatchesKind); preexisting code with these edits of ours:
 *   - plain-language labels: "harness notice" for the turn-rules flag and caption, "assistant" for agent steps;
 *   - unreadable reasoning (Haiku's encrypted blocks, SPAR's "[redacted reasoning]") shows as hidden by the provider;
 *   - an error result whose text equals its error message prints once;
 *   - every tool call carries its number k (0-based over the agent's calls, SGA rungs.py calls_of), an anchor, and
 *     the code ladder's and the judge's marks that cite it, and the board posts and reads it made;
 *   - step and call ids are prefixed with the agent so several transcripts fit on one page;
 *   - the sidebar and the eval-log links are left out; a find box and a kind filter sit above the steps.
 */
import { asText, debounce, firstLine, fmtInt, h, plural } from './dom.js';
import { badge, extraBadges, isBlockedResult, lazyDetails, longText, muted, reasoningView, select, splitSignature,
  toolCallView } from './components.js';

const STEP_KINDS = [
  ['', 'All'],
  ['agent', 'assistant steps'],
  ['user', 'user steps'],
  ['system', 'system steps'],
  ['tools', 'steps with tool calls'],
  ['cited', 'steps the code ladder or a judge cites'],
  ['board', 'steps that post to or read the board'],
  ['notice', 'harness notices'],
  ['error', 'tool errors'],
];

const FLAG_BADGE = {
  blocked: ['BLOCKED', 'blocked'],
  refusal: ['REFUSAL', 'refusal'],
  'refusal-text': ['REFUSAL TEXT (model-written)', 'plain'],
  withheld: ['WITHHELD', 'withheld'],
  rules: ['harness notice', 'notice'],
  error: ['tool error', 'bad'],
};

const SRC_LABEL = { system: 'system', user: 'user', agent: 'assistant' };

export const stepAnchor = (agent, id) => `a-${agent}-step-${id}`;
export const callAnchor = (agent, k) => `a-${agent}-call-${k}`;

/** Reasoning as text, or a note when the provider hid it. */
export function readableReasoning(raw) {
  if (raw == null || raw === '') return null;
  const s = typeof raw === 'string' ? raw : JSON.stringify(raw);
  const t = s.trim();
  if (t === '[redacted reasoning]') return { hidden: 'reasoning hidden by the provider (redacted)' };
  if (t.startsWith('[{')) {
    try {
      const blocks = JSON.parse(t);
      if (Array.isArray(blocks)) {
        const part = b => {
          if (!b || typeof b !== 'object') return '';
          if (typeof b.text === 'string') return b.text;
          if (typeof b.summary === 'string') return b.summary;
          if (Array.isArray(b.summary)) return b.summary.map(x => (x && x.text) || '').join('\n');
          return '';
        };
        const text = blocks.map(part).filter(Boolean).join('\n\n');
        return text ? { text } : { hidden: `reasoning hidden by the provider (${plural(blocks.length, 'encrypted block')})` };
      }
    } catch (e) { /* not JSON: show as text */ }
  }
  return { text: s };
}

function stepPreview(s, msg) {
  if (msg.trim()) return firstLine(msg, 100);
  const tc = (s.tool_calls || [])[0];
  if (tc) {
    let a = tc.arguments;
    if (a && typeof a === 'object') a = a.command || a.code || a.message || a.answer || JSON.stringify(a);
    return `${tc.function_name}: ${firstLine(asText(a), 90)}`;
  }
  if (s.reasoning_content) {
    const r = splitSignature(s.reasoning_content).text;
    return r ? '(reasoning) ' + firstLine(r.replace(/\*\*/g, ''), 90) : '(reasoning signature only)';
  }
  return '(empty)';
}

function resultView(r, orphan) {
  const rx = r.extra || {};
  const blocked = isBlockedResult(rx);
  const isErr = !blocked && (!!rx.is_error || !!rx.error);
  const withheld = !!rx.withheld_from_model_by_defense;
  const content = asText(r.content);
  const sameAsError = rx.error && String(rx.error).trim() === content.trim();
  return h('div', { class: ['result', blocked && 'is-blocked', isErr && 'is-error', withheld && 'is-withheld'] },
    h('div', { class: 'res-head' },
      h('span', { class: 'res-label' }, orphan ? 'result without a matching call' : 'result'),
      blocked ? badge('BLOCKED', 'blocked') : null,
      isErr ? badge('error', 'bad') : null,
      withheld ? badge('WITHHELD', 'withheld', 'withheld from the model by a defense') : null,
      extraBadges(rx, ['is_error', 'error', 'withheld_from_model_by_defense', 'withheld_text']),
      h('span', { class: 'spacer' }),
      orphan && r.source_call_id ? h('span', { class: 'tc-id' }, r.source_call_id) : null),
    rx.error ? h('pre', { class: 'pre errtext' }, String(rx.error)) : null,
    rx.withheld_text ? lazyDetails('original content (never reached the model)', () => longText(asText(rx.withheld_text), { lines: 30 }), { cls: 'sub' }) : null,
    sameAsError ? null : content ? longText(content, { lines: 30 }) : muted('(empty output)', 'small block'));
}

function marksView(marks) {
  if (!marks || !marks.length) return null;
  return h('div', { class: 'call-marks' }, marks.map(m => h('div', { class: ['call-mark', 'cm-' + m.source] },
    badge(m.badge, m.kind, m.title), m.text ? h('span', { class: 'cm-text' }, m.text) : null)));
}

function renderStep(s, i, k0, ctx) {
  const ex = s.extra || {};
  const results = (s.observation && s.observation.results) || [];
  const src = s.source || 'unknown';
  const msg = asText(s.message);
  const flags = new Set();
  if (ex.swarm_turn_rules) flags.add('rules');
  if (ex.withheld_from_model_by_defense) flags.add('withheld');
  if (ex.defense_refusal) flags.add('refusal');
  else if (ex.refusal_text_written_by_model) flags.add('refusal-text');
  for (const r of results) {
    const rx = r.extra || {};
    if (isBlockedResult(rx)) flags.add('blocked');
    else if (rx.is_error || rx.error) flags.add('error');
    if (rx.withheld_from_model_by_defense) flags.add('withheld');
  }
  const providerOnly = flags.has('rules');
  const sid = s.step_id != null ? s.step_id : i + 1;
  const el = h('article', {
    class: ['step', 'src-' + src, providerOnly && 'provider-only', flags.has('refusal') && 'is-refusal',
      flags.has('blocked') && 'has-blocked', ex.withheld_from_model_by_defense && 'is-withheld'],
    id: stepAnchor(ctx.agent, sid),
  });
  const srcLabel = src === 'user' && i === 1 ? 'user (the task)' : (SRC_LABEL[src] || src);
  const head = h('header', { class: 'step-head' },
    h('span', { class: 'step-id' }, `#${sid}`),
    h('span', { class: 'step-src' }, srcLabel),
    s.model_name ? muted(s.model_name, 'small') : null,
    [...flags].map(f => badge(...FLAG_BADGE[f])),
    extraBadges(ex, ['swarm_turn_rules', 'withheld_from_model_by_defense', 'withheld_text',
      'defense_refusal', 'blocked_output_message_id', 'refusal_text_written_by_model']),
    h('span', { class: 'spacer' }),
    s.llm_call_count != null ? muted(plural(s.llm_call_count, 'model call'), 'small') : null);
  el.appendChild(head);
  if (providerOnly) {
    el.appendChild(h('div', { class: 'caption caption-notice' },
      `Harness prompt (${ex.swarm_turn_rules}): the run's turn rules added this message; the model saw it, the agent did not write it.`));
  }
  if (ex.withheld_from_model_by_defense) {
    el.appendChild(h('div', { class: 'caption caption-withheld' }, 'WITHHELD: a defense withheld this message from the model; the model received the placeholder shown below.'));
    if (ex.withheld_text) el.appendChild(lazyDetails('original content (never reached the model)', () => longText(asText(ex.withheld_text), { lines: 30 }), { cls: 'sub' }));
  }
  const reasoning = readableReasoning(s.reasoning_content);
  if (reasoning && reasoning.text) el.appendChild(reasoningView(reasoning.text, 60));
  else if (reasoning && reasoning.hidden) el.appendChild(muted(reasoning.hidden, 'small block reasoning-hidden'));
  if (flags.has('refusal')) {
    el.appendChild(h('div', { class: 'caption caption-refusal' },
      'A defense blocked this completion and substituted the refusal sentence.'));
  } else if (flags.has('refusal-text')) {
    el.appendChild(h('div', { class: 'caption caption-notice' },
      'The model wrote the refusal sentence itself; no defense blocked this completion.'));
  }
  // Leading blank lines are dropped for display only.
  if (msg.trim()) el.appendChild(longText(msg.replace(/^\s*\n/, ''), { cls: 'msg-text' }));
  else if (!(s.tool_calls || []).length && !reasoning) el.appendChild(muted('(empty message)', 'small block'));

  // Tool calls are numbered over agent steps only (SGA rungs.py calls_of); results pair by id, then by position.
  const calls = src === 'agent' ? (s.tool_calls || []) : [];
  const used = new Set();
  const callEls = [];
  let cited = false;
  let onBoard = false;
  calls.forEach((tc, j) => {
    const k = k0 + j;
    let rs = results.filter(r => r.source_call_id === tc.tool_call_id);
    if (!rs.length && results[j] && !results[j].source_call_id && !used.has(results[j])) rs = [results[j]];
    rs.forEach(r => used.add(r));
    const box = toolCallView(tc.function_name, tc.tool_call_id, tc.arguments, rs.map(r => resultView(r, false)));
    box.id = callAnchor(ctx.agent, k);
    box.dataset.tcid = tc.tool_call_id || '';
    const marks = [...(ctx.marks.get(k) || []), ...(ctx.boardCalls.get(tc.tool_call_id) || [])];
    if ((ctx.marks.get(k) || []).length) cited = true;
    if ((ctx.boardCalls.get(tc.tool_call_id) || []).length) onBoard = true;
    const tcHead = box.querySelector('.tc-head');
    tcHead.insertBefore(h('span', { class: 'call-no', title: 'the call number the code ladder and the judges cite as [k]' }, `call ${k}`), tcHead.children[1]);
    if (marks.length) {
      box.classList.add('is-cited');
      tcHead.after(marksView(marks));
    }
    callEls.push([k, tc.tool_call_id, box]);
    el.appendChild(box);
  });
  for (const tc of src === 'agent' ? [] : (s.tool_calls || [])) {
    el.appendChild(toolCallView(tc.function_name, tc.tool_call_id, tc.arguments, []));
  }
  for (const r of results) if (!used.has(r)) el.appendChild(resultView(r, true));
  if (cited) el.classList.add('is-cited');

  const hay = [srcLabel, msg, reasoning && reasoning.text,
    (s.tool_calls || []).map(tc => `${tc.function_name} ${asText(tc.arguments)}`).join('\n'),
    results.map(r => `${asText(r.content)} ${(r.extra && r.extra.error) || ''}`).join('\n')]
    .filter(Boolean).join('\n').toLowerCase();
  return { step: s, el, flags, hay, src, cited, onBoard, callEls, nCalls: calls.length,
    preview: stepPreview(s, msg), hasTools: !!(s.tool_calls || []).length };
}

function stepMatchesKind(e, k) {
  switch (k) {
    case '': return true;
    case 'agent': case 'user': case 'system': return e.src === k;
    case 'tools': return e.hasTools;
    case 'cited': return e.cited;
    case 'board': return e.onBoard;
    case 'notice': return e.flags.has('rules');
    default: return e.flags.has(k);
  }
}

/**
 * Render a transcript. ctx: {agent, marks: Map<call k, [mark]>, boardCalls: Map<tool_call_id, [mark]>}.
 * A mark is {source: 'code'|'judge'|'board', badge, kind, title, text}.
 * Returns {el, findCall(k), findToolCall(id), showAll()}.
 */
export function renderTranscript(t, ctx) {
  const steps = (t && t.trajectory && t.trajectory.steps) || [];
  const entries = [];
  let k = 0;
  steps.forEach((s, i) => {
    const e = renderStep(s, i, k, ctx);
    k += e.nCalls;
    entries.push(e);
  });
  const byCall = new Map();
  const byTcid = new Map();
  for (const e of entries) for (const [n, id, box] of e.callEls) { byCall.set(n, [e, box]); if (id) byTcid.set(id, [e, box]); }
  const counts = { notice: 0, error: 0, cited: 0 };
  for (const e of entries) {
    if (e.flags.has('rules')) counts.notice++;
    if (e.flags.has('error')) counts.error++;
    if (e.cited) counts.cited++;
  }

  const find = h('input', { type: 'search', class: 'search small-search', placeholder: 'find in this transcript', 'aria-label': 'Find in this transcript' });
  let kindVal = '';
  const countEl = h('span', { class: 'muted small' });
  const kindSel = select(STEP_KINDS, '', v => { kindVal = v; apply(); }, { 'aria-label': 'Which steps to show' });
  function apply() {
    const needle = find.value.trim().toLowerCase();
    let n = 0;
    for (const e of entries) {
      const ok = (!needle || e.hay.includes(needle)) && stepMatchesKind(e, kindVal);
      e.el.hidden = !ok;
      if (ok) n++;
    }
    countEl.textContent = `${fmtInt(n)} of ${plural(entries.length, 'step')} shown`;
  }
  find.addEventListener('input', debounce(apply, 150));
  apply();
  const meta = t.trajectory && t.trajectory.final_metrics;
  const bar = h('div', { class: 'tr-bar' },
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Find'), find),
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Show'), kindSel),
    countEl,
    h('span', { class: 'spacer' }),
    muted(`${plural(k, 'tool call')} · ${plural(counts.cited, 'cited step')} · ${plural(counts.notice, 'harness notice')} · ${plural(counts.error, 'step')} with tool errors` +
      (meta ? ` · tokens in ${fmtInt(meta.total_prompt_tokens)}, out ${fmtInt(meta.total_completion_tokens)}` : ''), 'small'));
  const el = h('div', { class: 'transcript' }, bar,
    h('div', { class: 'steps' }, entries.length ? entries.map(e => e.el) : muted('This transcript has no steps.')));
  const showAll = () => {
    if (find.value || kindVal) { find.value = ''; kindVal = ''; kindSel.value = ''; apply(); }
  };
  return {
    el, showAll, nCalls: k,
    findCall: n => (byCall.get(n) || [null, null])[1],
    findToolCall: id => (byTcid.get(id) || [null, null])[1],
  };
}
