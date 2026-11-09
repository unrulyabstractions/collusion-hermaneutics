/* Boot and hash router of the SGA transcript viewer. Written for this viewer; the security rule is SGA's
 * (src/ui/static/app/main.js 3-5): every piece of run data goes in through textContent or text nodes, never innerHTML.
 *
 *   #/                       every run, with filters (view_index.js)
 *   #/run/<collection>/<run> one run: labels, evidence, judges, board, transcripts (view_run.js)
 *       ?agent=<id>&call=<k> opens that agent's transcript at tool call k
 */
import { h } from './dom.js';
import { viewIndex } from './view_index.js';
import { viewRun } from './view_run.js';
import { viewLadders } from './view_ladders.js';

const main = () => document.getElementById('main');
let token = 0;

function shell() {
  const app = document.getElementById('app');
  app.replaceChildren(
    h('header', { class: 'topbar' },
      h('a', { class: 'brand', href: '#/' }, 'SGA transcripts'),
      h('span', { class: 'muted small' }, 'code-ladder and judge labels for cheating and collaboration'),
      h('span', { class: 'spacer' }),
      h('a', { class: 'btn btn-small', href: '#/ladders' }, 'Ladders'), ' ',
      h('a', { class: 'btn btn-small', href: '#/' }, 'All runs')),
    h('main', { id: 'main', tabindex: '-1' }));
}

async function route() {
  const my = ++token;
  const hash = location.hash || '#/';
  const [path, qs] = hash.slice(1).split('?');
  const q = new URLSearchParams(qs || '');
  const after = [];
  const view = { mounted: fn => after.push(fn) };
  main().replaceChildren(h('div', { class: 'loading' }, 'Loading…'));
  try {
    let node;
    const m = /^\/run\/(.+)$/.exec(path);
    if (m) node = await viewRun(m[1].split('/').map(decodeURIComponent).join('/'), q, view);
    else if (path === '/ladders') { node = await viewLadders(q); document.title = 'Ladders · SGA transcripts'; }
    else { node = await viewIndex(q); document.title = 'Runs · SGA transcripts'; }
    if (my !== token) return;
    main().replaceChildren(node);
    if (m) window.scrollTo(0, 0);
    for (const fn of after) fn();
  } catch (e) {
    if (my !== token) return;
    main().replaceChildren(h('div', { class: 'error-box' }, h('div', { class: 'error-title' }, 'Error'),
      h('pre', { class: 'pre' }, String((e && e.stack) || e))));
  }
}

shell();
// The index rewrites its own query while filters change; only a change of page reroutes.
let lastPath = null;
window.addEventListener('hashchange', () => {
  const p = (location.hash || '#/').split('?')[0];
  if (p === lastPath && p === '#/') return;
  lastPath = p;
  route();
});
lastPath = (location.hash || '#/').split('?')[0];
route();
