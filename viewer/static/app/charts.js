/* Preexisting code: copied verbatim from SparGuardianAgents (unrulyabstractions/spar-guardian-agents)
 * @ 26b218e, src/ui/static/app/charts.js lines 8 and 10-53 (svgEl, ROLE_ORDER, roleGlyph, roleIcon). */

import { appendAll, h } from './dom.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
let vizUid = 0;
/** A fresh number for the ids of one chart's clip paths, markers, patterns and gradients. */
export function nextVizUid() { return ++vizUid; }
export const r2 = n => Math.round(n * 100) / 100;

/** Build an SVG element. Strings in children become text nodes; attrs never take markup. */
export function svgEl(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.setAttribute('class', Array.isArray(v) ? v.filter(Boolean).join(' ') : v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, typeof v === 'number' ? String(r2(v)) : String(v));
    }
  }
  appendAll(el, children);
  return el;
}

// Roles: shape carries the role (circle worker, square regulator, triangle disregulator);
// the Okabe-Ito colours fill the glyph through the --role-* tokens in style.css.
export const ROLE_ORDER = ['disregulator', 'worker', 'regulator'];
export const ROLE_PLURAL = { disregulator: 'disregulators', worker: 'workers', regulator: 'regulators' };

/** A role glyph centred on (cx, cy). cls replaces the role fill class (outcome fills in the cascade). */
export function roleGlyph(role, cx, cy, r, cls) {
  const c = ['glyph', cls || 'role-' + (ROLE_ORDER.includes(role) ? role : 'none')];
  if (role === 'regulator') {
    const a = r * 0.88;
    return svgEl('rect', { class: c, x: cx - a, y: cy - a, width: 2 * a, height: 2 * a });
  }
  if (role === 'disregulator') {
    const a = r * 1.22;
    return svgEl('polygon', { class: c, points: `${r2(cx)},${r2(cy - a)} ${r2(cx + a)},${r2(cy + a * 0.75)} ${r2(cx - a)},${r2(cy + a * 0.75)}` });
  }
  return svgEl('circle', { class: c, cx, cy, r });
}

export function roleIcon(role, size = 12) {
  return svgEl('svg', { class: 'role-icon', width: size, height: size, viewBox: `0 0 ${size} ${size}`, 'aria-hidden': 'true', focusable: 'false' },
    roleGlyph(role, size / 2, size / 2 + (role === 'disregulator' ? 0.8 : 0), size / 2 - 1.6));
}
