#!/bin/bash
# Independent post-publish check for unrulyabstractions/apart-swarm. Retargeted 2026-11-07: the earlier
# arena-capstone baseline was the wrong repo and is discarded. Verifies the ABSOLUTE final state per the
# coordinator (moved-vs-baseline dropped: pre-final revision not retrievable). Prints actual next to
# expected for every item. Exit 1 on any FAIL. Run ONLY after the publisher declares DONE.
set -u
export HERE=$(cd "$(dirname "$0")" && pwd)
export REPO=unrulyabstractions/apart-swarm
export EXPECT_N=37368 EXPECT_HEAD=1830ebb6 EXPECT_ROWS=4466 EXPECT_CANON=4157
fail=0
echo "== 1. repo state ($REPO) =="
hf datasets info "$REPO" > "$HERE/.v_info.json" 2>"$HERE/.v_info.err" || { echo "  FAIL cannot read $REPO: $(head -c 300 "$HERE/.v_info.err")"; exit 1; }
python3 - <<'PY' || fail=1
import json, os
d=json.load(open(os.environ['HERE']+'/.v_info.json')); sib=d.get('siblings') or []
sha=d.get('sha') or ''; mod=str(d.get('lastModified') or d.get('last_modified') or ''); n=len(sib)
EN=int(os.environ['EXPECT_N']); EH=os.environ['EXPECT_HEAD']
print(f"  repo        : {d.get('id')}  private={d.get('private')}")
print(f"  HEAD sha    : actual={sha}  expected={EH}* or later")
print(f"  lastModified: actual={mod}  expected=2026-11-07*")
print(f"  n_files     : actual={n}  expected={EN}")
ok_n = n==EN; ok_d = mod.startswith('2026-11-07')
print('  PASS n_files exact' if ok_n else '  FAIL n_files'); print('  PASS modified today' if ok_d else '  FAIL lastModified not today')
open(os.environ['HERE']+'/.v_head','w').write(sha); open(os.environ['HERE']+'/.v_mod','w').write(mod)
raise SystemExit(0 if (ok_n and ok_d) else 1)
PY
HEAD=$(cat "$HERE/.v_head"); HEADMOD=$(cat "$HERE/.v_mod")
if [[ "$HEAD" == $EXPECT_HEAD* ]]; then echo "  PASS HEAD is $EXPECT_HEAD"; else
  hf datasets info "$REPO" --revision "$EXPECT_HEAD" > "$HERE/.v_ref.json" 2>"$HERE/.v_ref.err"
  REFMOD=$(python3 -c "import json;d=json.load(open('$HERE/.v_ref.json'));print(str(d.get('lastModified') or d.get('last_modified') or ''))" 2>/dev/null)
  REFSHA=$(python3 -c "import json;print(json.load(open('$HERE/.v_ref.json')).get('sha',''))" 2>/dev/null)
  [ -z "$REFMOD" ] && echo "  (ref revision lookup returned no date; raw: $(head -c 160 "$HERE/.v_ref.json") err: $(head -c 160 "$HERE/.v_ref.err"))"
  if [ -n "$REFMOD" ] && [[ "$HEADMOD" > "$REFMOD" || "$HEADMOD" == "$REFMOD" ]]; then echo "  PASS HEAD $HEAD is at/after $EXPECT_HEAD (ref $REFSHA @ $REFMOD <= head @ $HEADMOD)"
  else echo "  FAIL HEAD $HEAD is not $EXPECT_HEAD and not shown later (ref lastModified='$REFMOD')"; fail=1; fi
fi
echo "== 2. live README =="
OUT="$HERE/hf_verify_apart"; rm -rf "$OUT"
hf download "$REPO" --type dataset --include README.md --local-dir "$OUT" --quiet >/dev/null 2>&1 || { echo "  FAIL README download"; fail=1; }
hf download "$REPO" --type dataset --include manifest.jsonl --local-dir "$OUT" --quiet >/dev/null 2>&1 || { echo "  FAIL manifest download"; fail=1; }
R="$OUT/README.md"
chk(){ m=$(grep -oE "$1" "$R" 2>/dev/null | head -1); if [ -n "$m" ]; then echo "  PASS $2  (found: '$m')"; else echo "  FAIL $2  (pattern: $1)"; fail=1; fi; }
chk '192\.85'                          'total spend $192.85'
chk '136\.78'                          'valid $136.78'
chk '56\.07'                           'overhead $56.07'
chk '4,?157'                           'N = 4,157'
chk 'n ?= ?11|8 cheat|3 solo'          'qwen-chess n=11 (3 solo / 8 cheat)'
chk 'Status: complete \(final set\)'   'Status: complete (final set)'
chk '262,?144|262k'                    '262k-context limitation'
echo "== 3. manifest.jsonl =="
export M="$OUT/manifest.jsonl"
if [ -f "$M" ]; then python3 - <<'PY' || fail=1
import json, os
rows=[json.loads(l) for l in open(os.environ['M']) if l.strip()]
ER=int(os.environ['EXPECT_ROWS']); EC=int(os.environ['EXPECT_CANON'])
n=len(rows); keys=list(rows[0].keys()) if rows else []
rk=next((k for k in keys if k in ('run_name','name','run')), None) or next((k for k in keys if 'run' in k.lower() and 'name' in k.lower()), None)
uniq=len({r.get(rk) for r in rows}) if rk else -1
print(f"  rows        : actual={n}  expected={ER}"); print(f"  canonical   : actual(unique '{rk}')={uniq}  expected={EC}"); print(f"  fields      : {keys[:14]}")
ok = n==ER and uniq==EC
print('  PASS manifest rows + canonical' if ok else '  FAIL manifest rows/canonical'); raise SystemExit(0 if ok else 1)
PY
else echo "  FAIL manifest.jsonl missing"; fail=1; fi
echo "== result =="; [ $fail -eq 0 ] && echo "PUBLISH_VERIFIED repo=$REPO sha=$HEAD" || echo "PUBLISH_CHECK_FAILED repo=$REPO sha=$HEAD"
exit $fail
