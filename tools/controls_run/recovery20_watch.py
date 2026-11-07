import json, time, sys
import orchestrator_v2 as o
cfg=json.load(open('orch_config.json')); hosts=cfg['hosts']
aliases={h:hc['ssh'] for h,hc in hosts.items()}
PAIRS=[tuple(p) for p in json.load(open('recovery20_pairs.json'))]
DEADLINE=time.time()+95*60  # hard stop 95 min
def status(pr):
    host,name=pr; a=aliases[host]
    # terminal if MARKER_DONE in log; else check process alive
    rc,out,_=o.ssh_out(a, f"grep -h 'MARKER_DONE' {o.LOGDIR}/{name}.log 2>/dev/null | tail -1; echo ---; pgrep -fc -- '--name {name} ' 2>/dev/null || echo 0", timeout=30)
    parts=out.split('---'); marker=parts[0].strip(); nproc=(parts[1].strip().splitlines() or ['0'])[-1] if len(parts)>1 else '0'
    if marker:
        d=dict(p.split('=',1) for p in marker.split() if '=' in p)
        return ('DONE', d.get('rc','?'), d.get('wall_s','?'))
    if nproc in ('0',''):
        return ('DIED','','')  # no marker + no process
    return ('RUNNING','','')
while True:
    res=o.pmap(status, PAIRS, workers=20)
    done=[p for p in PAIRS if res[p][0]=='DONE']
    died=[p for p in PAIRS if res[p][0]=='DIED']
    running=[p for p in PAIRS if res[p][0]=='RUNNING']
    print(f"[{time.strftime('%H:%M:%S')}] done={len(done)} died={len(died)} running={len(running)}", flush=True)
    if len(done)+len(died)>=20:
        print("ALL TERMINAL. done rcs:", {n:res[(h,n)][1:] for h,n in done}, flush=True)
        print("DIED:", [n for h,n in died], flush=True)
        print("RECOVERY20_COMPLETE", flush=True)
        break
    if time.time()>DEADLINE:
        print("WATCH DEADLINE HIT. still running:", [n for h,n in running], flush=True)
        print("RECOVERY20_TIMEOUT", flush=True)
        break
    time.sleep(180)
