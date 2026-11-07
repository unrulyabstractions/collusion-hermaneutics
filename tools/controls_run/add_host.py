#!/usr/bin/env python3
"""Fold a ready run host into the live orchestrator WITHOUT a restart.
Usage: python3 add_host.py <hostkey> [swe_cap]   e.g. python3 add_host.py cake 3
Adds {ssh: arena9-<hostkey>, conc_total:7, swe_cap:<n>, min_free_mb:3000} to orch_config.json
and removes <hostkey> from the key pool (a run host must not lend its key).
"""
import json, sys
from pathlib import Path
CFG = Path(__file__).resolve().parent / "orch_config.json"
host = sys.argv[1]
swe = int(sys.argv[2]) if len(sys.argv) > 2 else 3
c = json.loads(CFG.read_text())
c["hosts"][host] = {"ssh": f"arena9-{host}", "conc_total": 7, "swe_cap": swe, "min_free_mb": 3000}
c["key_hosts"] = [kh for kh in c["key_hosts"] if kh[0] != host]
CFG.write_text(json.dumps(c, indent=2))
print(f"added host {host} (swe_cap={swe}); hosts now {sorted(c['hosts'])}; keys {len(c['key_hosts'])}")
