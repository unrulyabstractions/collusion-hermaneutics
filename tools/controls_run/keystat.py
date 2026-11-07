import json, sys, urllib.request
k = sys.stdin.readline().strip()
print("key prefix ok:", k.startswith("sk-or-"), "len", len(k))
r = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {k}"})
d = json.loads(urllib.request.urlopen(r, timeout=30).read())["data"]
print({x: d.get(x) for x in ("label", "limit", "usage", "limit_remaining")})
