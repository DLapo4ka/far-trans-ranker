import csv, json, sys
from collections import defaultdict

d = sys.argv[1]
print("== metrics ==")
for name, v in json.load(open(f"{d}/metrics.json")).items():
    print(name, v)

rt = json.load(open(f"{d}/run_manifest.json"))["runtime"]
print("\ninvalid:", rt["raw_invalid_count"], "| parse repaired:", rt["parse_repair_count"],
      "| new calls:", rt.get("new_calls"), "| cache hits:", rt.get("cache_hits"),
      "| device:", rt.get("device"), rt.get("dtype"), "| include_profile:", rt.get("include_profile"))

secs = {}
for line in open(f"{d}/generations.jsonl"):
    r = json.loads(line)
    secs[r["cache_key"]] = r["response"]["seconds"]
print("LLM generation seconds (unique requests):", round(sum(secs.values()), 1))

rows = defaultdict(dict)
for row in csv.DictReader(open(f"{d}/predictions.csv")):
    rows[row["method"]].setdefault(row["user_id"], []).append((int(row["rank"]), row["ISIN"]))
lists = lambda m: {u: [i for _, i in sorted(v)] for u, v in rows[m].items()}
a, b = lists("llmrank"), lists("candidate_order_first10")
fb = set()
for line in open(f"{d}/final_responses.jsonl"):
    r = json.loads(line)
    if r["fallback"]:
        fb.add(str(r["user_id"]))
valid = [u for u in a if u not in fb]
print("\nvalid responses:", len(valid))
print("valid, identical to CBF order:", sum(a[u] == b[u] for u in valid))
print("valid, same 10 items reordered:", sum(a[u] != b[u] and set(a[u]) == set(b[u]) for u in valid))
print("valid, pulled items from positions 11-50:", sum(set(a[u]) != set(b[u]) for u in valid))
