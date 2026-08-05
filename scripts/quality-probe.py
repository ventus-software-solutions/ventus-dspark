#!/usr/bin/env python3
"""Temp-0 quality probes for A/B gating.

Captures deterministic outputs on fixed prompts. Run before and after a
config change; byte-identical outputs prove the greedy path is untouched.

    ./scripts/quality-probe.py --base-url http://HEAD:8000/v1 --out before.json
    ./scripts/quality-probe.py --base-url http://HEAD:8000/v1 --diff before.json
"""
import argparse, json, sys, urllib.request

PROBES = {
    "math": "Compute 847*39 - 1200/8. Show only the final number.",
    "code": "Write a Python function `merge(a, b)` merging two sorted lists. Code only.",
    "prose": "Explain in exactly two sentences why the sky is blue.",
    "json": 'Return a JSON object {"name": str, "primes": [first 5 primes]} for name "test". JSON only.',
    "long": "List the first 20 square numbers as 'n: n^2', one per line.",
}

def post(url, body):
    r = urllib.request.Request(url, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.load(resp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="/models/v4-flash-0731")
    ap.add_argument("--out")
    ap.add_argument("--diff")
    a = ap.parse_args()

    results = {}
    for name, prompt in PROBES.items():
        out = post(a.base_url + "/chat/completions", {
            "model": a.model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"thinking": False},
        })
        results[name] = out["choices"][0]["message"]["content"]
    # tool-call round trip
    tool = post(a.base_url + "/chat/completions", {
        "model": a.model, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": "Weather in Berlin?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                  "required": ["city"]}}}],
        "chat_template_kwargs": {"thinking": False},
    })
    m = tool["choices"][0]["message"]
    results["tool"] = json.dumps(m.get("tool_calls", [{}])[0].get("function", {}), sort_keys=True)

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"captured {len(results)} probes -> {a.out}")
    if a.diff:
        before = json.load(open(a.diff))
        bad = [k for k in before if before[k] != results.get(k)]
        for k in bad:
            print(f"DIFF in {k}:\n  before: {before[k][:120]!r}\n  after:  {results.get(k,'')[:120]!r}")
        print("QUALITY GATE:", "PASS — outputs byte-identical" if not bad else f"FAIL — {len(bad)} probes differ")
        sys.exit(1 if bad else 0)

main()
