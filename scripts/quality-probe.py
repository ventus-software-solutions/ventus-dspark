#!/usr/bin/env python3
"""Temp-0 quality probes for A/B gating.

Captures deterministic outputs on fixed prompts. Run before and after a
config change; byte-identical outputs prove the greedy path is untouched.

    ./scripts/quality-probe.py --base-url http://HEAD:8000/v1 --out before.json
    ./scripts/quality-probe.py --base-url http://HEAD:8000/v1 --diff before.json
"""
import argparse, json, re, sys, urllib.request

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

def require_quiet(base_url):
    """Refuse to probe while other requests share the batch — co-batching
    legally perturbs temp-0 outputs (near-tie flips), which is exactly the
    noise this gate must exclude."""
    root = base_url.removesuffix("/v1")
    with urllib.request.urlopen(root + "/metrics", timeout=30) as r:
        body = r.read().decode()
    for line in body.splitlines():
        m = re.match(r"[a-z_:]*num_requests_(running|waiting)\{[^}]*\} (\S+)", line)
        if m and float(m.group(2)) > 0:
            sys.exit(f"endpoint not quiet ({m.group(1)}={m.group(2)}) — pause clients first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="/models/v4-flash-0731")
    ap.add_argument("--out")
    ap.add_argument("--diff")
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    require_quiet(a.base_url)

    results = {}
    for name, prompt in PROBES.items():
        seen = []
        for _ in range(a.samples):
            out = post(a.base_url + "/chat/completions", {
                "model": a.model, "temperature": 0, "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"thinking": False},
            })
            seen.append(out["choices"][0]["message"]["content"])
        results[name] = sorted(set(seen))
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
    results["tool"] = [json.dumps(m.get("tool_calls", [{}])[0].get("function", {}), sort_keys=True)]

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"captured {len(results)} probes -> {a.out}")
    if a.diff:
        before = json.load(open(a.diff))
        bad, info = [], []
        for name, baseline in before.items():
            now = results.get(name, [])
            if len(baseline) > 1:
                # Sampling at baseline already produced several variants: this
                # prompt sits on a floating-point near-tie and legally flips
                # with batch shape at any config. Report, never gate.
                info.append(f"{name}: tie-sensitive ({len(baseline)} variants at baseline)")
            elif now != baseline:
                bad.append(name)
        for name in bad:
            print(f"DIFF in {name}:\n  before: {before[name][0][:110]!r}\n  after:  {(results.get(name) or [''])[0][:110]!r}")
        for note in info:
            print("info:", note)
        stable = len(before) - len(info)
        print("QUALITY GATE:",
              f"PASS — {stable} stable probes identical" if not bad
              else f"FAIL — {len(bad)} of {stable} stable probes differ")
        sys.exit(1 if bad else 0)

main()
