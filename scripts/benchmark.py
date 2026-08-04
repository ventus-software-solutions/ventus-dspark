#!/usr/bin/env python3
"""ventus-dspark throughput benchmark.

Measures prefill, decode and DSpark draft acceptance against a running
endpoint. Standard library only — it runs on the head node with no pip
install.

Two profiles, because there are two different questions:

  --profile compat   reproduce MiaAI-Lab's published conditions
                     (temperature 0.6, top_p 0.95, one trial, no warmup)
                     so our table is directly comparable to theirs.
  --profile strict   our default: temperature 0, three trials, median,
                     one discarded warmup per cell.

Two corpora, because repeated filler text is trivially predictable and
inflates DSpark acceptance well above real agent traffic:

  --corpus synthetic  repeated filler (what the upstream script uses)
  --corpus code       real source text read from --code-dir

The delta between the two is the acceptance-inflation measurement.

Usage:
    ./scripts/benchmark.py --output results/ventus-dspark-0731.json
"""

import argparse
import asyncio
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

PROFILES = {
    "compat": {"temperature": 0.6, "top_p": 0.95, "trials": 1, "warmup": 0},
    "strict": {"temperature": 0.0, "top_p": 1.0, "trials": 3, "warmup": 1},
}

# Match the cumulative counter and nothing else. vLLM also exposes a
# `_created` sibling (a unix timestamp) and a `_per_pos_total` breakdown for
# each of these; summing the family would add epoch-seconds to a token count
# and double-count the per-position rows. The metric prefix varies by
# version, so only the leading namespace is loose.
ACCEPTED_RE = re.compile(r"^[a-z_:]*spec_decode_num_accepted_tokens_total(\{|$)")
DRAFTED_RE = re.compile(r"^[a-z_:]*spec_decode_num_draft_tokens_total(\{|$)")

CODE_SUFFIXES = (".py", ".sh", ".yml", ".yaml", ".md", ".toml")


def post_json(url, body, timeout=3600):
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # A bare "HTTP Error 404" says nothing about which endpoint was
            # missing; 4xx will not fix itself on retry either.
            detail = error.read()[:400].decode(errors="replace")
            raise RuntimeError(f"POST {url} -> HTTP {error.code}: {detail}") from None
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def server_root(base_url):
    return base_url.removesuffix("/").removesuffix("/v1")


def count_tokens(base_url, model, text):
    return post_json(f"{server_root(base_url)}/tokenize", {"model": model, "prompt": text})["count"]


def scrape_spec_decode(base_url):
    """Return (accepted, drafted) cumulative counters, or None if unexposed."""
    try:
        with urllib.request.urlopen(f"{server_root(base_url)}/metrics", timeout=30) as response:
            body = response.read().decode()
    except urllib.error.URLError:
        return None
    return parse_spec_decode(body)


def parse_spec_decode(body):
    """Sum the spec-decode totals across label sets in a Prometheus exposition.

    Returns None rather than zeros when the counters are absent, so a missing
    metric reads as "unknown" in the report instead of "no acceptance".
    """
    accepted = drafted = None
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if ACCEPTED_RE.match(name):
            accepted = (accepted or 0.0) + number
        elif DRAFTED_RE.match(name):
            drafted = (drafted or 0.0) + number
    if accepted is None or drafted is None:
        return None
    return accepted, drafted


def acceptance_between(before, after):
    if not before or not after:
        return None
    drafted = after[1] - before[1]
    if drafted <= 0:
        return None
    return (after[0] - before[0]) / drafted


def load_code_corpus(code_dir):
    chunks = []
    for path in sorted(Path(code_dir).rglob("*")):
        if path.suffix not in CODE_SUFFIXES or not path.is_file():
            continue
        if any(part in {".git", "node_modules", "results"} for part in path.parts):
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    if not chunks:
        raise SystemExit(f"no source files under {code_dir} — use --corpus synthetic")
    return "\n\n".join(chunks)


def build_prompt(base_url, model, target, nonce, corpus, code_text):
    """Grow a prompt to >= target tokens, prefixed with a unique nonce.

    The nonce matters: the server runs with --enable-prefix-caching, so a
    repeated identical prompt would be served from cache and report a
    fictional prefill rate. Every trial gets its own.
    """
    head = f"unique request {nonce}\n"
    if corpus == "code":
        body = code_text
        while count_tokens(base_url, model, head + body) < target:
            body += "\n\n" + code_text
        # Trim back toward the target so cases stay comparable in size.
        while len(body) > 200 and count_tokens(base_url, model, head + body[: len(body) // 2]) >= target:
            body = body[: len(body) // 2]
        return head + body
    unit = "benchmark context datum "
    body = unit * max(1, target // 3)
    while True:
        total = count_tokens(base_url, model, head + body)
        if total >= target:
            return head + body
        body += unit * max(1, (target - total) // 3)


def stream_one(base_url, model, prompt, settings, max_tokens, thinking):
    """One streaming completion. Separates prefill from decode.

    ttft covers prompt processing; decode rate deliberately excludes the
    first token, so it measures steady-state generation:
        (tokens - 1) / (t_last - t_first)
    """
    instruction = "\n\nReturn exactly 128 numbered lowercase English words, then stop."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + instruction}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "chat_template_kwargs": (
            {"thinking": False}
            if thinking == "off"
            else {"thinking": True, "reasoning_effort": thinking}
        ),
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = last = None
    usage = None
    pieces = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            text = (delta.get("reasoning") or delta.get("reasoning_content") or "") + (
                delta.get("content") or ""
            )
            if text:
                now = time.perf_counter()
                first = first if first is not None else now
                last = now
                pieces.append(text)
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()

    output_tokens = (usage or {}).get("completion_tokens")
    if output_tokens is None:
        output_tokens = count_tokens(base_url, model, "".join(pieces))
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    ttft = (first if first is not None else finished) - started
    decode_window = (last - first) if (first is not None and last is not None and last > first) else None

    return {
        "ttft_s": ttft,
        "elapsed_s": finished - started,
        "prompt_tokens": prompt_tokens,
        "prefill_tok_s": prompt_tokens / ttft if ttft > 0 else None,
        "output_tokens": output_tokens,
        "decode_tok_s": (output_tokens - 1) / decode_window if decode_window else None,
        "t_first": first,
        "t_last": last,
    }


async def run_trial(base_url, model, target, concurrency, settings, corpus, code_text, max_tokens, thinking, tag):
    prompts = await asyncio.gather(
        *[
            asyncio.to_thread(
                build_prompt, base_url, model, target, f"{tag}-r{index}", corpus, code_text
            )
            for index in range(concurrency)
        ]
    )
    before = scrape_spec_decode(base_url)
    results = await asyncio.gather(
        *[
            asyncio.to_thread(
                stream_one, base_url, model, prompt, settings, max_tokens, thinking
            )
            for prompt in prompts
        ]
    )
    after = scrape_spec_decode(base_url)

    firsts = [item["t_first"] for item in results if item["t_first"] is not None]
    lasts = [item["t_last"] for item in results if item["t_last"] is not None]
    total = sum(item["output_tokens"] - 1 for item in results)
    window = (max(lasts) - min(firsts)) if firsts and lasts and max(lasts) > min(firsts) else None
    streams = [item["decode_tok_s"] for item in results if item["decode_tok_s"]]
    prefills = [item["prefill_tok_s"] for item in results if item["prefill_tok_s"]]

    return {
        "aggregate_decode_tok_s": total / window if window else None,
        "mean_stream_decode_tok_s": statistics.fmean(streams) if streams else None,
        "median_prefill_tok_s": statistics.median(prefills) if prefills else None,
        "median_ttft_s": statistics.median(item["ttft_s"] for item in results),
        "acceptance": acceptance_between(before, after),
        "requests": results,
    }


async def run_case(args, settings, code_text, target, concurrency):
    label = f"p{target}-c{concurrency}"
    for index in range(settings["warmup"]):
        print(f"  {label} warmup {index + 1}/{settings['warmup']}", flush=True)
        await run_trial(
            args.base_url, args.model, target, concurrency, settings, args.corpus,
            code_text, args.max_tokens, args.thinking, f"{label}-warm{index}",
        )

    trials = []
    for index in range(settings["trials"]):
        print(f"  {label} trial {index + 1}/{settings['trials']}", flush=True)
        trials.append(
            await run_trial(
                args.base_url, args.model, target, concurrency, settings, args.corpus,
                code_text, args.max_tokens, args.thinking, f"{label}-t{index}",
            )
        )

    def median_of(key):
        values = [trial[key] for trial in trials if trial[key] is not None]
        return statistics.median(values) if values else None

    return {
        "target_prompt_tokens": target,
        "concurrency": concurrency,
        "aggregate_decode_tok_s": median_of("aggregate_decode_tok_s"),
        "mean_stream_decode_tok_s": median_of("mean_stream_decode_tok_s"),
        "median_prefill_tok_s": median_of("median_prefill_tok_s"),
        "median_ttft_s": median_of("median_ttft_s"),
        "acceptance": median_of("acceptance"),
        "trials": trials,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="/models/v4-flash-0731")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="strict")
    parser.add_argument("--corpus", choices=("synthetic", "code"), default="synthetic")
    parser.add_argument("--code-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--prompt-lengths", default="256,2048,8192,32768,131072")
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking", choices=("off", "low", "high", "max"), default="off",
                        help="reasoning effort; 'off' keeps reasoning tokens out of the decode measurement")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = PROFILES[args.profile]
    code_text = load_code_corpus(args.code_dir) if args.corpus == "code" else ""

    if scrape_spec_decode(args.base_url) is None:
        print("warning: /metrics exposes no spec-decode counters — acceptance will be null", flush=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "profile": args.profile,
        "settings": settings,
        "corpus": args.corpus,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "cases": [],
    }

    for target in [int(value) for value in args.prompt_lengths.split(",")]:
        for concurrency in [int(value) for value in args.concurrency.split(",")]:
            case = await run_case(args, settings, code_text, target, concurrency)
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({k: v for k, v in case.items() if k != "trials"}, sort_keys=True), flush=True)

    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
