#!/usr/bin/env python3
"""Fleet dashboard for ventus-dspark.

Stdlib only, like scripts/benchmark.py — no pip install, no build step.

It polls vLLM server-side and serves the result as JSON, rather than letting
the browser hit /metrics directly: vLLM sets no CORS headers, so a browser-side
fetch would be blocked, and this way the dashboard can be exposed on a
different port from the inference API.

Deliberately not Prometheus + Grafana. Two more services and a config surface,
for a two-node fleet, on a tool whose pitch is "one command, no .env editing".

Config (all optional):
  VLLM_URL     inference endpoint      (default http://127.0.0.1:8000)
  DASH_PORT    port to serve on        (default 8500)
  WORKER_HOST  worker IP, for display
  NCCL_IB_HCA  IB device, for display
  NCCL_IB_GID_INDEX
  MAX_NUM_SEQS
"""

import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
DASH_PORT = int(os.environ.get("DASH_PORT", "8500"))
CACHE_TTL = 2.0

_cache: dict = {"at": 0.0, "data": None}
# Previous raw counter sample, for deriving rates across polls.
_prev: dict = {"at": 0.0, "gen": None, "prompt": None}


def _rate(name: str, key: str, m: dict, now: float):
    """Tokens/s from a cumulative counter across two polls.

    vLLM 0.25 removed avg_*_throughput_toks_per_s; the counters are
    what remain. None until a second sample exists, and None when the
    fleet is idle — an honest blank beats a fabricated zero.
    """
    cur = m.get(name)
    prev, prev_at = _prev.get(key), _prev.get("at", 0.0)
    _prev[key] = cur
    _prev["at"] = now
    if cur is None or prev is None or now <= prev_at:
        return None
    rate = (cur - prev) / (now - prev_at)
    return round(rate, 1) if rate > 0.05 else None


def _get(path: str, timeout: float = 4.0):
    try:
        with urllib.request.urlopen(f"{VLLM_URL}{path}", timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def parse_metrics(text: str) -> dict:
    """Prometheus text format -> {name: float}, last sample wins.

    Keeps the bare metric name as the key, dropping any label set, which is
    enough here because vLLM exposes one model per server.
    """
    out: dict = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name, _, rest = line.partition(" ")
        if "{" in name:
            name = name[: name.index("{")]
        try:
            out[name] = float(rest.strip().split()[-1])
        except (ValueError, IndexError):
            continue
    return out


NODE_TELEMETRY_DIR = os.environ.get("NODE_TELEMETRY_DIR", "")
NODE_STALE_S = 60.0


def nodes():
    """Per-node hardware samples, written by the host-side collector.

    The container has neither ssh keys nor nvidia-smi — handing it either so
    it could read a temperature would be a bad trade — so the host writes
    key=value files and this only parses them. A sample older than
    NODE_STALE_S is reported stale rather than shown as current: silently
    serving a five-minute-old temperature is worse than admitting the gap.
    """
    if not NODE_TELEMETRY_DIR or not os.path.isdir(NODE_TELEMETRY_DIR):
        return []
    out = []
    now = time.time()
    for name in sorted(os.listdir(NODE_TELEMETRY_DIR)):
        path = os.path.join(NODE_TELEMETRY_DIR, name)
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        fields = {}
        for line in lines:
            key, sep, value = line.partition("=")
            if not sep:
                continue
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
        if not fields:
            continue
        fields["node"] = name
        fields["age_s"] = round(age, 1)
        fields["stale"] = age > NODE_STALE_S
        out.append(fields)
    return out


def acceptance(m: dict):
    """Draft acceptance alpha.

    Prefer the gauge; fall back to the counters. The metric names moved
    between vLLM releases and both engine lanes are supported, so try each
    rather than assuming one.
    """
    for key in ("vllm:spec_decode_draft_acceptance_rate",
                "vllm:spec_decode_efficiency"):
        if key in m:
            return m[key]
    acc = m.get("vllm:spec_decode_num_accepted_tokens_total")
    draft = m.get("vllm:spec_decode_num_draft_tokens_total")
    if acc is not None and draft:
        return acc / draft
    return None


def collect() -> dict:
    now = time.time()
    if _cache["data"] is not None and now - _cache["at"] < CACHE_TTL:
        return _cache["data"]

    health = _get("/health") is not None
    metrics_txt = _get("/metrics") or ""
    m = parse_metrics(metrics_txt)

    model = None
    ctx = None
    models_raw = _get("/v1/models")
    if models_raw:
        try:
            d = json.loads(models_raw)["data"][0]
            model = d.get("id")
            ctx = d.get("max_model_len")
        except (ValueError, KeyError, IndexError):
            pass

    data = {
        "ok": health,
        "model": model,
        "ctx": ctx,
        "worker": os.environ.get("WORKER_HOST"),
        "ib": {
            "hca": os.environ.get("NCCL_IB_HCA"),
            "gid": os.environ.get("NCCL_IB_GID_INDEX"),
        },
        "kv_used": m.get("vllm:kv_cache_usage_perc"),
        "running": m.get("vllm:num_requests_running"),
        "waiting": m.get("vllm:num_requests_waiting"),
        "max_seqs": os.environ.get("MAX_NUM_SEQS"),
        "decode": _rate("vllm:generation_tokens_total", "gen", m, now),
        "prefill": _rate("vllm:prompt_tokens_total", "prompt", m, now),
        "alpha": acceptance(m),
        "nodes": nodes(),
        "at": now,
    }
    _cache.update(at=now, data=data)
    return data


PAGE = """<!doctype html><meta charset=utf-8>
<title>ventus-dspark — fleet</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0b1220;color:#e2e8f0;
      font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 main{max-width:760px;margin:0 auto;padding:32px 20px}
 h1{font-size:15px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
    color:#94a3b8;margin:0 0 20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
 .tile{background:#111a2e;border:1px solid #27354e;border-radius:10px;padding:14px 16px}
 .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#64748b}
 .v{font-size:22px;margin-top:4px}
 .sub{font-size:12px;color:#94a3b8;margin-top:2px}
 .wide{grid-column:1/-1}
 .ok{color:#34d399}.bad{color:#f87171}.warn{color:#fbbf24}.dim{color:#64748b}
 footer{margin-top:20px;font-size:12px;color:#64748b}
</style>
<main>
 <h1>ventus-dspark — fleet</h1>
 <div class=grid id=g></div>
 <div class=grid id=n style="margin-top:12px"></div>
 <footer id=f>connecting…</footer>
</main>
<script>
const el = (h) => { const d=document.createElement('div'); d.innerHTML=h; return d.firstChild };
const num = (v,d=0) => v==null ? '–' : Number(v).toFixed(d);

function tile(k, v, sub, cls) {
  return `<div class="tile"><div class="k">${k}</div>
          <div class="v ${cls||''}">${v}</div>
          ${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
}

async function tick() {
  let d;
  try { d = await (await fetch('api/status')).json(); }
  catch { document.getElementById('f').textContent = 'dashboard cannot reach the API'; return; }

  const kv = d.kv_used == null ? null : d.kv_used * 100;
  const a  = d.alpha;
  // Acceptance sits next to decode on purpose: decode tracks alpha almost
  // linearly on this hardware, so a slow number is usually an unpredictable
  // prompt rather than a broken fleet.
  const aCls = a == null ? 'dim' : a < 0.45 ? 'warn' : 'ok';

  document.getElementById('g').innerHTML = [
    tile('nodes', d.ok ? 'head ✓ worker ✓' : 'head ✗',
         d.worker ? `worker ${d.worker}` : '', d.ok ? 'ok' : 'bad'),
    tile('InfiniBand', d.ib.hca || '–',
         d.ib.gid ? `gid ${d.ib.gid} · RoCEv2` : 'not probed'),
    tile('model', (d.model || '–').replace(/^\\/models\\//, ''),
         d.ctx ? `ctx ${d.ctx.toLocaleString()}` : ''),
    tile('KV pool', kv == null ? '–' : num(kv,1) + '%', 'of allocated pool'),
    tile('streams', `${num(d.running)} / ${d.max_seqs || '?'}`,
         `${num(d.waiting)} queued`),
    tile('decode', d.decode == null ? '–' : num(d.decode,1) + ' tok/s',
         a == null ? 'no acceptance metric' : `α ${a.toFixed(2)}`, aCls),
  ].join('');

  // Per-node hardware. Rendered as its own row because it answers a different
  // question from the fleet tiles above: not "is it serving" but "is the box
  // healthy" — the question nobody asks until a node wedges.
  const nodes = d.nodes || [];
  document.getElementById('n').innerHTML = nodes.length === 0 ? '' : nodes.map(n => {
    const memCls  = n.mem_used_pct > 92 ? 'bad' : n.mem_used_pct > 85 ? 'warn' : 'ok';
    const tempCls = n.gpu_temp_c  > 85 ? 'bad' : n.gpu_temp_c  > 75 ? 'warn' : 'ok';
    // A stale sample is dimmed rather than hidden: an unreachable node is
    // itself the news, and blanking it would look like a healthy idle box.
    const stale = n.stale ? ` · <span class=bad>stale ${num(n.age_s)}s</span>` : '';
    return `<div class="tile wide" ${n.stale ? 'style=opacity:.55' : ''}>
      <div class="k">${n.node}${stale}</div>
      <div class="sub" style="font-size:13px;margin-top:6px">
        <span class="${tempCls}">${num(n.gpu_temp_c)}°C</span> &nbsp;
        gpu ${num(n.gpu_util_pct)}% &nbsp;
        ${num(n.gpu_power_w,1)}W &nbsp;
        mem <span class="${memCls}">${num(n.mem_used_pct,1)}%</span>
        <span class=dim>of ${num(n.mem_total_gib)}G</span> &nbsp;
        load ${num(n.load1,2)} &nbsp; disk ${num(n.disk_used_pct)}% &nbsp;
        up ${num(n.uptime_h)}h &nbsp;
        nvrm <span class="${n.nvrm_oom > 0 ? 'warn' : 'dim'}">${num(n.nvrm_oom)}</span>
      </div></div>`;
  }).join('');

  document.getElementById('f').textContent =
    (a != null && a < 0.45 ? 'low α — unpredictable prompts, not a fleet fault · ' : '') +
    'updated ' + new Date().toLocaleTimeString();
}
tick(); setInterval(tick, 5000);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif self.path.rstrip("/").endswith("api/status"):
            # Viewer heartbeat. The host-side collector reads this and only
            # probes nodes while somebody is actually looking, so a dashboard
            # nobody has open costs nothing at all.
            if NODE_TELEMETRY_DIR:
                try:
                    with open(os.path.join(NODE_TELEMETRY_DIR, ".watch"), "w") as fh:
                        fh.write(str(time.time()))
                except OSError:
                    pass
            body = json.dumps(collect()).encode()
            ctype = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet; the fleet log is the interesting one
        pass


if __name__ == "__main__":
    print(f"[ventus-dspark] dashboard on :{DASH_PORT} -> {VLLM_URL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", DASH_PORT), Handler).serve_forever()
