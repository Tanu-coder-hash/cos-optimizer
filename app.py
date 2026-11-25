"""CoS Optimizer - web interface.   Run:  streamlit run app.py"""
from __future__ import annotations

import glob
import json
import os
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cosopt import tasks, vocab
from cosopt.compiler import TranslationLayer
from cosopt.llm import make_llm
from cosopt.render import est_tokens, grade, render_block, render_prompt
from cosopt.schema import _report, load_instances

st.set_page_config(page_title="CoS Optimizer", page_icon="◈", layout="wide")

# Validated categorical palette (fixed order, never cycled)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COND = {"cot": ("CoT (natural language)", C[0]), "default_cos": ("CoS (hand-designed)", C[1]), "optimized": ("CoS (optimised)", C[2])}
ORIGINS = ["seed:cot", "seed:default-cos", "seed:random", "seed:user", "llm", "mutate", "crossover", "random"]
ORIGIN_COLOR = {"seed:cot": C[0], "seed:default-cos": C[1], "seed:random": "#8a8984", "seed:user": "#8a8984",
                "llm": C[2], "mutate": C[3], "crossover": C[4], "random": C[6]}
DEFAULT_MODEL = {"simulated": "offline-demo", "anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini", "openai_compatible": "llama3.1"}
LAYOUT = dict(template="plotly_white", font=dict(size=13, color="#0b0b0b"), margin=dict(l=10, r=10, t=40, b=10),
              plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)")

ss = st.session_state
ss.setdefault("user_seeds", [])

# ============================================================== sidebar: model
with st.sidebar:
    st.header("Model settings")
    backend = st.selectbox("Solver backend", list(DEFAULT_MODEL), format_func=lambda b: {
        "simulated": "Simulated (offline demo)", "anthropic": "Anthropic (Claude)", "openai": "OpenAI",
        "openai_compatible": "OpenAI-compatible / Ollama / vLLM"}[b])
    model = st.text_input("Solver model", DEFAULT_MODEL[backend], key=f"model_{backend}")
    api_key, base_url = None, None
    if backend in ("anthropic", "openai"):
        env = "ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY"
        api_key = st.text_input(f"API key (or set {env})", type="password") or None
    if backend == "openai_compatible":
        base_url = st.text_input("Base URL", "http://localhost:11434/v1")
        api_key = st.text_input("API key (if needed)", type="password") or None
    if backend != "simulated":
        meta_model = st.text_input("Judge + proposer model", model, help="A stronger model here usually helps. Same backend as the solver.")
        c1, c2 = st.columns(2)
        price_in = c1.number_input("$ / 1M in", 0.0, 100.0, 0.0, 0.1)
        price_out = c2.number_input("$ / 1M out", 0.0, 200.0, 0.0, 0.1)
    else:
        meta_model, price_in, price_out = None, 0.0, 0.0
    max_tokens = st.slider("Max output tokens", 256, 8192, 768 if backend == "openai_compatible" else 1024, 256, key=f"mt_{backend}")
    concurrency = st.slider("Parallel requests", 1, 16, 2 if backend == "openai_compatible" else 4, key=f"cc_{backend}",
                            help="Local Ollama usually answers one request at a time; 2 is plenty.")
    if backend != "simulated" and st.button("Test connection"):
        with st.spinner("Sending a test message..."):
            _c = make_llm(backend, model, api_key=api_key, base_url=base_url, max_tokens=20, cache_path=None, timeout=60).complete(
                "Reply with exactly: OK", "Say OK", retries=1)
        if _c.error:
            st.error(f"Failed: {_c.error}")
        else:
            st.success(f"Connected. Model replied: {_c.text.strip()[:60]}")
    if backend == "simulated":
        st.warning("Simulated mode calls no model. Use it to try the interface; its numbers are synthetic, not results.")


def get_solver():
    extra = {} if backend == "simulated" else {"timeout": 180}
    return make_llm(backend, model, api_key=api_key, base_url=base_url, temperature=0.0, max_tokens=max_tokens,
                    price_in=price_in, price_out=price_out, **extra)


def get_meta():
    if backend == "simulated":
        return None
    return make_llm(backend, meta_model, api_key=api_key, base_url=base_url, max_tokens=2000, price_in=price_in, price_out=price_out)


# ============================================================== helpers
def plot_instance(inst):
    d = inst.data
    if inst.domain == "grid":
        g = d["grid"]
        z = [[1 if ch == "#" else 0 for ch in row] for row in g]
        fig = go.Figure(go.Heatmap(z=z, colorscale=[[0, "#f0efec"], [1, "#52514e"]], showscale=False, xgap=2, ygap=2,
                                   hovertemplate="row %{y}, col %{x}<extra></extra>"))
        path = inst.answer["path"]
        fig.add_trace(go.Scatter(x=[p[1] for p in path], y=[p[0] for p in path], mode="lines", line=dict(color=C[0], width=2),
                                 name="shortest path", hoverinfo="skip"))
        for (r, c), lab, col in ((d["start"], "start", C[2]), (d["goal"], "goal", C[1])):
            fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers+text", text=[lab], textposition="top center",
                                     marker=dict(size=14, color=col, line=dict(color="#fcfcfb", width=2)), name=lab))
        fig.update_yaxes(autorange="reversed", showgrid=False, dtick=1)
        fig.update_xaxes(showgrid=False, dtick=1)
        fig.update_layout(**LAYOUT, height=340, showlegend=False, title="Instance (gray = wall)")
        return fig
    if inst.domain == "graph":
        import networkx as nx
        G = nx.DiGraph() if d.get("directed", True) else nx.Graph()
        G.add_nodes_from(d["nodes"])
        G.add_weighted_edges_from([tuple(e) for e in d["edges"]])
        pos = nx.spring_layout(G, seed=1)
        on_path = set(zip(inst.answer["path"], inst.answer["path"][1:]))
        fig = go.Figure()
        for u, v, w in d["edges"]:
            hot = (u, v) in on_path or (not d.get("directed", True) and (v, u) in on_path)
            fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
                                     line=dict(color=C[0] if hot else "#c3c2b7", width=3 if hot else 1.5), hoverinfo="skip"))
            fig.add_annotation(x=(pos[u][0] + pos[v][0]) / 2, y=(pos[u][1] + pos[v][1]) / 2, text=str(w), showarrow=False,
                               font=dict(size=11, color="#52514e"), bgcolor="#fcfcfb")
        col = [C[2] if n == d["source"] else C[1] if n == d["target"] else "#52514e" for n in d["nodes"]]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in d["nodes"]], y=[pos[n][1] for n in d["nodes"]], mode="markers+text",
                                 text=d["nodes"], textposition="middle center", textfont=dict(color="#fcfcfb", size=11),
                                 marker=dict(size=24, color=col, line=dict(color="#fcfcfb", width=2)), hovertemplate="%{text}<extra></extra>"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(**LAYOUT, height=340, showlegend=False,
                          title=f"Instance ({'directed' if d.get('directed', True) else 'undirected'}; green = source, orange = target)")
        return fig
    return None


def results_charts(r):
    # 1. test accuracy with CIs
    fig = go.Figure()
    for k in ("cot", "default_cos", "optimized"):
        s = r["test"][k]
        fig.add_trace(go.Bar(x=[COND[k][0]], y=[s["acc"]], marker=dict(color=COND[k][1], cornerradius=4), width=0.5,
                             error_y=dict(type="data", symmetric=False, array=[s["ci95"][1] - s["acc"]],
                                          arrayminus=[s["acc"] - s["ci95"][0]], color="#52514e", thickness=1.5),
                             text=[f"{100*s['acc']:.1f}%"], textposition="inside", insidetextanchor="start", textfont=dict(color="#fcfcfb"), name=COND[k][0],
                             hovertemplate=f"{COND[k][0]}<br>accuracy %{{y:.1%}}<br>95% CI {s['ci95'][0]:.1%} - {s['ci95'][1]:.1%}<br>tokens {s['tokens']:.0f}<extra></extra>"))
    fig.update_layout(**LAYOUT, height=360, showlegend=False, title="Held-out test accuracy (95% bootstrap CI)",
                      yaxis=dict(range=[0, 1.12], tickformat=".0%", gridcolor="#ebeae6"))
    # 2. accuracy vs tokens, all candidates
    cands = [c for c in r["all_candidates"]]
    fig2 = go.Figure()
    for o in ORIGINS:
        pts = [c for c in cands if c.get("origin") == o]
        if not pts:
            continue
        fig2.add_trace(go.Scatter(x=[p["tokens"] for p in pts], y=[p["acc"] for p in pts], mode="markers", name=o,
                                  marker=dict(size=10, color=ORIGIN_COLOR[o], line=dict(color="#fcfcfb", width=2),
                                              symbol=["circle" if p["n"] >= max(c["n"] for c in cands) else "circle-open" for p in pts]),
                                  text=[p["spec"]["name"] for p in pts],
                                  hovertemplate="%{text}<br>acc %{y:.1%}<br>tokens %{x:.0f}<extra>" + o + "</extra>"))
    b = r["best"]
    fig2.add_annotation(x=b["tokens"], y=b["acc"], text="best", showarrow=True, arrowhead=0, ay=-28, font=dict(color="#0b0b0b"))
    fig2.update_layout(**LAYOUT, height=360, title="Dev accuracy vs. tokens per call (open = pruned at screening)",
                       xaxis=dict(title="mean tokens (prompt + output)", gridcolor="#ebeae6"),
                       yaxis=dict(title="dev accuracy", tickformat=".0%", gridcolor="#ebeae6"), legend=dict(orientation="h", y=-0.25))
    # 3. best-so-far fitness by generation
    gens = sorted({c["generation"] for c in cands})
    best_so_far, cur = [], -1e9
    for g in gens:
        cur = max(cur, max(c["fitness"] for c in cands if c["generation"] == g))
        best_so_far.append(cur)
    fig3 = go.Figure(go.Scatter(x=gens, y=best_so_far, mode="lines+markers", line=dict(color=C[0], width=2),
                                marker=dict(size=9, line=dict(color="#fcfcfb", width=2)), name="best fitness",
                                hovertemplate="generation %{x}<br>best fitness %{y:.3f}<extra></extra>"))
    fig3.update_layout(**LAYOUT, height=300, title="Best fitness so far, by generation",
                       xaxis=dict(title="generation", dtick=1, gridcolor="#ebeae6"), yaxis=dict(gridcolor="#ebeae6"))
    return fig, fig2, fig3


def spec_editor(domain, prefix):
    mode = st.radio("Mode", ["cos", "nl"], horizontal=True, key=f"{prefix}_mode",
                    format_func=lambda m: "Chain-of-Symbol" if m == "cos" else "Chain-of-Thought (natural language)")
    if mode == "nl":
        return vocab.nl_baseline(domain)
    base = vocab.default_cos(domain)["params"]
    space = vocab.SPACES[domain]
    params = {}
    cols = st.columns(3)
    for i, (k, opts) in enumerate(space["categorical"].items()):
        params[k] = cols[i % 3].selectbox(k, opts, index=opts.index(base[k]), key=f"{prefix}_{domain}_{k}")
    st.caption("Token slots - type any 1-4 character symbol")
    cols = st.columns(4)
    for i, k in enumerate(space["tokens"]):
        params[k] = cols[i % 4].text_input(k, base[k], key=f"{prefix}_{domain}_{k}")
    name = st.text_input("Name", "my-vocab", key=f"{prefix}_name")
    return {"name": name, "domain": domain, "mode": "cos", "params": params}


# ============================================================== header
st.title("◈ CoS Optimizer")
st.caption("Automated search for Chain-of-Symbol vocabularies on spatial and graph reasoning, compared against Chain-of-Thought.")
tab_data, tab_lab, tab_opt, tab_res, tab_tr, tab_about = st.tabs(
    ["1 · Data", "2 · Vocabulary lab", "3 · Optimise", "4 · Results", "5 · Translate", "About"])

# ============================================================== 1. data
with tab_data:
    src = st.radio("Source", ["Built-in task generator", "Upload your own structures"], horizontal=True)
    if src.startswith("Built-in"):
        c = st.columns(5)
        domain = c[0].selectbox("Task family", tasks.DOMAINS, format_func=lambda d: {"grid": "Grid navigation", "graph": "Weighted shortest path", "bricks": "Brick World stacking"}[d])
        diff = c[1].selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        n_dev = c[2].number_input("Dev instances", 4, 500, 24)
        n_test = c[3].number_input("Test instances", 4, 2000, 60)
        seed = c[4].number_input("Seed", 0, 10**6, 0)
        if st.button("Generate dataset", type="primary"):
            dev = tasks.make_dataset(domain, int(n_dev), diff, seed=int(seed), prefix="dev-")
            test = tasks.make_dataset(domain, int(n_test), diff, seed=int(seed) + 1000, prefix="test-")
            ss.update(domain=domain, dev=dev, test=test, schema={**_report(domain, "built-in generator", dev[0].data), "difficulty": diff},
                      exemplar=tasks.make_exemplar(domain, dev[0]))
    else:
        st.markdown("Upload JSON / JSONL containing grids, graphs (edge list, node-link or adjacency) or brick stacks. "
                    "Missing queries (source/target, target brick) are generated; answers come from exact solvers.")
        up = st.file_uploader("Structure file", type=["json", "jsonl"])
        txt = st.text_area("...or paste JSON", height=120, placeholder='{"edges": [["A","B",3], ["B","C",1], ["A","C",7], ["C","D",2]], "directed": true}')
        c = st.columns(3)
        qps = c[0].number_input("Queries per structure", 1, 200, 20)
        dev_frac = c[1].slider("Dev fraction", 0.1, 0.9, 0.3)
        if st.button("Load", type="primary"):
            try:
                raw = up.read().decode("utf-8") if up else txt
                insts, schema = load_instances(raw, queries_per_structure=int(qps))
                random.Random(0).shuffle(insts)
                k = max(2, int(len(insts) * dev_frac))
                if len(insts) - k < 2:
                    raise ValueError(f"only {len(insts)} instances - add more structures or queries")
                ss.update(domain=schema["domain"], dev=insts[:k], test=insts[k:], schema=schema,
                          exemplar=tasks.make_exemplar(schema["domain"], insts[0]))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load: {e}")
    if "dev" in ss:
        st.success(f"{ss.domain}: {len(ss.dev)} dev / {len(ss.test)} test instances")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Inferred schema")
            st.json(ss.schema)
        with c2:
            idx = st.slider("Preview instance", 0, len(ss.dev) - 1, 0)
            inst = ss.dev[idx]
            fig = plot_instance(inst)
            if fig:
                st.plotly_chart(fig, width="stretch")
            a, b = st.columns(2)
            a.markdown("**Rendered as CoT (natural language)**")
            a.code(render_block(inst, vocab.nl_baseline(ss.domain)), language=None, wrap_lines=True)
            b.markdown("**Rendered as default CoS**")
            b.code(render_block(inst, vocab.default_cos(ss.domain)), language=None, wrap_lines=True)
            st.caption(f"Gold answer: {json.dumps(inst.answer if inst.domain != 'graph' else {'cost': inst.answer['cost'], 'path': inst.answer['path']})}")

# ============================================================== 2. lab
with tab_lab:
    if "dev" not in ss:
        st.info("Load a dataset in tab 1 first.")
    else:
        spec = spec_editor(ss.domain, "lab")
        ok, errs = vocab.validate(spec)
        if not ok:
            st.error("; ".join(errs))
        else:
            inst = ss.dev[0]
            system, user = render_prompt(inst, spec, ss.exemplar)
            st.markdown(f"**Full prompt for dev instance 0** · about {est_tokens(system + user)} tokens")
            st.code(user, language=None, wrap_lines=True)
            c1, c2, c3 = st.columns(3)
            n_try = c1.number_input("Instances to test", 1, len(ss.dev), min(3, len(ss.dev)))

            def show_row(row):
                gold = row["gold"]
                head = f"{'✅' if row['correct'] else '❌'} {row['id']} - model answered {row['parsed']} - correct answer {gold} - {row['secs']:.0f}s"
                with st.expander(head):
                    if row["error"]:
                        st.error(f"Model call failed: {row['error']}")
                    st.code(row["text"] or "(empty response)", language=None, wrap_lines=True)

            if c2.button("Test this vocabulary"):
                import time as _t
                solver = get_solver()
                rows = []
                status = st.status(f"Asking {solver.label} ... (local models can take 10-60 s per problem)", expanded=True)
                for i, it in enumerate(ss.dev[: int(n_try)]):
                    status.write(f"Problem {i + 1}/{int(n_try)}: waiting for the model...")
                    s, u = render_prompt(it, spec, ss.exemplar)
                    t0 = _t.time()
                    comp = solver.solve(s, u, ctx={"instance": it, "spec": spec})
                    g = grade(it, spec, comp.text)
                    row = {"id": it.id, "correct": g["correct"], "parsed": g["parsed"], "text": comp.text, "error": comp.error,
                           "secs": _t.time() - t0, "tokens": comp.in_tokens + comp.out_tokens,
                           "gold": it.answer.get("cost", it.answer.get("sequence", it.answer.get("moves")))}
                    rows.append(row)
                    status.write(f"Problem {i + 1}: {'✅ correct' if g['correct'] else '❌ wrong'}"
                                 + (f" - ERROR: {comp.error}" if comp.error else ""))
                    if comp.error and i == 0:
                        status.write("Stopping: the first call failed. Is Ollama running (`ollama serve`) and is the model name exact (`ollama list`)?")
                        break
                status.update(label="Done", state="complete", expanded=False)
                ss.lab_result = {"spec_name": spec["name"], "mode": spec["mode"], "rows": rows}
                timed = [r["secs"] for r in rows if r["secs"] > 0.5 and not r["error"]]
                if timed:
                    ss.sec_per_call = sum(timed) / len(timed)
            if ss.get("lab_result"):
                lr = ss.lab_result
                rows = lr["rows"]
                acc = sum(r["correct"] for r in rows) / max(1, len(rows))
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Accuracy ({'CoT' if lr['mode'] == 'nl' else 'CoS'}: {lr['spec_name']})", f"{acc:.0%}", help=f"{len(rows)} problems")
                m2.metric("Mean tokens / problem", f"{sum(r['tokens'] for r in rows) / max(1, len(rows)):.0f}")
                m3.metric("Mean seconds / problem", f"{sum(r['secs'] for r in rows) / max(1, len(rows)):.0f}")
                for r in rows:
                    show_row(r)
            if c3.button("Add to optimiser seeds"):
                ss.user_seeds.append(spec)
                st.toast(f"Added {spec['name']} ({len(ss.user_seeds)} user seeds)")
        if ss.user_seeds:
            st.caption("User seeds: " + ", ".join(s["name"] for s in ss.user_seeds))

# ============================================================== 3. optimise
PRESETS = {
    "Quick demo (fastest, for a live demo)": dict(gens=1, pop=3, kids=2, screen=4, jsamp=1, n_dev=6, n_test=12),
    "Small real run": dict(gens=1, pop=3, kids=2, screen=4, jsamp=1, n_dev=8, n_test=24),
    "Standard run": dict(gens=3, pop=5, kids=3, screen=6, jsamp=2, n_dev=16, n_test=40),
    "Full run (best with an API model)": dict(gens=4, pop=6, kids=4, screen=8, jsamp=3, n_dev=24, n_test=60),
    "Custom": None,
}


@st.cache_resource
def job_store():
    """Survives page reloads and clicks: the optimisation runs as a separate process."""
    return {}


def _expected_calls(p, n_seeds_extra):
    n_cand = p["pop"] + n_seeds_extra
    solver = n_cand * p["n_dev"] + p["gens"] * p["kids"] * (p["screen"] + 0.5 * (p["n_dev"] - p["screen"])) + 3 * p["n_test"]
    meta = (n_cand + p["gens"] * p["kids"]) * p["jsamp"] + p["gens"]
    return int(solver), int(meta)


def _tail(path, n=14):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


def start_job(p, prop_random):
    import subprocess
    import sys
    import time as _t
    app_dir = os.path.dirname(os.path.abspath(__file__))
    jdir = os.path.join(app_dir, "jobs", _t.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(jdir, exist_ok=True)
    dev, test = ss.dev[: p["n_dev"]], ss.test[: p["n_test"]]
    for name, data in (("dev.jsonl", dev), ("test.jsonl", test)):
        with open(os.path.join(jdir, name), "w", encoding="utf-8") as f:
            for i in data:
                f.write(json.dumps(i.to_dict(), ensure_ascii=False) + "\n")
    with open(os.path.join(jdir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(ss.schema, f)
    with open(os.path.join(jdir, "seeds.json"), "w", encoding="utf-8") as f:
        json.dump(ss.user_seeds, f, ensure_ascii=False)
    args = [sys.executable, "-u", "-m", "cosopt", "optimize",
            "--dev-file", os.path.join(jdir, "dev.jsonl"), "--test-file", os.path.join(jdir, "test.jsonl"),
            "--schema-file", os.path.join(jdir, "schema.json"), "--seeds-file", os.path.join(jdir, "seeds.json"),
            "--backend", backend, "--generations", str(p["gens"]), "--population", str(p["pop"]),
            "--children", str(p["kids"]), "--screen", str(min(p["screen"], len(dev))), "--judge-samples", str(p["jsamp"]),
            "--concurrency", str(concurrency), "--max-tokens", str(max_tokens), "--timeout", "240",
            "--price-in", str(price_in), "--price-out", str(price_out), "--out", os.path.join(app_dir, "runs")]
    if backend != "simulated":
        args += ["--model", model, "--meta-model", meta_model or model]
    if base_url:
        args += ["--base-url", base_url]
    if prop_random:
        args += ["--random-proposer"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    if api_key:
        env["ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY"] = api_key
    log_path = os.path.join(jdir, "log.txt")
    log = open(log_path, "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(args, cwd=app_dir, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags)
    solver_calls, meta_calls = _expected_calls(p, len(ss.user_seeds))
    job_store()["current"] = {"proc": proc, "log": log_path, "started": _t.time(), "expected": solver_calls,
                              "meta": meta_calls, "loaded": False, "label": f"{backend}:{model}", "domain": ss.domain}


@st.fragment(run_every=3)
def job_panel():
    import time as _t
    job = job_store().get("current")
    if not job:
        return
    proc = job["proc"]
    code = proc.poll()
    lines = _tail(job["log"], 400)
    run_dir = next((l.split("Run directory:", 1)[1].strip() for l in lines if l.startswith("Run directory:")), None)
    done = 0
    if run_dir and os.path.exists(os.path.join(run_dir, "calls.jsonl")):
        with open(os.path.join(run_dir, "calls.jsonl"), encoding="utf-8", errors="replace") as f:
            done = sum(1 for _ in f)
    elapsed = _t.time() - job["started"]
    job["expected"] = max(job["expected"], done)
    frac = min(0.99, done / max(1, job["expected"])) if code is None else 1.0
    st.markdown(f"**Background run** · {job['label']} · {job['domain']}")
    st.progress(frac, text=f"{done} of about {job['expected']} solver calls · {elapsed / 60:.1f} min elapsed")
    if code is None:
        if done >= 2:
            eta = (elapsed / done) * max(0, job["expected"] - done)
            st.caption(f"Roughly {eta / 60:.0f} more minutes (estimate improves as it runs). "
                       "You can switch tabs, use the Vocabulary lab or reload the page; the run keeps going.")
        else:
            st.caption("Starting... the first calls to a local model can take a minute while it loads.")
        if st.button("Stop run", key="stop_run"):
            proc.terminate()
            st.warning("Stopped.")
    elif code == 0:
        res_path = os.path.join(run_dir, "results.json") if run_dir else None
        if res_path and os.path.exists(res_path):
            if not job["loaded"]:
                with open(res_path, encoding="utf-8") as f:
                    ss.result = json.load(f)
                job["loaded"] = True
                st.rerun(scope="app")
            t = ss.get("result", {}).get("test", {})
            if t:
                st.success("Finished. Test accuracy: " + " · ".join(
                    f"{COND[k][0]} {t[k]['acc']:.0%}" for k in ("cot", "default_cos", "optimized") if k in t) +
                    ". Full charts are in the Results tab.")
        else:
            st.error("The run ended without results. Log below.")
    else:
        st.error(f"The run stopped (exit code {code}). Last lines of the log are below.")
    with st.expander("Live log", expanded=code not in (None, 0)):
        st.code("\n".join(lines[-14:]) or "(waiting for output)", language=None)
    if code is not None and st.button("Clear", key="clear_job"):
        job_store().pop("current", None)
        st.rerun(scope="app")


with tab_opt:
    if "dev" not in ss:
        st.info("Load a dataset in tab 1 first.")
    else:
        running = job_store().get("current") and job_store()["current"]["proc"].poll() is None
        preset_name = st.selectbox("Run size", list(PRESETS), index=0 if backend != "anthropic" else 1)
        p = PRESETS[preset_name]
        if p is None:
            c = st.columns(4)
            p = dict(gens=c[0].number_input("Generations", 1, 50, 2), pop=c[1].number_input("Seed population", 2, 30, 4),
                     kids=c[2].number_input("Proposals / generation", 1, 20, 3), screen=c[3].number_input("Screening subset", 2, 100, 4))
            c = st.columns(4)
            p.update(jsamp=c[0].number_input("Judged traces / candidate", 0, 20, 2),
                     n_dev=c[1].number_input("Dev problems used", 2, len(ss.dev), min(12, len(ss.dev))),
                     n_test=c[2].number_input("Test problems used", 2, len(ss.test), min(30, len(ss.test))))
        p = {**p, "n_dev": min(p["n_dev"], len(ss.dev)), "n_test": min(p["n_test"], len(ss.test))}
        prop = st.radio("Proposer", ["LLM proposer", "Random evolutionary (ablation)"], horizontal=True, disabled=backend == "simulated")
        solver_calls, meta_calls = _expected_calls(p, len(ss.user_seeds))
        spc = ss.get("sec_per_call") or {"simulated": 0.01, "openai_compatible": 20}.get(backend, 4)
        par = 1 if backend == "openai_compatible" else concurrency
        mins = (solver_calls + meta_calls) * spc / par / 60
        st.caption(f"{p['gens']} generation(s), {p['pop']} seed vocabularies + {p['kids']} AI proposals each generation, "
                   f"{p['n_dev']} dev / {p['n_test']} test problems. About {solver_calls + meta_calls} model calls, "
                   f"roughly {max(1, round(mins))} min at ~{spc:.0f} s per call"
                   + (" (measured in the Vocabulary lab)." if ss.get("sec_per_call") else ". Run a lab test first for a better estimate."))
        if st.button("Start optimisation in the background", type="primary", disabled=bool(running)):
            start_job(p, prop.startswith("Random"))
            st.rerun()
        if running:
            st.caption("A run is in progress. Wait for it to finish or stop it before starting another.")
        job_panel()

# ============================================================== 4. results
with tab_res:
    runs = sorted(glob.glob("runs/*/results.json"), reverse=True)
    if runs:
        pick = st.selectbox("Load a saved run", ["(current session)"] + runs)
        if pick != "(current session)":
            with open(pick, encoding="utf-8") as f:
                ss.result = json.load(f)
    r = ss.get("result")
    if not r:
        st.info("Run an optimisation (tab 3) or load a saved run.")
    else:
        if r.get("simulated"):
            st.warning("Simulated run: these numbers are synthetic and only show what a real report looks like.")
        cols = st.columns(4)
        for col, k in zip(cols, ("cot", "default_cos", "optimized")):
            s = r["test"][k]
            delta = None if k == "cot" else f"{100 * (s['acc'] - r['test']['cot']['acc']):+.1f} pts vs CoT"
            col.metric(COND[k][0], f"{s['acc']:.1%}", delta, help=f"95% CI {s['ci95'][0]:.1%}-{s['ci95'][1]:.1%}; {s['tokens']:.0f} tokens/call")
        p = r["significance"]["optimized_vs_cot"]["p_value"]
        cols[3].metric("Optimised vs CoT (McNemar p)", f"{p:.4f}", help="Exact paired test on the held-out test set")
        f1, f2, f3 = results_charts(r)
        a, b = st.columns(2)
        a.plotly_chart(f1, width="stretch")
        b.plotly_chart(f2, width="stretch")
        a, b = st.columns([1, 1])
        a.plotly_chart(f3, width="stretch")
        with b:
            st.markdown("**Significance (exact McNemar, paired)**")
            st.dataframe(pd.DataFrame([{"comparison": k.replace("_", " "), "A-only correct": v["a_only"],
                                        "B-only correct": v["b_only"], "p": round(v["p_value"], 4)} for k, v in r["significance"].items()]),
                         hide_index=True, width="stretch")
            st.markdown("**Optimised vocabulary**")
            st.json(r["best"]["spec"])
        st.subheader("Leaderboard (dev)")
        st.dataframe(pd.DataFrame([{"name": h["spec"]["name"], "origin": h.get("origin"), "generation": h.get("generation"),
                                    "dev acc": h["acc"], "judge": round(h["judge"], 3), "tokens": round(h["tokens"]),
                                    "fitness": round(h["fitness"], 3), "params": json.dumps(h["spec"].get("params", {}), ensure_ascii=False)}
                                   for h in r["leaderboard"]]), hide_index=True, width="stretch")
        if r["best"].get("failures"):
            st.subheader("Judge critiques on the best vocabulary")
            for f in r["best"]["failures"]:
                st.markdown(f"- {f.get('critique', '')} _{f.get('suggestion', '')}_")
        st.subheader("Prompt produced by the compiled layer")
        st.code(r["layer"]["preview"]["user"], language=None, wrap_lines=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("compiled_layer.json", json.dumps(r["layer"], indent=2, ensure_ascii=False), "compiled_layer.json")
        d2.download_button("results.json", json.dumps(r, indent=2, ensure_ascii=False, default=str), "results.json")
        rep = os.path.join(r["run_dir"], "report.md")
        if os.path.exists(rep):
            d3.download_button("report.md", open(rep, encoding="utf-8").read(), "report.md")
        st.caption(f"Saved in `{r['run_dir']}`. Usage: {r['usage']}")

# ============================================================== 5. translate
with tab_tr:
    st.markdown("Apply a compiled layer to new structural JSON to get ready-to-send symbolic prompts.")
    lf = st.file_uploader("Compiled layer (leave empty to use the current result)", type=["json"], key="layer_up")
    layer = None
    try:
        if lf:
            layer = TranslationLayer(json.loads(lf.read().decode("utf-8")))
        elif ss.get("result"):
            layer = TranslationLayer(ss.result["layer"])
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
    if layer:
        st.caption(f"Layer for **{layer.domain}**: {layer.spec['name']}")
        raw = st.text_area("Input JSON", height=150, key="tr_in")
        nq = st.number_input("Queries to generate (if none given)", 1, 50, 3)
        run_it = st.checkbox("Also send to the current solver model and grade")
        if st.button("Translate") and raw.strip():
            try:
                for s, u, inst in layer.translate(raw, queries=int(nq)):
                    with st.expander(f"{inst.id} · gold {inst.answer.get('cost', inst.answer.get('sequence', inst.answer.get('moves')))}", expanded=True):
                        st.code(u, language=None, wrap_lines=True)
                        if run_it:
                            comp = get_solver().solve(s, u, ctx={"instance": inst, "spec": layer.spec})
                            st.code(comp.text or comp.error or "", language=None, wrap_lines=True)
                            st.write(layer.grade(inst, comp.text))
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    else:
        st.info("No layer yet: run an optimisation or upload compiled_layer.json.")

# ============================================================== about
with tab_about:
    readme = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme):
        st.markdown(open(readme, encoding="utf-8").read())
