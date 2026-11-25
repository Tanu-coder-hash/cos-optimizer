"""Command-line entry point.

Examples
--------
python -m cosopt optimize --domain grid --backend simulated
python -m cosopt optimize --domain graph --backend anthropic --model claude-sonnet-5-5 --generations 5
python -m cosopt optimize --data my_graph.json --backend ollama --model llama3.1
python -m cosopt translate --layer runs/<run>/compiled_layer.json --input my_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys

from . import tasks
from .judge import HeuristicJudge, LLMJudge
from .llm import make_llm
from .optimizer import OptConfig, Optimizer
from .proposer import LLMProposer, RandomProposer
from .schema import load_instances


def _read_instances(path):
    from .tasks import Instance
    with open(path, encoding="utf-8") as f:
        return [Instance.from_dict(json.loads(l)) for l in f if l.strip()]


def build_split(domain, data_path, n_dev, n_test, difficulty, seed, dev_file=None, test_file=None):
    if dev_file and test_file:
        dev, test = _read_instances(dev_file), _read_instances(test_file)
        from .schema import _report
        return dev[0].domain, dev, test, _report(dev[0].domain, "saved split", dev[0].data)
    if data_path:
        with open(data_path, encoding="utf-8") as f:
            insts, schema = load_instances(f.read(), queries_per_structure=max(10, (n_dev + n_test)), seed=seed)
        domain = schema["domain"]
        if len(insts) < 4:
            raise SystemExit("need at least 4 instances from the uploaded data")
        k = max(2, int(len(insts) * n_dev / (n_dev + n_test)))
        return domain, insts[:k], insts[k:], schema
    dev = tasks.make_dataset(domain, n_dev, difficulty, seed=seed, prefix="dev-")
    test = tasks.make_dataset(domain, n_test, difficulty, seed=seed + 1000, prefix="test-")
    from .schema import _report
    schema = {**_report(domain, "built-in generator", dev[0].data), "difficulty": difficulty}
    return domain, dev, test, schema


def cmd_optimize(a):
    domain, dev, test, schema = build_split(a.domain, a.data, a.dev, a.test, a.difficulty, a.seed, a.dev_file, a.test_file)
    if a.schema_file:
        with open(a.schema_file, encoding="utf-8") as f:
            schema = json.load(f)
    kw = dict(temperature=0.0, max_tokens=a.max_tokens, price_in=a.price_in, price_out=a.price_out)
    if a.backend != "simulated":
        kw["timeout"] = a.timeout
    solver = make_llm(a.backend, a.model, base_url=a.base_url, **kw)
    if solver.is_simulated:
        print("NOTE: simulated backend - numbers are synthetic, for pipeline testing only.", file=sys.stderr)
        judge, proposer = HeuristicJudge(), RandomProposer(a.seed)
    else:
        meta = make_llm(a.meta_backend or a.backend, a.meta_model or a.model, base_url=a.base_url, max_tokens=2000, timeout=a.timeout)
        judge = LLMJudge(meta) if a.judge_samples else None
        proposer = LLMProposer(meta, seed=a.seed) if not a.random_proposer else RandomProposer(a.seed)
    cfg = OptConfig(domain=domain, generations=a.generations, population=a.population, children=a.children,
                    screen_size=min(a.screen, len(dev)), judge_samples=a.judge_samples, concurrency=a.concurrency, seed=a.seed,
                    w_judge=a.w_judge, w_tokens=a.w_tokens)
    seeds = []
    if a.seeds_file:
        with open(a.seeds_file, encoding="utf-8") as f:
            seeds = json.load(f)
    exemplar = tasks.make_exemplar(domain, dev[0])

    def cb(kind, *args):
        if kind == "log":
            print(args[0], flush=True)
        elif kind == "candidate":
            r = args[0]
            print(f"  [{r['generation']}] {r['spec']['name']:<24} acc={r['acc']:.3f} judge={r['judge']:.2f} "
                  f"tok={r['tokens']:.0f} fit={r['fitness']:.3f} n={r['n']}", flush=True)
        elif kind == "final":
            r = args[0]
            for k, s in r["test"].items():
                print(f"TEST {k:<12} acc={s['acc']:.3f} CI95=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] tokens={s['tokens']:.0f}")
            print(f"Report: {r['run_dir']}/report.md", flush=True)

    Optimizer(solver, judge, proposer, dev, test, exemplar, schema, cfg, out_dir=a.out, callback=cb, extra_seeds=seeds).run()


def cmd_translate(a):
    from .compiler import TranslationLayer
    layer = TranslationLayer.load(a.layer)
    with open(a.input, encoding="utf-8") as f:
        raw = json.load(f)
    for s, u, inst in layer.translate(raw, queries=a.queries):
        print("=" * 80)
        print("[system]", s)
        print(u)
        print(f"[gold answer] {inst.answer}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cosopt")
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("optimize")
    o.add_argument("--domain", default="grid", choices=tasks.DOMAINS)
    o.add_argument("--data", help="JSON/JSONL structure file (overrides --domain)")
    o.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    o.add_argument("--dev", type=int, default=24)
    o.add_argument("--test", type=int, default=60)
    o.add_argument("--backend", default="simulated", choices=["simulated", "anthropic", "openai", "openai_compatible", "ollama"])
    o.add_argument("--model")
    o.add_argument("--base-url")
    o.add_argument("--meta-backend", help="backend for judge+proposer (default: same as solver)")
    o.add_argument("--meta-model")
    o.add_argument("--generations", type=int, default=4)
    o.add_argument("--population", type=int, default=6)
    o.add_argument("--children", type=int, default=4)
    o.add_argument("--screen", type=int, default=8)
    o.add_argument("--judge-samples", type=int, default=3)
    o.add_argument("--random-proposer", action="store_true", help="ablation: evolutionary search without the LLM proposer")
    o.add_argument("--concurrency", type=int, default=4)
    o.add_argument("--max-tokens", type=int, default=1024)
    o.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens (cost tracking)")
    o.add_argument("--price-out", type=float, default=0.0)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--out", default="runs")
    o.add_argument("--dev-file", help="JSONL of saved dev instances (used by the web app)")
    o.add_argument("--test-file", help="JSONL of saved test instances")
    o.add_argument("--schema-file")
    o.add_argument("--seeds-file", help="JSON list of extra seed vocabularies")
    o.add_argument("--w-judge", type=float, default=0.10)
    o.add_argument("--w-tokens", type=float, default=0.05)
    o.add_argument("--timeout", type=float, default=300, help="seconds before one model call is abandoned")
    o.set_defaults(func=cmd_optimize)
    t = sub.add_parser("translate")
    t.add_argument("--layer", required=True)
    t.add_argument("--input", required=True)
    t.add_argument("--queries", type=int, default=3)
    t.set_defaults(func=cmd_translate)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
