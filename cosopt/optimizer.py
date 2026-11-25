"""Automated Chain-of-Symbol vocabulary optimisation.

Loop
----
1. Seed population: CoT natural-language baseline, hand-designed CoS, random CoS.
2. Evaluate each vocabulary on the dev split: exact verifier accuracy, token cost,
   LLM-judge process score on a sample of traces (+ critiques).
3. Each generation the proposer (LLM or random-evolutionary) reads the leaderboard
   and critiques and proposes new vocabularies.
4. Successive halving: children are screened on a small subset; only promising
   ones are evaluated on the full dev split.
5. The winner is compared against CoT and the hand-designed CoS on the held-out
   test split (bootstrap CIs, exact McNemar) and compiled into a translation layer.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from . import vocab
from .compiler import compile_layer
from .render import est_tokens, grade, render_prompt
from .stats import bootstrap_ci, mcnemar
from .tasks import Instance


@dataclass
class OptConfig:
    domain: str = "grid"
    generations: int = 4
    population: int = 6          # seed vocabularies (incl. CoT + default CoS)
    children: int = 4            # proposals per generation
    screen_size: int = 8         # successive-halving rung 1
    promote_frac: float = 0.5
    judge_samples: int = 3       # traces judged per candidate (0 = no judge)
    w_judge: float = 0.10
    w_tokens: float = 0.05
    concurrency: int = 4
    seed: int = 0


def fitness(acc, judge, tokens, ref_tokens, cfg: OptConfig):
    rel = (tokens / ref_tokens - 1.0) if ref_tokens else 0.0
    return acc + cfg.w_judge * judge - cfg.w_tokens * max(-1.0, min(2.0, rel))


class Optimizer:
    def __init__(self, solver, judge, proposer, dev: list[Instance], test: list[Instance], exemplar: Instance,
                 schema: dict, cfg: OptConfig, out_dir: str = "runs", callback=None, extra_seeds: list | None = None):
        self.solver, self.judge, self.proposer = solver, judge, proposer
        self.dev, self.test, self.exemplar, self.schema, self.cfg = dev, test, exemplar, schema, cfg
        self.cb = callback or (lambda *a, **k: None)
        self.extra_seeds = extra_seeds or []
        self.rng = random.Random(cfg.seed)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(out_dir, f"{stamp}-{cfg.domain}-{solver.label.replace(':', '_').replace('/', '_')}")
        os.makedirs(self.dir, exist_ok=True)
        self.history: list[dict] = []
        self.ref_tokens = None
        self._calls = open(os.path.join(self.dir, "calls.jsonl"), "a", encoding="utf-8")

    # ------------------------------------------------------------------ eval
    def _run_one(self, inst: Instance, spec: dict) -> dict:
        system, user = render_prompt(inst, spec, self.exemplar)
        c = self.solver.solve(system, user, ctx={"instance": inst, "spec": spec})
        g = grade(inst, spec, c.text)
        rec = {"id": inst.id, "spec": spec["name"], "correct": g["correct"], "valid": g["valid"],
               "answer_line_found": g["answer_line_found"], "parsed": g["parsed"],
               "in_tokens": c.in_tokens or est_tokens(system + user), "out_tokens": c.out_tokens or est_tokens(c.text),
               "error": c.error, "cached": c.cached, "output": c.text}
        self._calls.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._calls.flush()
        return rec

    def evaluate(self, spec: dict, instances: list[Instance], prior: list[dict] | None = None, do_judge=True) -> dict:
        with ThreadPoolExecutor(self.cfg.concurrency) as ex:
            recs = list(ex.map(lambda i: self._run_one(i, spec), instances))
        failed = [r for r in recs if r["error"]]
        if recs and len(failed) == len(recs):
            raise RuntimeError(f"Every model call failed, so stopping instead of reporting 0%. Last error: {failed[-1]['error']}. "
                               "Check the model is running (for Ollama: `ollama serve`) and the model name is exact (`ollama list`).")
        recs = (prior or []) + recs
        n = len(recs)
        acc = sum(r["correct"] for r in recs) / n
        tokens = sum(r["in_tokens"] + r["out_tokens"] for r in recs) / n
        judged = []
        if do_judge and self.cfg.judge_samples and self.judge:
            # prefer judging failures (more informative critiques), then successes
            pool = [r for r in recs if not r["correct"]] + [r for r in recs if r["correct"]]
            by_id = {i.id: i for i in self.dev + self.test}
            sample = pool[: self.cfg.judge_samples]
            with ThreadPoolExecutor(self.cfg.concurrency) as ex:
                judged = list(ex.map(lambda r: {**self.judge.judge(by_id[r["id"]], spec, r["output"], r["correct"]),
                                                "id": r["id"], "correct": r["correct"], "output": r["output"]}, sample))
        jscore = sum(j["score"] for j in judged) / len(judged) if judged else 0.0
        return {"spec": spec, "n": n, "acc": acc, "tokens": tokens,
                "valid": sum(r["valid"] for r in recs) / n,
                "parse_rate": sum(r["answer_line_found"] for r in recs) / n,
                "errors": sum(1 for r in recs if r["error"]),
                "judge": jscore, "judged": judged,
                "failures": [j for j in judged if not j["correct"]],
                "records": recs}

    def _finish(self, res, gen, origin):
        if self.ref_tokens is None:
            self.ref_tokens = res["tokens"]
        res["fitness"] = fitness(res["acc"], res["judge"], res["tokens"], self.ref_tokens, self.cfg)
        res["generation"] = gen
        res["origin"] = res["spec"].get("origin", origin)
        self.history.append(res)
        self.history.sort(key=lambda r: (-(r["n"] >= len(self.dev)), -r["fitness"]))
        slim = {k: v for k, v in res.items() if k not in ("records",)}
        with open(os.path.join(self.dir, "candidates.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(slim, ensure_ascii=False, default=str) + "\n")
        self.cb("candidate", res)
        return res

    def leaderboard(self):
        full = [h for h in self.history if h["n"] >= len(self.dev)]
        return sorted(full, key=lambda r: -r["fitness"])

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg, d = self.cfg, self.cfg.domain
        t0 = time.time()
        self.cb("log", f"Run directory: {self.dir}")
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump({"config": asdict(cfg), "solver": self.solver.label, "judge": getattr(self.judge, "kind", None),
                       "proposer": getattr(self.proposer, "kind", None), "schema": self.schema,
                       "dev": len(self.dev), "test": len(self.test), "simulated": self.solver.is_simulated}, f, indent=2)

        seeds = [vocab.nl_baseline(d), vocab.default_cos(d)]
        seeds[0]["origin"], seeds[1]["origin"] = "seed:cot", "seed:default-cos"
        for s in self.extra_seeds:
            if vocab.validate(s)[0] and vocab.key(s) not in {vocab.key(x) for x in seeds}:
                seeds.append({**s, "origin": "seed:user"})
        while len(seeds) < cfg.population:
            s = vocab.random_spec(d, self.rng)
            s["origin"] = "seed:random"
            if vocab.key(s) not in {vocab.key(x) for x in seeds}:
                seeds.append(s)
        self.cb("log", f"Generation 0: evaluating {len(seeds)} seed vocabularies on {len(self.dev)} dev instances")
        for s in seeds:
            self._finish(self.evaluate(s, self.dev), 0, s["origin"])
        self.cb("generation", 0, self.leaderboard())

        for g in range(1, cfg.generations + 1):
            kids = self.proposer.propose(d, self.schema, self.leaderboard(), cfg.children)
            seen = {vocab.key(h["spec"]) for h in self.history}
            kids = [k for k in kids if vocab.key(k) not in seen]
            self.cb("log", f"Generation {g}: {len(kids)} new proposals ({', '.join(k.get('origin', '?') for k in kids)})")
            screen, rest = self.dev[: cfg.screen_size], self.dev[cfg.screen_size:]
            screened = [(k, self.evaluate(k, screen, do_judge=False)) for k in kids]
            bar = sorted((r["acc"] for _, r in screened), reverse=True)
            cut = bar[max(0, int(len(bar) * cfg.promote_frac) - 1)] if bar else 0
            best_acc = self.leaderboard()[0]["acc"] if self.leaderboard() else 0
            for k, r in screened:
                if r["acc"] >= cut and r["acc"] >= best_acc - 0.25 and rest:
                    full = self.evaluate(k, rest, prior=r["records"])
                    self._finish(full, g, k.get("origin", "?"))
                else:
                    r["fitness"] = fitness(r["acc"], 0, r["tokens"], self.ref_tokens, cfg)
                    self._finish(r, g, k.get("origin", "?"))
                    self.cb("log", f"  pruned {k['name']} after screening (acc {r['acc']:.2f})")
            self.cb("generation", g, self.leaderboard())

        lb = self.leaderboard()
        best = next(h for h in lb if h["spec"]["mode"] == "cos")
        self.cb("log", f"Best vocabulary: {best['spec']['name']} (dev acc {best['acc']:.3f}). Running held-out test...")
        cot, dcos = vocab.nl_baseline(d), vocab.default_cos(d)
        finals = {"cot": cot, "default_cos": dcos, "optimized": best["spec"]}
        if vocab.key(best["spec"]) == vocab.key(dcos):
            finals.pop("default_cos")
        test_res = {name: self.evaluate(s, self.test, do_judge=False) for name, s in finals.items()}
        if "default_cos" not in test_res:
            test_res["default_cos"] = test_res["optimized"]
        summary = {}
        for name, r in test_res.items():
            m, lo, hi = bootstrap_ci([x["correct"] for x in r["records"]], seed=cfg.seed)
            summary[name] = {"acc": m, "ci95": [lo, hi], "tokens": r["tokens"], "n": r["n"], "spec": r["spec"]}
        opt = [x["correct"] for x in test_res["optimized"]["records"]]
        sig = {"optimized_vs_cot": mcnemar(opt, [x["correct"] for x in test_res["cot"]["records"]]),
               "optimized_vs_default_cos": mcnemar(opt, [x["correct"] for x in test_res["default_cos"]["records"]]),
               "default_cos_vs_cot": mcnemar([x["correct"] for x in test_res["default_cos"]["records"]],
                                             [x["correct"] for x in test_res["cot"]["records"]])}
        layer = compile_layer(best["spec"], self.exemplar, metrics={"dev_acc": best["acc"], "test": summary["optimized"],
                                                                    "solver": self.solver.label})
        result = {"run_dir": self.dir, "domain": d, "solver": self.solver.label, "simulated": self.solver.is_simulated,
                  "elapsed_sec": time.time() - t0, "usage": {"solver": self.solver.usage.as_dict()},
                  "test": summary, "significance": sig, "best": {k: v for k, v in best.items() if k != "records"},
                  "leaderboard": [{k: v for k, v in h.items() if k not in ("records", "judged")} for h in lb],
                  "all_candidates": [{k: v for k, v in h.items() if k not in ("records", "judged")} for h in self.history],
                  "layer": layer, "config": asdict(cfg), "schema": self.schema}
        with open(os.path.join(self.dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(self.dir, "compiled_layer.json"), "w", encoding="utf-8") as f:
            json.dump(layer, f, indent=2, ensure_ascii=False)
        from .report import markdown_report
        with open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(markdown_report(result))
        self._calls.close()
        self.cb("final", result)
        return result
