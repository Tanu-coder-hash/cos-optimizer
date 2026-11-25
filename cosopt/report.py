"""Markdown experiment report."""
from __future__ import annotations

import json


def _pct(x):
    return f"{100 * x:.1f}%"


def markdown_report(r: dict) -> str:
    L = [f"# CoS vocabulary optimisation - {r['domain']}", ""]
    if r.get("simulated"):
        L += ["> **SIMULATED RUN.** The solver was the offline simulator; every number below is synthetic and only "
              "demonstrates the pipeline. Re-run with a real model before drawing conclusions.", ""]
    L += [f"- Solver: `{r['solver']}`", f"- Run dir: `{r['run_dir']}`", f"- Wall time: {r['elapsed_sec']:.0f}s",
          f"- Calls: {r['usage']['solver']}", ""]
    L += ["## Held-out test", "", "| Condition | Accuracy | 95% CI | Mean tokens | n |", "|---|---|---|---|---|"]
    names = {"cot": "CoT (natural language)", "default_cos": "CoS (hand-designed)", "optimized": "CoS (optimised)"}
    for k in ("cot", "default_cos", "optimized"):
        s = r["test"][k]
        L.append(f"| {names[k]} | {_pct(s['acc'])} | {_pct(s['ci95'][0])} - {_pct(s['ci95'][1])} | {s['tokens']:.0f} | {s['n']} |")
    L += ["", "## Paired significance (exact McNemar)", "", "| Comparison | A-only correct | B-only correct | p |", "|---|---|---|---|"]
    for k, v in r["significance"].items():
        L.append(f"| {k.replace('_', ' ')} | {v['a_only']} | {v['b_only']} | {v['p_value']:.4f} |")
    L += ["", "## Optimised vocabulary", "", "```json", json.dumps(r["best"]["spec"], indent=2, ensure_ascii=False), "```", ""]
    if r["best"]["spec"].get("rationale"):
        L += [f"Proposer rationale: {r['best']['spec']['rationale']}", ""]
    L += ["## Dev leaderboard", "", "| # | Name | Origin | Gen | Acc | Judge | Tokens | Fitness |", "|---|---|---|---|---|---|---|---|"]
    for i, h in enumerate(r["leaderboard"][:15], 1):
        L.append(f"| {i} | {h['spec']['name']} | {h.get('origin', '')} | {h.get('generation', '')} | {_pct(h['acc'])} | "
                 f"{h['judge']:.2f} | {h['tokens']:.0f} | {h['fitness']:.3f} |")
    L += ["", "## Example prompt with the optimised layer", "", "```", r["layer"]["preview"]["user"], "```", ""]
    return "\n".join(L)
