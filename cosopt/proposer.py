"""Vocabulary proposers: LLM-driven (OPRO/APE-style) and random evolutionary."""
from __future__ import annotations

import json
import random
import re

from . import vocab

PROPOSER_SYSTEM = ("You are a research assistant designing symbolic notations (Chain-of-Symbol vocabularies) "
                   "that help language models reason about structured spatial / graph problems. Reply with JSON only.")

PROPOSER_PROMPT = """Domain: {domain}
Structure detected in the user's data:
{schema}

A vocabulary decides how the structured input is serialised and what notation the worked example's reasoning trace uses.
The parameter space:
{space}

Evaluated vocabularies so far (best first). acc = verified accuracy on the dev set, judge = LLM-judge process score (0-1), tokens = mean prompt+output tokens:
{history}

Recent failure cases and judge critiques:
{failures}

Propose {n} NEW vocabularies that you expect to beat the best one. Mix exploitation (small, targeted edits to the leaders addressing the critiques) and exploration (structurally different choices or novel token symbols). Do not repeat an evaluated configuration.

Reply with a JSON list, each item: {{"rationale": "<one sentence>", "params": {{...every slot...}}}}"""


def _fmt_history(history, k=8):
    rows = []
    for r in history[:k]:
        rows.append(json.dumps({"name": r["spec"]["name"], "mode": r["spec"]["mode"], "params": r["spec"].get("params", {}),
                                "acc": round(r["acc"], 3), "judge": round(r["judge"], 3), "tokens": round(r["tokens"])},
                               ensure_ascii=False))
    return "\n".join(rows) or "(none)"


def _fmt_failures(history, k=4):
    out = []
    for r in history[:3]:
        for f in r.get("failures", [])[:2]:
            out.append(f"[{r['spec']['name']}] output tail: {f['output'][-300:]!r}\n  critique: {f.get('critique', '')}  suggestion: {f.get('suggestion', '')}")
    return "\n".join(out[:k]) or "(none recorded)"


class RandomProposer:
    kind = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def propose(self, domain, schema, history, n):
        parents = [h["spec"] for h in history if h["spec"]["mode"] == "cos"][:4] or [vocab.default_cos(domain)]
        out = []
        for i in range(n):
            r = self.rng.random()
            if r < 0.5:
                child = vocab.mutate(self.rng.choice(parents), self.rng)
                child["origin"] = "mutate"
            elif r < 0.8 and len(parents) > 1:
                a, b = self.rng.sample(parents, 2)
                child = vocab.crossover(a, b, self.rng)
                child["origin"] = "crossover"
            else:
                child = vocab.random_spec(domain, self.rng)
                child["origin"] = "random"
            out.append(child)
        return out


class LLMProposer:
    kind = "llm"

    def __init__(self, llm, seed=0, fallback_ratio=0.25):
        self.llm = llm
        self.fallback = RandomProposer(seed)
        self.fallback_ratio = fallback_ratio
        self.last_raw = ""

    def propose(self, domain, schema, history, n):
        n_llm = max(1, round(n * (1 - self.fallback_ratio)))
        prompt = PROPOSER_PROMPT.format(domain=domain, schema=json.dumps(schema, ensure_ascii=False),
                                        space=vocab.describe_space(domain), history=_fmt_history(history),
                                        failures=_fmt_failures(history), n=n_llm)
        c = self.llm.complete(PROPOSER_SYSTEM, prompt, max_tokens=2000, temperature=0.7)
        self.last_raw = c.text
        out = []
        m = re.search(r"\[.*\]", c.text or "", flags=re.S)
        items = []
        if m:
            try:
                items = json.loads(m.group())
            except json.JSONDecodeError:
                items = []
        seen = {vocab.key(h["spec"]) for h in history}
        base = vocab.default_cos(domain)["params"]
        for i, it in enumerate(items if isinstance(items, list) else []):
            if not isinstance(it, dict):
                continue
            params = {**base, **(it.get("params") or {})}
            spec = {"name": f"llm-{len(history)}-{i}", "domain": domain, "mode": "cos", "params": params,
                    "origin": "llm", "rationale": str(it.get("rationale", ""))[:300]}
            ok, _ = vocab.validate(spec)
            if ok and vocab.key(spec) not in seen:
                seen.add(vocab.key(spec))
                out.append(spec)
        out += self.fallback.propose(domain, schema, history, n - len(out))
        return out[:n]
