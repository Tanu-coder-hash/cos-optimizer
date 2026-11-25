"""LLM-as-a-judge for reasoning traces, plus an offline heuristic fallback.

Accuracy is always measured by exact verifiers; the judge adds a *process*
signal (is the symbolic trace consistent and faithful?) and a written critique
that the proposer uses to design the next vocabulary.
"""
from __future__ import annotations

import json
import re

from .render import extract_answer_line, render_block

DIMS = ("symbol_consistency", "step_validity", "faithfulness", "concision")

JUDGE_SYSTEM = "You are a strict evaluator of step-by-step reasoning traces. You reply with JSON only."

JUDGE_PROMPT = """A model solved a structured reasoning task using a prescribed notation.

NOTATION SPEC (JSON):
{spec}

TASK AS SHOWN TO THE MODEL:
{task}

REFERENCE ANSWER: {gold}
VERIFIER RESULT: the model's final answer is {verdict}.

MODEL OUTPUT:
<<<
{output}
>>>

Score each dimension from 1 (very poor) to 5 (excellent):
- symbol_consistency: uses the prescribed symbols consistently, no drift into ad-hoc notation
- step_validity: each intermediate step follows from the previous ones and the task constraints
- faithfulness: steps refer to facts actually present in the task (no hallucinated walls/edges/bricks)
- concision: no redundant or bloated steps
Then write one or two sentences of critique naming the main notation-related failure (or "none"), and suggest one concrete change to the notation that would help.

Reply with JSON only:
{{"symbol_consistency": int, "step_validity": int, "faithfulness": int, "concision": int, "critique": str, "suggestion": str}}"""


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


class LLMJudge:
    kind = "llm"

    def __init__(self, llm):
        self.llm = llm

    def judge(self, inst, spec, output: str, correct: bool) -> dict:
        prompt = JUDGE_PROMPT.format(spec=json.dumps(spec, ensure_ascii=False), task=render_block(inst, spec),
                                     gold=json.dumps(inst.answer, ensure_ascii=False)[:400],
                                     verdict="CORRECT" if correct else "INCORRECT", output=(output or "")[:4000])
        c = self.llm.complete(JUDGE_SYSTEM, prompt, max_tokens=400)
        d = _parse_json(c.text) or {}
        scores = {}
        for k in DIMS:
            try:
                scores[k] = max(1, min(5, int(d.get(k, 3))))
            except (TypeError, ValueError):
                scores[k] = 3
        scores["score"] = sum(scores[k] for k in DIMS) / (5 * len(DIMS))
        scores["critique"] = str(d.get("critique", "judge output unparsable" if not d else ""))[:500]
        scores["suggestion"] = str(d.get("suggestion", ""))[:300]
        return scores


class HeuristicJudge:
    """Offline proxy (no model calls). Only for smoke tests."""
    kind = "heuristic"

    def judge(self, inst, spec, output: str, correct: bool) -> dict:
        line, found = extract_answer_line(output or "")
        body = (output or "").rsplit("ANSWER", 1)[0]
        toks = [v for k, v in spec.get("params", {}).items() if isinstance(v, str) and k not in ("layout", "trace", "coord_fmt", "weight_fmt", "node_names", "dist_fmt")]
        used = sum(1 for t in toks if t and t in body) / max(1, len(toks)) if toks else 0.6
        n = len(body)
        s = {"symbol_consistency": 1 + round(4 * used), "step_validity": 5 if correct else 2,
             "faithfulness": 5 if found else 2, "concision": 5 if n < 300 else 4 if n < 700 else 3 if n < 1500 else 2}
        s["score"] = sum(s[k] for k in DIMS) / (5 * len(DIMS))
        s["critique"] = "heuristic judge (offline): " + ("ok" if correct else "final answer wrong; trace likely broke down")
        s["suggestion"] = ""
        return s
