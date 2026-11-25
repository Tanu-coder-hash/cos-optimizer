"""Symbol vocabularies: the search space the optimizer explores.

A *spec* is a JSON-serialisable dict:

    {"name": str, "domain": "grid"|"graph"|"bricks", "mode": "nl"|"cos", "params": {...}}

``mode == "nl"`` is the Chain-of-Thought natural-language baseline.
``mode == "cos"`` is a Chain-of-Symbol vocabulary whose ``params`` pick
structural choices (layout, trace style - from a fixed menu) and token choices
(any short string - the LLM proposer may invent new ones).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random

# Categorical slots: value must come from the list.
# Token slots: value is a short string; the list holds suggested seeds.
SPACES: dict[str, dict] = {
    "grid": {
        "categorical": {
            "layout": ["matrix", "coord_list", "indexed_rows"],
            "coord_fmt": ["(r,c)", "rRcC", "[r][c]"],
            "trace": ["coords", "moves", "coords+moves"],
            "legend": [True, False],
        },
        "tokens": {
            "free": [".", "0", "_", "·"],
            "wall": ["#", "1", "X", "█"],
            "start": ["S", "@", "A"],
            "goal": ["G", "*", "B"],
            "move_up": ["U", "↑", "N", "^"],
            "move_down": ["D", "↓", "S", "v"],
            "move_left": ["L", "←", "W", "<"],
            "move_right": ["R", "→", "E", ">"],
            "chain": ["→", "->", "=>", "|"],
        },
    },
    "graph": {
        "categorical": {
            "layout": ["edge_list", "adjacency", "matrix"],
            "weight_fmt": ["inline", "suffix", "paren"],
            "node_names": ["original", "letters", "numbers"],
            "trace": ["dijkstra_table", "path_chain", "frontier_sets"],
            "dist_fmt": ["d(X)=k", "X:k", "X[k]"],
            "legend": [True, False],
        },
        "tokens": {
            "edge_op": ["->", "→", ">", "=>"],
            "chain": ["→", "->", "⇒", "|"],
            "sep": [",", ";", "|", "/"],
        },
    },
    "bricks": {
        "categorical": {
            "layout": ["pairs", "stacks"],
            "trace": ["remove_chain", "stack_rebuild"],
            "legend": [True, False],
        },
        "tokens": {
            "on_op": ["/", ">", "^", "|"],
            "table": ["_", "⊥", "▭", "≡"],
            "sep": [",", ";", "|", "&"],
            "chain": ["⇒", "->", "=>", ":"],
        },
    },
}

# Characters that would break parsing or the final ANSWER line.
_FORBIDDEN = set(" \n\t\r`\"'")


def default_cos(domain: str) -> dict:
    """Hand-designed CoS vocabulary in the spirit of Hu et al. (2023)."""
    p = {
        "grid": {"layout": "matrix", "coord_fmt": "(r,c)", "trace": "coords+moves", "legend": True,
                 "free": ".", "wall": "#", "start": "S", "goal": "G",
                 "move_up": "↑", "move_down": "↓", "move_left": "←", "move_right": "→", "chain": "|"},
        "graph": {"layout": "edge_list", "weight_fmt": "inline", "node_names": "original", "trace": "dijkstra_table",
                  "dist_fmt": "X:k", "legend": True, "edge_op": "->", "chain": "⇒", "sep": ","},
        "bricks": {"layout": "pairs", "trace": "remove_chain", "legend": True,
                   "on_op": "/", "table": "_", "sep": ",", "chain": "⇒"},
    }[domain]
    return {"name": "default-cos", "domain": domain, "mode": "cos", "params": p}


def nl_baseline(domain: str) -> dict:
    return {"name": "cot-natural-language", "domain": domain, "mode": "nl", "params": {}}


def token_slots(domain: str) -> list[str]:
    return list(SPACES[domain]["tokens"].keys())


def validate(spec: dict) -> tuple[bool, list[str]]:
    errs: list[str] = []
    dom = spec.get("domain")
    if dom not in SPACES:
        return False, [f"unknown domain {dom!r}"]
    if spec.get("mode") == "nl":
        return True, []
    if spec.get("mode") != "cos":
        return False, ["mode must be 'nl' or 'cos'"]
    params = spec.get("params", {})
    space = SPACES[dom]
    for k, opts in space["categorical"].items():
        if k not in params:
            errs.append(f"missing {k}")
        elif params[k] not in opts:
            errs.append(f"{k}={params[k]!r} not in {opts}")
    seen: dict[str, str] = {}
    for k in space["tokens"]:
        v = params.get(k)
        if not isinstance(v, str) or not v:
            errs.append(f"token {k} must be a non-empty string")
            continue
        if len(v) > 4:
            errs.append(f"token {k}={v!r} longer than 4 chars")
        if any(ch in _FORBIDDEN for ch in v):
            errs.append(f"token {k}={v!r} contains whitespace/quote")
        if v.isdigit() and dom != "grid":
            errs.append(f"token {k}={v!r} is numeric and would collide with weights/answers")
        if v in seen:
            errs.append(f"token {k}={v!r} collides with {seen[v]}")
        seen[v] = k
    if dom == "grid":
        mv = [params.get(k) for k in ("move_up", "move_down", "move_left", "move_right")]
        if any(m and m.isdigit() for m in mv):
            errs.append("move tokens must not be digits")
    if dom == "bricks":
        for k in space["tokens"]:
            v = params.get(k, "")
            if isinstance(v, str) and len(v) == 1 and v.isupper() and v.isalpha():
                errs.append(f"token {k}={v!r} could collide with brick names (A-Z)")
    if dom == "graph":
        for k in space["tokens"]:
            v = params.get(k, "")
            if isinstance(v, str) and v.isalpha():
                errs.append(f"token {k}={v!r} could collide with node names")
    return (not errs), errs


def random_spec(domain: str, rng: random.Random, name: str | None = None) -> dict:
    space = SPACES[domain]
    for _ in range(200):
        p = {k: rng.choice(v) for k, v in space["categorical"].items()}
        p.update({k: rng.choice(v) for k, v in space["tokens"].items()})
        spec = {"name": name or f"rand-{rng.randrange(10**6):06d}", "domain": domain, "mode": "cos", "params": p}
        if validate(spec)[0]:
            return spec
    return default_cos(domain)


def mutate(spec: dict, rng: random.Random, n_changes: int = 2) -> dict:
    dom = spec["domain"]
    base = spec if spec["mode"] == "cos" else default_cos(dom)
    space = SPACES[dom]
    slots = list(space["categorical"]) + list(space["tokens"])
    for _ in range(100):
        child = copy.deepcopy(base)
        for k in rng.sample(slots, k=min(n_changes, len(slots))):
            pool = space["categorical"].get(k) or space["tokens"].get(k)
            child["params"][k] = rng.choice(pool)
        child["name"] = f"mut-{rng.randrange(10**6):06d}"
        if validate(child)[0] and key(child) != key(base):
            return child
    return random_spec(dom, rng)


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    dom = a["domain"]
    if a["mode"] != "cos" or b["mode"] != "cos":
        return mutate(a if a["mode"] == "cos" else b, rng)
    for _ in range(50):
        p = {k: (a if rng.random() < 0.5 else b)["params"][k] for k in a["params"]}
        child = {"name": f"x-{rng.randrange(10**6):06d}", "domain": dom, "mode": "cos", "params": p}
        if validate(child)[0]:
            return child
    return mutate(a, rng)


def key(spec: dict) -> str:
    payload = json.dumps({"d": spec["domain"], "m": spec["mode"], "p": spec.get("params", {})}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def describe_space(domain: str) -> str:
    space = SPACES[domain]
    lines = ["Categorical slots (choose one of the listed values):"]
    for k, v in space["categorical"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("Token slots (any string of 1-4 non-space characters; must be unique; examples given):")
    for k, v in space["tokens"].items():
        lines.append(f"  - {k}: e.g. {v}")
    return "\n".join(lines)
