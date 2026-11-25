"""Schema inference: map arbitrary structural JSON onto a task family.

Accepted inputs (one object, a list of objects, or JSONL):

Grid
  {"grid": ["..#", "S.G"]}                     strings; S/G may sit inside the grid
  {"grid": [[0,0,1],[0,0,0]], "start": [0,0], "goal": [1,2]}   0/1 or bool matrix
Graph
  {"nodes": [...], "edges": [[u,v,w], ...], "source": u, "target": v, "directed": true}
  {"edges": [{"source": u, "target": v, "weight": w}, ...]}
  {"nodes": [{"id": u}, ...], "links": [{"source": u, "target": v, "weight": w}]}   (networkx node-link)
  {"adjacency": {"A": {"B": 3}, ...}}  or  {"adjacency": {"A": ["B", "C"]}}
  -> missing source/target: queries are generated automatically.
Bricks
  {"on": [["C","B"], ["B","table"], ...], "target": "B"}
  {"stacks": [["A","B","C"], ["D"]], "target": "B"}   (bottom -> top)
  -> missing target: queries are generated automatically.
Wrapped
  {"domain": "graph", "data": {...}, "answer": {...}}
"""
from __future__ import annotations

import json
import random
from collections import Counter

from .tasks import Instance, dijkstra_trace, solve

WALL_VALUES = {"#", "1", 1, True, "X", "x", "█", "wall", "W"}


def _report(domain, detected_by, data):
    if domain == "grid":
        g = data["grid"]
        return {"domain": "grid", "detected_by": detected_by, "dims": [len(g), len(g[0])],
                "entities": sum(row.count(".") for row in g), "relations": "4-neighbour adjacency",
                "walls": sum(row.count("#") for row in g)}
    if domain == "graph":
        w = Counter(e[2] for e in data["edges"])
        return {"domain": "graph", "detected_by": detected_by, "entities": len(data["nodes"]),
                "relations": len(data["edges"]), "directed": data.get("directed", True),
                "weighted": len(w) > 1, "weight_range": [min(w), max(w)] if w else None}
    return {"domain": "bricks", "detected_by": detected_by, "entities": len(data["bricks"]),
            "relations": len(data["on"]), "relation_type": "on-top-of (stacking)"}


# --------------------------------------------------------------------------- grid
def _norm_grid(obj):
    raw = obj["grid"] if isinstance(obj, dict) else obj
    rows = [list(r) if isinstance(r, str) else list(r) for r in raw]
    start = obj.get("start") if isinstance(obj, dict) else None
    goal = obj.get("goal") if isinstance(obj, dict) else None
    grid = []
    for r, row in enumerate(rows):
        s = ""
        for c, v in enumerate(row):
            if v in ("S", "s", "@") and start is None:
                start = [r, c]
            if v in ("G", "g", "*") and goal is None:
                goal = [r, c]
            s += "#" if v in WALL_VALUES else "."
        grid.append(s)
    if len({len(r) for r in grid}) != 1:
        raise ValueError("grid rows have different lengths")
    if start is None or goal is None:
        raise ValueError("grid needs a start and goal (keys 'start'/'goal' or S/G cells)")
    return {"grid": grid, "start": list(start), "goal": list(goal)}


# --------------------------------------------------------------------------- graph
def _edge_tuple(e):
    if isinstance(e, dict):
        u = e.get("source", e.get("u", e.get("from")))
        v = e.get("target", e.get("v", e.get("to")))
        w = e.get("weight", e.get("w", e.get("cost", 1)))
    else:
        u, v = e[0], e[1]
        w = e[2] if len(e) > 2 else 1
    return str(u), str(v), int(round(float(w)))


def _norm_graph(obj):
    directed = obj.get("directed", True)
    if "adjacency" in obj or "adj" in obj:
        adj = obj.get("adjacency", obj.get("adj"))
        edges = []
        for u, nbrs in adj.items():
            items = nbrs.items() if isinstance(nbrs, dict) else [(v, 1) for v in nbrs]
            for v, w in items:
                edges.append((str(u), str(v), int(w if not isinstance(w, dict) else w.get("weight", 1))))
        detected = "adjacency map"
    else:
        raw = obj.get("edges", obj.get("links"))
        edges = [_edge_tuple(e) for e in raw]
        detected = "node-link" if "links" in obj else "edge list"
    nodes = obj.get("nodes")
    if nodes:
        nodes = [str(n["id"]) if isinstance(n, dict) else str(n) for n in nodes]
    else:
        nodes = []
    for u, v, _ in edges:
        for x in (u, v):
            if x not in nodes:
                nodes.append(x)
    if any(w <= 0 for _, _, w in edges):
        raise ValueError("edge weights must be positive integers for shortest-path tasks")
    base = {"nodes": nodes, "edges": [list(e) for e in edges], "directed": bool(directed)}
    return base, detected


def _graph_queries(base, n, rng):
    if "source" in base and "target" in base:
        return [base]
    out, tries = [], 0
    nodes = base["nodes"]
    while len(out) < n and tries < 50 * n:
        tries += 1
        s, t = rng.sample(nodes, 2)
        d = {**base, "source": s, "target": t}
        try:
            tr = dijkstra_trace(d)
        except ValueError:
            continue
        if len(tr["path"]) >= 3 and all((o["source"], o["target"]) != (s, t) for o in out):
            out.append(d)
    if not out:
        raise ValueError("could not find any reachable source/target pair with a path of >= 2 edges")
    return out


# --------------------------------------------------------------------------- bricks
def _norm_bricks(obj):
    if "stacks" in obj:
        on = []
        for st in obj["stacks"]:
            for i, b in enumerate(st):
                on.append([str(b), "table" if i == 0 else str(st[i - 1])])
    else:
        on = [[str(a), "table" if str(b).lower() in ("table", "_", "floor", "ground") else str(b)] for a, b in obj["on"]]
    bricks = sorted({a for a, _ in on})
    return {"bricks": bricks, "on": on}


def _brick_queries(base, target, n, rng):
    if target:
        return [{**base, "target": str(target)}]
    lowers = sorted({b for _, b in base["on"] if b != "table"})
    picks = lowers[:] if lowers else base["bricks"][:]
    rng.shuffle(picks)
    return [{**base, "target": t} for t in picks[:n]]


# --------------------------------------------------------------------------- main
def detect(obj) -> str:
    if isinstance(obj, dict):
        if "domain" in obj and "data" in obj:
            return obj["domain"]
        if "grid" in obj or "map" in obj:
            return "grid"
        if any(k in obj for k in ("edges", "links", "adjacency", "adj")):
            return "graph"
        if "on" in obj or "stacks" in obj:
            return "bricks"
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, str)) and all(len(r) == len(obj[0]) for r in obj):
        return "grid"
    raise ValueError("could not recognise structure. Expected grid / graph (edges, links, adjacency) / bricks (on, stacks). "
                     "See cosopt/schema.py for accepted formats.")


def normalize(obj, queries_per_structure: int = 10, seed: int = 0) -> tuple[list[dict], dict]:
    """Return (list of canonical task data dicts, schema report)."""
    rng = random.Random(seed)
    dom = detect(obj)
    if isinstance(obj, dict) and "domain" in obj and "data" in obj:
        return normalize(obj["data"], queries_per_structure, seed)
    if dom == "grid":
        if isinstance(obj, dict) and "map" in obj and "grid" not in obj:
            obj = {**obj, "grid": obj["map"]}
        d = _norm_grid(obj)
        solve("grid", d)  # raises if unsolvable
        return [d], _report("grid", "grid matrix", d)
    if dom == "graph":
        base, how = _norm_graph(obj)
        if "source" in obj and "target" in obj:
            base.update(source=str(obj["source"]), target=str(obj["target"]))
        datas = _graph_queries(base, queries_per_structure, rng)
        return datas, _report("graph", how, datas[0])
    base = _norm_bricks(obj)
    datas = _brick_queries(base, obj.get("target"), queries_per_structure, rng)
    for d in datas:
        solve("bricks", d)
    return datas, _report("bricks", "stacks" if "stacks" in obj else "on-relations", datas[0])


def load_instances(text: str, queries_per_structure: int = 10, seed: int = 0, prefix: str = "user") -> tuple[list[Instance], dict]:
    """Parse JSON / JSONL text into Instances (answers computed by exact solvers)."""
    text = text.strip()
    try:
        payload = json.loads(text)
        objs = payload if isinstance(payload, list) and payload and isinstance(payload[0], dict) else [payload]
    except json.JSONDecodeError:
        objs = [json.loads(l) for l in text.splitlines() if l.strip()]
    insts, report, domains = [], None, set()
    for i, o in enumerate(objs):
        datas, rep = normalize(o, queries_per_structure, seed + i)
        report = report or rep
        domains.add(rep["domain"])
        for j, d in enumerate(datas):
            insts.append(Instance(id=f"{prefix}-{i}-{j}", domain=rep["domain"], data=d, answer=solve(rep["domain"], d),
                                  meta={"source": "upload"}))
    if len(domains) > 1:
        raise ValueError(f"mixed domains in one upload: {domains}; upload one family at a time")
    report["instances"] = len(insts)
    report["structures"] = len(objs)
    return insts, report
