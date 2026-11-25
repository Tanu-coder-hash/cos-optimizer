"""Benchmark task families with verifiable ground truth.

Three structural families, each represented as plain JSON so the same data can
be rendered under any symbol vocabulary:

* grid   - 2D grid navigation (shortest obstacle-free path)
* graph  - weighted shortest-path cost on a (directed or undirected) graph
* bricks - Brick-World style stacking: which bricks to remove, in order, to take a target

Every instance carries a canonical answer plus solver traces used to build the
few-shot exemplar in whichever vocabulary is being tested.
"""
from __future__ import annotations

import heapq
import random
import string
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

DOMAINS = ("grid", "graph", "bricks")
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

DIFFICULTY = {
    "grid": {"easy": dict(size=5, density=0.20), "medium": dict(size=7, density=0.25), "hard": dict(size=10, density=0.28)},
    "graph": {"easy": dict(n=6, extra=0.25), "medium": dict(n=9, extra=0.25), "hard": dict(n=13, extra=0.22)},
    "bricks": {"easy": dict(n=6, stacks=2), "medium": dict(n=10, stacks=3), "hard": dict(n=15, stacks=4)},
}


@dataclass
class Instance:
    id: str
    domain: str
    data: dict
    answer: Any
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(id=d["id"], domain=d["domain"], data=d["data"], answer=d.get("answer"), meta=d.get("meta", {}))


# --------------------------------------------------------------------------- grid
def grid_bfs(grid: list[str], start, goal):
    """Return (path_coords, moves) of one shortest path, or None."""
    H, W = len(grid), len(grid[0])
    s, g = tuple(start), tuple(goal)
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            break
        for m, (dr, dc) in MOVES.items():
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != "#" and (nr, nc) not in prev:
                prev[(nr, nc)] = (cur, m)
                q.append((nr, nc))
    if g not in prev:
        return None
    path, moves, cur = [g], [], g
    while prev[cur] is not None:
        p, m = prev[cur]
        moves.append(m)
        path.append(p)
        cur = p
    return [list(p) for p in path[::-1]], moves[::-1]


def solve_grid(data: dict) -> dict:
    res = grid_bfs(data["grid"], data["start"], data["goal"])
    if res is None:
        raise ValueError("grid has no path from start to goal")
    path, moves = res
    return {"length": len(moves), "moves": moves, "path": path}


def check_grid_moves(data: dict, moves: list[str]) -> tuple[bool, bool]:
    """Return (valid, reached_goal)."""
    grid = data["grid"]
    H, W = len(grid), len(grid[0])
    r, c = data["start"]
    for m in moves:
        if m not in MOVES:
            return False, False
        dr, dc = MOVES[m]
        r, c = r + dr, c + dc
        if not (0 <= r < H and 0 <= c < W) or grid[r][c] == "#":
            return False, False
    return True, [r, c] == list(data["goal"])


def gen_grid(rng: random.Random, size: int, density: float) -> dict:
    for _ in range(500):
        H = W = size
        cells = [["#" if rng.random() < density else "." for _ in range(W)] for _ in range(H)]
        free = [(r, c) for r in range(H) for c in range(W) if cells[r][c] == "."]
        if len(free) < 2:
            continue
        s, g = rng.sample(free, 2)
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < size // 2:
            continue
        grid = ["".join(row) for row in cells]
        if grid_bfs(grid, s, g):
            return {"grid": grid, "start": list(s), "goal": list(g)}
    raise RuntimeError("could not generate a solvable grid")


# --------------------------------------------------------------------------- graph
def dijkstra_trace(data: dict) -> dict:
    nodes, directed = data["nodes"], data.get("directed", True)
    adj: dict[str, list[tuple[str, int]]] = {n: [] for n in nodes}
    for u, v, w in data["edges"]:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    for n in adj:
        adj[n].sort()
    src, tgt = data["source"], data["target"]
    dist = {src: 0}
    prev: dict[str, str | None] = {src: None}
    heap = [(0, src)]
    settled: set[str] = set()
    steps = []  # each: {"node", "dist", "relax": [(nbr, newdist)], "frontier": {node: dist}}
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        relax = []
        if u != tgt:
            for v, w in adj[u]:
                if v in settled:
                    continue
                nd = d + w
                if nd < dist.get(v, 10**9):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
                    relax.append((v, nd))
        frontier = {n: dist[n] for n in dist if n not in settled}
        steps.append({"node": u, "dist": d, "relax": relax, "frontier": dict(sorted(frontier.items(), key=lambda x: (x[1], x[0])))})
        if u == tgt:
            break
    if tgt not in settled:
        raise ValueError("target unreachable from source")
    path, cur = [], tgt
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path = path[::-1]
    wmap = {}
    for u, v, w in data["edges"]:
        wmap[(u, v)] = min(w, wmap.get((u, v), 10**9))
        if not directed:
            wmap[(v, u)] = min(w, wmap.get((v, u), 10**9))
    cum = [0]
    for a, b in zip(path, path[1:]):
        cum.append(cum[-1] + wmap[(a, b)])
    return {"cost": dist[tgt], "path": path, "cum": cum, "steps": steps}


def solve_graph(data: dict) -> dict:
    t = dijkstra_trace(data)
    return {"cost": t["cost"], "path": t["path"]}


def _node_names(n: int) -> list[str]:
    if n <= 26:
        return list(string.ascii_uppercase[:n])
    return [f"N{i}" for i in range(n)]


def gen_graph(rng: random.Random, n: int, extra: float, directed: bool | None = None, wmax: int = 9) -> dict:
    directed = rng.random() < 0.5 if directed is None else directed
    for _ in range(500):
        names = _node_names(n)
        order = names[:]
        rng.shuffle(order)
        edges = {}
        # random spanning structure so most nodes are reachable
        for i in range(1, n):
            u = order[rng.randrange(0, i)]
            v = order[i]
            edges[(u, v)] = rng.randint(1, wmax)
        for u in names:
            for v in names:
                if u != v and (u, v) not in edges and (v, u) not in edges and rng.random() < extra / 2:
                    edges[(u, v)] = rng.randint(1, wmax)
        edge_list = [[u, v, w] for (u, v), w in edges.items()]
        rng.shuffle(edge_list)
        src = order[0]
        tgt = rng.choice(order[n // 2:])
        data = {"nodes": names, "edges": edge_list, "directed": directed, "source": src, "target": tgt}
        try:
            t = dijkstra_trace(data)
        except ValueError:
            continue
        if len(t["path"]) >= 3:
            return data
    raise RuntimeError("could not generate graph")


# --------------------------------------------------------------------------- bricks
def bricks_stacks(data: dict) -> list[list[str]]:
    """Reconstruct stacks bottom->top from 'on' relations."""
    above = {}
    on_table = []
    for upper, lower in data["on"]:
        if lower == "table":
            on_table.append(upper)
        else:
            above[lower] = upper
    stacks = []
    for b in sorted(on_table):
        st = [b]
        while st[-1] in above:
            st.append(above[st[-1]])
        stacks.append(st)
    return stacks


def solve_bricks(data: dict) -> dict:
    tgt = data["target"]
    for st in bricks_stacks(data):
        if tgt in st:
            i = st.index(tgt)
            return {"sequence": list(reversed(st[i:])), "stack": st}
    raise ValueError("target brick not found")


def gen_bricks(rng: random.Random, n: int, stacks: int) -> dict:
    names = list(string.ascii_uppercase[:n])
    rng.shuffle(names)
    piles = [[] for _ in range(stacks)]
    for i, b in enumerate(names):
        piles[i % stacks if i < stacks else rng.randrange(stacks)].append(b)
    on = []
    for p in piles:
        for i, b in enumerate(p):
            on.append([b, "table" if i == 0 else p[i - 1]])
    rng.shuffle(on)
    candidates = [b for p in piles for b in p[:-1]] or names  # prefer non-top targets
    return {"bricks": sorted(names), "on": on, "target": rng.choice(candidates)}


# --------------------------------------------------------------------------- dispatch
SOLVERS = {"grid": solve_grid, "graph": solve_graph, "bricks": solve_bricks}


def solve(domain: str, data: dict) -> dict:
    return SOLVERS[domain](data)


def generate(domain: str, rng: random.Random, difficulty: str = "medium", **overrides) -> dict:
    p = {**DIFFICULTY[domain][difficulty], **overrides}
    if domain == "grid":
        return gen_grid(rng, **p)
    if domain == "graph":
        return gen_graph(rng, **p)
    if domain == "bricks":
        return gen_bricks(rng, **p)
    raise ValueError(domain)


def make_dataset(domain: str, n: int, difficulty: str = "medium", seed: int = 0, prefix: str = "") -> list[Instance]:
    rng = random.Random(f"{domain}-{difficulty}-{seed}")
    out = []
    for i in range(n):
        data = generate(domain, rng, difficulty)
        out.append(Instance(id=f"{prefix}{domain}-{difficulty}-{seed}-{i}", domain=domain, data=data,
                            answer=solve(domain, data), meta={"difficulty": difficulty}))
    return out


def make_exemplar(domain: str, like: Instance | None = None) -> Instance:
    """A small fixed solved example used as the 1-shot demonstration."""
    rng = random.Random(f"exemplar-{domain}")
    if domain == "graph":
        directed = like.data.get("directed", True) if like else True
        data = gen_graph(rng, n=5, extra=0.3, directed=directed)
    elif domain == "grid":
        data = gen_grid(rng, size=4, density=0.2)
    else:
        # a stack deep enough that the example demonstrates a multi-step removal chain
        for _ in range(500):
            data = gen_bricks(rng, n=6, stacks=2)
            st = next(s for s in bricks_stacks(data) if len(s) >= 4) if any(len(s) >= 4 for s in bricks_stacks(data)) else None
            if st:
                data["target"] = st[-4]
                break
        # use names that never occur in real instances (A-O) so the model cannot copy the example's answer
        ren = {a: b for a, b in zip("ABCDEFGHIJ", "PQRSTUVWXY")}
        data = {"bricks": sorted(ren[b] for b in data["bricks"]),
                "on": [[ren[u], "table" if l == "table" else ren[l]] for u, l in data["on"]],
                "target": ren[data["target"]]}
    return Instance(id=f"exemplar-{domain}", domain=domain, data=data, answer=solve(domain, data), meta={"exemplar": True})
