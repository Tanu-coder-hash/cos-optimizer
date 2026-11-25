"""Translation layer: structured JSON instance + vocabulary spec -> prompt.

Also contains answer decoding and scoring, which depend on the vocabulary
(e.g. grid move tokens).
"""
from __future__ import annotations

import re

from .tasks import Instance, check_grid_moves, dijkstra_trace, solve_bricks, solve_grid

WORD_MOVES = {"up": "U", "down": "D", "left": "L", "right": "R",
              "north": "U", "south": "D", "west": "L", "east": "R"}
MOVE_WORD = {"U": "up", "D": "down", "L": "left", "R": "right"}

SYSTEM = ("You are a careful reasoner solving a structured {what} problem. "
          "Reason step by step using exactly the notation shown in the example, "
          "then give the final answer on its own last line in the form 'ANSWER: ...'.")
WHAT = {"grid": "grid-navigation", "graph": "shortest-path", "bricks": "brick-stacking"}


# =========================================================================== grid
def _coord(r, c, fmt):
    return {"(r,c)": f"({r},{c})", "rRcC": f"r{r}c{c}", "[r][c]": f"[{r}][{c}]"}[fmt]


def _move_tok(p):
    return {"U": p["move_up"], "D": p["move_down"], "L": p["move_left"], "R": p["move_right"]}


def grid_env(data, spec):
    grid, (sr, sc), (gr, gc) = data["grid"], data["start"], data["goal"]
    H, W = len(grid), len(grid[0])
    if spec["mode"] == "nl":
        walls = [f"row {r}, column {c}" for r in range(H) for c in range(W) if grid[r][c] == "#"]
        return (f"The grid has {H} rows (numbered 0 to {H-1} from the top) and {W} columns (numbered 0 to {W-1} from the left). "
                f"The following cells are blocked by walls: {'; '.join(walls) if walls else 'none'}. "
                f"You start at row {sr}, column {sc}. The goal is at row {gr}, column {gc}. "
                "Each step moves one cell up, down, left or right; you cannot enter a wall or leave the grid.")
    p = spec["params"]

    def cell(r, c):
        if (r, c) == (sr, sc):
            return p["start"]
        if (r, c) == (gr, gc):
            return p["goal"]
        return p["wall"] if grid[r][c] == "#" else p["free"]

    single = all(len(p[k]) == 1 for k in ("free", "wall", "start", "goal"))
    sep = "" if single else " "
    if p["layout"] == "matrix":
        rows = [sep.join(cell(r, c) for c in range(W)) for r in range(H)]
        return f"map {H}x{W} (row 0 = top, col 0 = left):\n" + "\n".join(rows)
    if p["layout"] == "indexed_rows":
        head = "    " + " ".join(f"c{c}" for c in range(W))
        rows = [f"r{r}: " + " ".join(cell(r, c).ljust(2) for c in range(W)) for r in range(H)]
        return "map:\n" + head + "\n" + "\n".join(rows)
    walls = " ".join(_coord(r, c, p["coord_fmt"]) for r in range(H) for c in range(W) if grid[r][c] == "#")
    return (f"size {H}x{W}; {p['wall']} = {walls or '∅'}; "
            f"{p['start']} = {_coord(sr, sc, p['coord_fmt'])}; {p['goal']} = {_coord(gr, gc, p['coord_fmt'])}")


def grid_question(data, spec):
    if spec["mode"] == "nl":
        return "Question: what is the shortest sequence of moves from the start to the goal?"
    p = spec["params"]
    return f"Question: shortest path {p['start']} {p['chain']} {p['goal']}?"


def grid_trace(inst: Instance, spec):
    sol = solve_grid(inst.data)
    path, moves = sol["path"], sol["moves"]
    if spec["mode"] == "nl":
        parts = [f"I start at row {path[0][0]}, column {path[0][1]}."]
        for (r, c), m in zip(path[1:], moves):
            parts.append(f"I move {MOVE_WORD[m]} to row {r}, column {c}, which is not a wall.")
        parts.append(f"I have reached the goal in {len(moves)} moves.")
        return " ".join(parts), " ".join(MOVE_WORD[m] for m in moves)
    p = spec["params"]
    mt = _move_tok(p)
    cf = p["coord_fmt"]
    ch = f" {p['chain']} "
    coords = [_coord(r, c, cf) for r, c in path]
    if p["trace"] == "coords":
        body = f"{p['start']}{coords[0]}" + ch + ch.join(coords[1:-1] + [f"{p['goal']}{coords[-1]}"]) if len(coords) > 1 else coords[0]
    elif p["trace"] == "moves":
        body = " ".join(mt[m] for m in moves) + f" ({len(moves)})"
    else:
        segs = [coords[0]]
        for co, m in zip(coords[1:], moves):
            segs.append(f"{mt[m]} {co}")
        body = ch.join(segs) + f" ({len(moves)})"
    return body, " ".join(mt[m] for m in moves)


def grid_legend(spec):
    p = spec["params"]
    return (f"Legend: {p['free']}=free {p['wall']}=wall {p['start']}=start {p['goal']}=goal; "
            f"moves {p['move_up']}=up {p['move_down']}=down {p['move_left']}=left {p['move_right']}=right; "
            f"{p['chain']}=then; coords as {p['coord_fmt']}.")


def grid_answer_hint(spec):
    if spec["mode"] == "nl":
        return "Give the answer as space-separated words from {up, down, left, right}."
    p = spec["params"]
    return f"Give the answer as space-separated move symbols from {{{p['move_up']} {p['move_down']} {p['move_left']} {p['move_right']}}}."


def grid_decode(text, inst, spec):
    table = dict(WORD_MOVES)
    canon_ok = True
    if spec["mode"] == "cos":
        mt = _move_tok(spec["params"])
        for canon, tok in mt.items():
            table[tok.lower()] = canon
        p = spec["params"]
        # canonical U/D/L/R are accepted as a fallback only if no slot reuses them with another meaning
        for slot in ("free", "wall", "start", "goal", "chain", "move_up", "move_down", "move_left", "move_right"):
            v = p[slot]
            if v.upper() in "UDLR" and len(v) == 1 and mt.get(v.upper()) != v:
                canon_ok = False
    if canon_ok:
        for c in "UDLR":
            table.setdefault(c.lower(), c)
    toks = sorted(table, key=len, reverse=True)
    moves = []
    for piece in re.split(r"[\s,;]+", text.strip()):
        if not piece:
            continue
        s = piece.lower()
        if s in table:
            moves.append(table[s])
            continue
        i, ok = 0, True
        while i < len(s):
            for t in toks:
                if s.startswith(t, i):
                    moves.append(table[t])
                    i += len(t)
                    break
            else:
                ok = False
                break
        if not ok:
            return None
    return moves


def grid_score(inst, parsed):
    if parsed is None:
        return {"correct": False, "valid": False}
    valid, reached = check_grid_moves(inst.data, parsed)
    return {"correct": bool(valid and reached and len(parsed) == inst.answer["length"]), "valid": valid and reached}


# =========================================================================== graph
def _graph_names(data, spec):
    nodes = data["nodes"]
    if spec["mode"] == "nl" or spec["params"]["node_names"] == "original":
        return {n: str(n) for n in nodes}
    if spec["params"]["node_names"] == "numbers":
        return {n: str(i) for i, n in enumerate(nodes)}
    import string
    if len(nodes) <= 26:
        return {n: string.ascii_uppercase[i] for i, n in enumerate(nodes)}
    return {n: f"n{i}" for i, n in enumerate(nodes)}


def _dist(x, k, fmt):
    return {"d(X)=k": f"d({x})={k}", "X:k": f"{x}:{k}", "X[k]": f"{x}[{k}]"}[fmt]


def _edge(u, v, w, p):
    op = p["edge_op"]
    if p["weight_fmt"] == "inline":
        return f"{u}-{w}{op}{v}"
    if p["weight_fmt"] == "suffix":
        return f"{u}{op}{v}:{w}"
    return f"{u}{op}{v}({w})"


def graph_env(data, spec):
    nm = _graph_names(data, spec)
    directed = data.get("directed", True)
    if spec["mode"] == "nl":
        if directed:
            s = " ".join(f"There is a one-way road from {u} to {v} of length {w}." for u, v, w in data["edges"])
        else:
            s = " ".join(f"There is a two-way road between {u} and {v} of length {w}." for u, v, w in data["edges"])
        return f"There are {len(data['nodes'])} cities: {', '.join(map(str, data['nodes']))}. " + s
    p = spec["params"]
    sep = f"{p['sep']} "
    kind = "directed" if directed else "undirected (edges work both ways)"
    if p["layout"] == "matrix" and len(data["nodes"]) <= 15:
        names = [nm[n] for n in data["nodes"]]
        idx = {n: i for i, n in enumerate(data["nodes"])}
        M = [["·"] * len(names) for _ in names]
        for u, v, w in data["edges"]:
            M[idx[u]][idx[v]] = str(w)
            if not directed:
                M[idx[v]][idx[u]] = str(w)
        width = max(2, max(len(x) for x in names) + 1)
        head = " " * width + "".join(n.rjust(width) for n in names)
        rows = [names[i].ljust(width) + "".join(x.rjust(width) for x in M[i]) for i in range(len(names))]
        return f"graph ({kind}) weight matrix row{p['edge_op']}col:\n" + head + "\n" + "\n".join(rows)
    if p["layout"] == "adjacency":
        adj: dict[str, list] = {n: [] for n in data["nodes"]}
        for u, v, w in data["edges"]:
            adj[u].append((v, w))
            if not directed:
                adj[v].append((u, w))
        lines = []
        for n in data["nodes"]:
            if adj[n]:
                lines.append(f"{nm[n]}{p['edge_op']} " + sep.join(
                    (f"{nm[v]}:{w}" if p["weight_fmt"] == "suffix" else f"{nm[v]}({w})" if p["weight_fmt"] == "paren" else f"{w}{nm[v]}")
                    for v, w in sorted(adj[n])))
        return f"graph ({kind}) adjacency:\n" + "\n".join(lines)
    return f"graph ({kind}): " + sep.join(_edge(nm[u], nm[v], w, p) for u, v, w in data["edges"])


def graph_question(data, spec):
    nm = _graph_names(data, spec)
    s, t = nm[data["source"]], nm[data["target"]]
    if spec["mode"] == "nl":
        return f"Question: what is the length of the shortest route from {s} to {t}?"
    return f"Question: min cost {s} {spec['params']['chain']} {t}?"


def graph_trace(inst, spec):
    tr = dijkstra_trace(inst.data)
    nm = _graph_names(inst.data, spec)
    if spec["mode"] == "nl":
        parts = [f"I start at {nm[inst.data['source']]} with distance 0."]
        for st in tr["steps"]:
            if st["node"] == inst.data["target"]:
                parts.append(f"The closest unvisited city is now {nm[st['node']]} at distance {st['dist']}, which is the destination.")
                break
            if st["relax"]:
                r = ", ".join(f"{nm[v]} with total distance {d}" for v, d in st["relax"])
                parts.append(f"From {nm[st['node']]} (distance {st['dist']}) I can improve {r}.")
            else:
                parts.append(f"From {nm[st['node']]} (distance {st['dist']}) nothing improves.")
        route = " to ".join(nm[x] for x in tr["path"])
        parts.append(f"The shortest route is {route} with length {tr['cost']}.")
        return " ".join(parts), str(tr["cost"])
    p = spec["params"]
    df, ch, sep = p["dist_fmt"], f" {p['chain']} ", f"{p['sep']} "
    lines = []
    if p["trace"] == "dijkstra_table":
        for st in tr["steps"]:
            rel = sep.join(_dist(nm[v], d, df) for v, d in st["relax"]) or "∅"
            lines.append(f"pop {_dist(nm[st['node']], st['dist'], df)} | {rel}")
    elif p["trace"] == "frontier_sets":
        for st in tr["steps"]:
            fr = sep.join(_dist(nm[v], d, df) for v, d in st["frontier"].items()) or "∅"
            lines.append(f"✓{nm[st['node']]} {{{fr}}}")
    path = ch.join(_dist(nm[x], c, df) for x, c in zip(tr["path"], tr["cum"]))
    lines.append(path)
    return "\n".join(lines), str(tr["cost"])


def graph_legend(spec):
    p = spec["params"]
    ex = _edge("A", "B", 3, p)
    return (f"Legend: {ex} = edge A{p['edge_op']}B with weight 3; {_dist('X', 5, p['dist_fmt'])} = best distance to X is 5; "
            f"{p['chain']} = next step on the route; {p['sep']} separates items; pop = settle the closest node; ✓ = settled.")


def graph_answer_hint(spec):
    return "Give the answer as a single integer (the total route length)."


def graph_decode(text, inst, spec):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def graph_score(inst, parsed):
    ok = parsed is not None and parsed == inst.answer["cost"]
    return {"correct": ok, "valid": parsed is not None}


# =========================================================================== bricks
def bricks_env(data, spec):
    if spec["mode"] == "nl":
        s = []
        for up, low in data["on"]:
            s.append(f"Brick {up} is on the table." if low == "table" else f"Brick {up} is on top of brick {low}.")
        return " ".join(s)
    p = spec["params"]
    tbl = lambda x: p["table"] if x == "table" else x  # noqa: E731
    sep = f"{p['sep']} "
    if p["layout"] == "pairs":
        return "bricks: " + sep.join(f"{u}{p['on_op']}{tbl(l)}" for u, l in data["on"])
    from .tasks import bricks_stacks
    stacks = [p["on_op"].join(list(reversed(st)) + [p["table"]]) for st in bricks_stacks(data)]
    return "stacks (top first): " + sep.join(stacks)


def bricks_question(data, spec):
    if spec["mode"] == "nl":
        return f"Question: to take brick {data['target']}, which bricks must be removed, in order (finishing with {data['target']} itself)?"
    return f"Question: get {data['target']}? removal order."


def bricks_trace(inst, spec):
    sol = solve_bricks(inst.data)
    st, seq = sol["stack"], sol["sequence"]
    tgt = inst.data["target"]
    if spec["mode"] == "nl":
        parts = [f"I need brick {tgt}."]
        top_down = list(reversed(st))
        for i, b in enumerate(top_down):
            below = top_down[i + 1] if i + 1 < len(top_down) else "the table"
            parts.append(f"Brick {b} is on {('brick ' + below) if below != 'the table' else below}.")
        if len(seq) > 1:
            parts.append(f"Nothing is on top of brick {seq[0]}, so I remove the bricks from the top: {', '.join(seq[:-1])}, and then I can take {tgt}.")
        else:
            parts.append(f"Nothing is on top of brick {tgt}, so I can take it directly.")
        return " ".join(parts), ", ".join(seq)
    p = spec["params"]
    chain = f" {p['chain']} "
    stack_str = p["on_op"].join(list(reversed(st)) + [p["table"]])
    if p["trace"] == "stack_rebuild":
        links = f"{p['sep']} ".join(f"{a}{p['on_op']}{b}" for a, b in zip(reversed(st), list(reversed(st))[1:] + [p["table"]]))
        body = f"{links}{chain}{stack_str}{chain}" + f"{p['sep']} ".join(seq)
    else:
        body = f"{tgt} ∈ {stack_str}{chain}" + f"{p['sep']} ".join(seq)
    return body, ", ".join(seq)


def bricks_legend(spec):
    p = spec["params"]
    return (f"Legend: X{p['on_op']}Y = brick X is directly on top of Y; {p['table']} = the table; "
            f"{p['sep']} separates items; {p['chain']} = therefore.")


def bricks_answer_hint(spec):
    return "Give the answer as comma-separated brick names, top-most first, ending with the target brick."


def bricks_decode(text, inst, spec):
    names = set(inst.data["bricks"])
    text = re.split(r"[.(\[]|\s-\s|\bbecause\b|\bsince\b", text)[0]  # drop trailing explanation
    pieces = [t.strip(" *`'\"") for t in re.split(r"[,;|\s]+", text) if t.strip()]
    if any(len(t) > 1 and t.isalpha() and t not in names for t in pieces):  # prose on the answer line
        pieces = [t for t in re.split(r"\s*,\s*", text.strip()) if t.strip() in names]
    # keep every name-like token (short alphanumeric) so foreign names, e.g. copied from the example, make it wrong
    out = [t.strip() for t in pieces if t.strip() in names or (t.strip().isalnum() and len(t.strip()) <= 3 and t.strip()[0].isupper())]
    return out if any(t in names for t in out) else None


def bricks_score(inst, parsed):
    return {"correct": parsed == inst.answer["sequence"], "valid": parsed is not None}


# =========================================================================== dispatch
R = {
    "grid": (grid_env, grid_question, grid_trace, grid_legend, grid_answer_hint, grid_decode, grid_score),
    "graph": (graph_env, graph_question, graph_trace, graph_legend, graph_answer_hint, graph_decode, graph_score),
    "bricks": (bricks_env, bricks_question, bricks_trace, bricks_legend, bricks_answer_hint, bricks_decode, bricks_score),
}


def render_block(inst: Instance, spec: dict) -> str:
    env, q, *_ = R[inst.domain]
    return env(inst.data, spec) + "\n" + q(inst.data, spec)


def render_prompt(inst: Instance, spec: dict, exemplar: Instance) -> tuple[str, str]:
    env, q, trace, legend, hint, _, _ = R[inst.domain]
    parts = []
    if spec["mode"] == "cos" and spec["params"].get("legend", True):
        parts.append(legend(spec))
    ex_trace, ex_ans = trace(exemplar, spec)
    parts.append("Example:\n" + env(exemplar.data, spec) + "\n" + q(exemplar.data, spec) +
                 "\nReasoning:\n" + ex_trace + f"\nANSWER: {ex_ans}")
    parts.append("Now solve:\n" + env(inst.data, spec) + "\n" + q(inst.data, spec) + "\n" + hint(spec) +
                 " Your last line must be 'ANSWER: ' followed only by the answer, with no explanation.\nReasoning:")
    return SYSTEM.format(what=WHAT[inst.domain]), "\n\n".join(parts)


def gold_output(inst: Instance, spec: dict) -> str:
    """What a perfect model would write (used by tests and the simulator)."""
    t, a = R[inst.domain][2](inst, spec)
    return t + f"\nANSWER: {a}"


def extract_answer_line(text: str) -> tuple[str | None, bool]:
    m = list(re.finditer(r"ANSWER\s*[:：]\s*(.+)", text, flags=re.IGNORECASE))
    if m:
        return m[-1].group(1).strip().strip("`*").strip(), True
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return (lines[-1].strip() if lines else None), False


def grade(inst: Instance, spec: dict, output: str) -> dict:
    line, found = extract_answer_line(output or "")
    *_, decode, score = R[inst.domain]
    parsed = decode(line, inst, spec) if line else None
    res = score(inst, parsed)
    res.update({"answer_line_found": found, "parsed": parsed})
    return res


def est_tokens(text: str) -> int:
    """Rough tokenizer-free estimate: ~4 chars per token for ASCII, ~1 token per non-ASCII char."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return int(ascii_chars / 4 + (len(text) - ascii_chars) + 0.5)
