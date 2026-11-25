import json
import random

import pytest

from cosopt import tasks, vocab
from cosopt.compiler import TranslationLayer, compile_layer
from cosopt.render import gold_output, grade, render_prompt
from cosopt.schema import load_instances


@pytest.mark.parametrize("domain", tasks.DOMAINS)
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_solvable(domain, difficulty):
    for inst in tasks.make_dataset(domain, 5, difficulty, seed=1):
        assert inst.answer


@pytest.mark.parametrize("domain", tasks.DOMAINS)
def test_gold_output_grades_correct_under_many_vocabs(domain):
    rng = random.Random(0)
    insts = tasks.make_dataset(domain, 6, "medium", seed=2)
    ex = tasks.make_exemplar(domain, insts[0])
    specs = [vocab.nl_baseline(domain), vocab.default_cos(domain)] + [vocab.random_spec(domain, rng) for _ in range(40)]
    for spec in specs:
        assert vocab.validate(spec)[0], vocab.validate(spec)
        for inst in insts:
            s, u = render_prompt(inst, spec, ex)
            assert "ANSWER" in u and len(u) > 50
            g = grade(inst, spec, gold_output(inst, spec))
            assert g["correct"], (spec, inst.id, gold_output(inst, spec), g)


@pytest.mark.parametrize("domain", tasks.DOMAINS)
def test_wrong_answers_rejected(domain):
    inst = tasks.make_dataset(domain, 1, "medium", seed=3)[0]
    spec = vocab.default_cos(domain)
    assert not grade(inst, spec, "ANSWER: 99999 Z Q")["correct"]
    assert not grade(inst, spec, "")["correct"]


def test_validation_catches_collisions():
    s = vocab.default_cos("grid")
    s["params"]["wall"] = s["params"]["free"]
    assert not vocab.validate(s)[0]
    s = vocab.default_cos("graph")
    s["params"]["edge_op"] = "has a space"
    assert not vocab.validate(s)[0]


def test_schema_formats():
    g, rep = load_instances(json.dumps({"grid": ["S..#", ".#..", "...G"]}))
    assert rep["domain"] == "grid" and g[0].answer["length"] == 5
    g, rep = load_instances(json.dumps({"grid": [[0, 0, 1], [1, 0, 0]], "start": [0, 0], "goal": [1, 2]}))
    assert g[0].answer["length"] == 3
    nl = {"nodes": [{"id": i} for i in "abcde"],
          "links": [{"source": "a", "target": "b", "weight": 2}, {"source": "b", "target": "c", "weight": 2},
                    {"source": "a", "target": "c", "weight": 9}, {"source": "c", "target": "d", "weight": 1},
                    {"source": "d", "target": "e", "weight": 1}]}
    g, rep = load_instances(json.dumps(nl), queries_per_structure=5)
    assert rep["domain"] == "graph" and len(g) >= 1
    g, rep = load_instances(json.dumps({"adjacency": {"A": {"B": 1, "C": 5}, "B": {"C": 1}, "C": {"D": 1}}, "source": "A", "target": "D"}))
    assert g[0].answer["cost"] == 3
    g, rep = load_instances(json.dumps({"stacks": [["A", "B", "C"], ["D", "E"]], "target": "B"}))
    assert g[0].answer["sequence"] == ["C", "B"]
    lines = "\n".join(json.dumps({"on": [["X", "Y"], ["Y", "table"], ["Z", "table"]], "target": "Y"}) for _ in range(3))
    g, rep = load_instances(lines)
    assert len(g) == 3 and g[0].answer["sequence"] == ["X", "Y"]


def test_compiled_layer_roundtrip(tmp_path):
    spec = vocab.default_cos("graph")
    ex = tasks.make_exemplar("graph")
    layer = compile_layer(spec, ex)
    p = tmp_path / "l.json"
    p.write_text(json.dumps(layer, ensure_ascii=False))
    tl = TranslationLayer.load(str(p))
    out = tl.translate({"edges": [["A", "B", 1], ["B", "C", 2], ["A", "C", 5], ["C", "D", 1]], "source": "A", "target": "D"})
    s, u, inst = out[0]
    assert inst.answer["cost"] == 4
    assert tl.grade(inst, "...\nANSWER: 4")["correct"]


def test_optimizer_simulated(tmp_path):
    from cosopt.judge import HeuristicJudge
    from cosopt.llm import make_llm
    from cosopt.optimizer import OptConfig, Optimizer
    from cosopt.proposer import RandomProposer
    for domain in tasks.DOMAINS:
        dev = tasks.make_dataset(domain, 10, "easy", seed=0)
        test = tasks.make_dataset(domain, 12, "easy", seed=9)
        cfg = OptConfig(domain=domain, generations=2, population=4, children=3, screen_size=4, concurrency=2)
        res = Optimizer(make_llm("simulated"), HeuristicJudge(), RandomProposer(0), dev, test,
                        tasks.make_exemplar(domain), {"domain": domain}, cfg, out_dir=str(tmp_path)).run()
        assert set(res["test"]) == {"cot", "default_cos", "optimized"}
        assert (tmp_path / res["run_dir"].split("/")[-1] / "report.md").exists()
