"""Compile an optimised vocabulary into a reusable translation layer.

    layer = TranslationLayer.load("compiled_layer.json")
    for system, user, inst in layer.translate(any_supported_json):
        text = my_llm(system, user)
        print(layer.grade(inst, text))
"""
from __future__ import annotations

import json
import time

from .render import grade, render_prompt
from .schema import normalize
from .tasks import Instance, solve

VERSION = "0.1"


def compile_layer(spec: dict, exemplar: Instance, metrics: dict | None = None) -> dict:
    sys_p, user_p = render_prompt(exemplar, spec, exemplar)
    return {"format": "cosopt-translation-layer", "version": VERSION, "compiled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "domain": spec["domain"], "spec": spec, "exemplar": exemplar.to_dict(), "metrics": metrics or {},
            "preview": {"system": sys_p, "user": user_p}}


class TranslationLayer:
    def __init__(self, layer: dict):
        if layer.get("format") != "cosopt-translation-layer":
            raise ValueError("not a cosopt translation layer")
        self.layer = layer
        self.spec = layer["spec"]
        self.domain = layer["domain"]
        self.exemplar = Instance.from_dict(layer["exemplar"])

    @classmethod
    def load(cls, path: str) -> "TranslationLayer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def translate(self, raw, queries: int = 5) -> list[tuple[str, str, Instance]]:
        if isinstance(raw, str):
            raw = json.loads(raw)
        datas, rep = normalize(raw, queries)
        if rep["domain"] != self.domain:
            raise ValueError(f"layer compiled for {self.domain!r} but input looks like {rep['domain']!r}")
        out = []
        for i, d in enumerate(datas):
            inst = Instance(id=f"q{i}", domain=self.domain, data=d, answer=solve(self.domain, d))
            s, u = render_prompt(inst, self.spec, self.exemplar)
            out.append((s, u, inst))
        return out

    def grade(self, inst: Instance, output: str) -> dict:
        return grade(inst, self.spec, output)
