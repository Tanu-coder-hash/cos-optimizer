"""Model backends with on-disk caching, retries and usage accounting.

Backends
--------
* ``anthropic``          - Claude models via the Anthropic API (ANTHROPIC_API_KEY)
* ``openai``             - OpenAI API (OPENAI_API_KEY)
* ``openai_compatible``  - any OpenAI-compatible server: Ollama, vLLM, LM Studio,
                           OpenRouter, Groq, Together ... (set base_url)
* ``simulated``          - offline stand-in for wiring tests and UI demos. It does
                           NOT call a model; its numbers are synthetic and must not
                           be reported as results.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    cached: bool = False
    latency: float = 0.0
    error: str | None = None


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, c: Completion):
        with self.lock:
            self.calls += 1
            self.cached += int(c.cached)
            self.in_tokens += c.in_tokens
            self.out_tokens += c.out_tokens
            self.errors += int(c.error is not None)

    def as_dict(self):
        return {"calls": self.calls, "cached": self.cached, "in_tokens": self.in_tokens,
                "out_tokens": self.out_tokens, "errors": self.errors}


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self.db.commit()


class LLM:
    backend = "base"
    is_simulated = False
    default_retries = 4

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024,
                 cache_path: str | None = ".cosopt_cache/llm.sqlite", price_in: float = 0.0, price_out: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = Cache(cache_path) if cache_path else None
        self.usage = Usage()
        self.price_in, self.price_out = price_in, price_out  # USD per million tokens

    @property
    def label(self):
        return f"{self.backend}:{self.model}"

    def cost(self) -> float:
        return (self.usage.in_tokens * self.price_in + self.usage.out_tokens * self.price_out) / 1e6

    # subclasses implement _call
    def _call(self, system: str, user: str, max_tokens: int, temperature: float) -> Completion:
        raise NotImplementedError

    def complete(self, system: str, user: str, max_tokens: int | None = None, temperature: float | None = None,
                 retries: int | None = None) -> Completion:
        retries = retries or self.default_retries
        mt = max_tokens or self.max_tokens
        temp = self.temperature if temperature is None else temperature
        k = hashlib.sha256(json.dumps([self.label, system, user, mt, temp]).encode()).hexdigest()
        if self.cache and temp == 0.0:
            hit = self.cache.get(k)
            if hit:
                c = Completion(**hit)
                c.cached = True
                self.usage.add(c)
                return c
        err = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                c = self._call(system, user, mt, temp)
                c.latency = time.time() - t0
                if self.cache and temp == 0.0:
                    self.cache.put(k, {"text": c.text, "in_tokens": c.in_tokens, "out_tokens": c.out_tokens})
                self.usage.add(c)
                return c
            except Exception as e:  # network / rate limit / server errors
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(30, 2 ** attempt + random.random()))
        c = Completion(text="", error=err)
        self.usage.add(c)
        return c

    # The evaluator calls solve(); real models just complete the prompt.
    def solve(self, system: str, user: str, ctx: dict | None = None) -> Completion:
        return self.complete(system, user)


class AnthropicLLM(LLM):
    backend = "anthropic"

    def __init__(self, model="claude-sonnet-5-5", api_key=None, timeout=300, **kw):
        super().__init__(model, **kw)
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout, max_retries=0)
        self._send_temperature = True

    def _call(self, system, user, max_tokens, temperature):
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if self._send_temperature:
            kwargs["temperature"] = temperature
        try:
            r = self.client.messages.create(**kwargs)
        except Exception as e:
            if "temperature" in str(e).lower() and self._send_temperature:
                self._send_temperature = False  # some models fix sampling params
                kwargs.pop("temperature", None)
                r = self.client.messages.create(**kwargs)
            else:
                raise
        text = "".join(getattr(b, "text", "") for b in r.content)
        return Completion(text=text, in_tokens=r.usage.input_tokens, out_tokens=r.usage.output_tokens)


class OpenAILLM(LLM):
    backend = "openai"

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None, timeout=300, **kw):
        super().__init__(model, **kw)
        from openai import OpenAI
        if base_url:
            self.backend = "openai_compatible"
            self.default_retries = 2  # a local server that fails twice is down or overloaded: don't stall for minutes
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed", base_url=base_url or None,
                             timeout=timeout, max_retries=0)

    def _call(self, system, user, max_tokens, temperature):
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        u = r.usage
        return Completion(text=r.choices[0].message.content or "",
                          in_tokens=getattr(u, "prompt_tokens", 0) or 0, out_tokens=getattr(u, "completion_tokens", 0) or 0)


class SimulatedLLM(LLM):
    """Deterministic fake solver for offline smoke tests.

    It reads the gold solution from ``ctx`` and corrupts it with a probability
    that depends on hand-set heuristics (prompt length, notation features). This
    exists ONLY so the pipeline and UI can be exercised without API keys; any
    'finding' it produces is an artefact of these heuristics.
    """
    backend = "simulated"
    is_simulated = True

    def __init__(self, model="offline-demo", **kw):
        kw["cache_path"] = None
        super().__init__(model, **kw)

    def solve(self, system, user, ctx=None):
        from .render import est_tokens, gold_output
        inst, spec = ctx["instance"], ctx["spec"]
        rng = random.Random(f"{inst.id}|{json.dumps(spec.get('params', {}), sort_keys=True)}|{spec['mode']}")
        p = 0.55 if spec["mode"] == "nl" else 0.72
        prm = spec.get("params", {})
        p += 0.06 * bool(prm.get("legend"))
        p += 0.05 * (prm.get("trace") in ("coords+moves", "dijkstra_table", "stack_rebuild"))
        p += 0.04 * (prm.get("layout") in ("matrix", "adjacency", "stacks"))
        p -= 0.04 * (prm.get("node_names") == "numbers")
        p -= 0.03 * any(len(str(v)) > 1 and not str(v).isascii() for v in prm.values())
        p -= 0.00004 * est_tokens(user)
        p = max(0.05, min(0.97, p))
        text = gold_output(inst, spec)
        if rng.random() > p:
            lines = text.splitlines()
            if inst.domain == "grid":
                lines[-1] = "ANSWER: " + " ".join(lines[-1].split(":", 1)[1].split()[:-1])  # stops one short
            elif inst.domain == "graph":
                lines[-1] = f"ANSWER: {inst.answer['cost'] + rng.choice([-2, -1, 1, 2, 3])}"
            else:
                lines[-1] = "ANSWER: " + ", ".join(reversed(inst.answer["sequence"])) + ", " + inst.data["bricks"][0]
            text = "\n".join(lines)
        c = Completion(text=text, in_tokens=est_tokens(system + user), out_tokens=est_tokens(text))
        self.usage.add(c)
        return c

    def _call(self, system, user, max_tokens, temperature):
        return Completion(text="")


def make_llm(backend: str, model: str | None = None, api_key: str | None = None, base_url: str | None = None, **kw) -> LLM:
    backend = backend.lower()
    if backend == "anthropic":
        return AnthropicLLM(model=model or "claude-sonnet-5-5", api_key=api_key, **kw)
    if backend == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini", api_key=api_key, **kw)
    if backend in ("openai_compatible", "ollama"):
        return OpenAILLM(model=model or "llama3.1", api_key=api_key, base_url=base_url or "http://localhost:11434/v1", **kw)
    if backend == "simulated":
        return SimulatedLLM(**{k: v for k, v in kw.items() if k in ("temperature", "max_tokens")})
    raise ValueError(f"unknown backend {backend!r}")
