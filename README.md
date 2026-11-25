# CoS Optimizer

**Automated prompt optimisation for Chain-of-Symbol (CoS) vs. Chain-of-Thought (CoT) in spatial and graph reasoning.**

This tool takes structured inputs (grids, graphs, stacking relations; built-in or your own JSON), works out their structure, searches over symbolic vocabularies using an LLM proposer and an LLM-as-a-judge feedback loop, and compiles the best vocabulary into a reusable **translation layer**. That layer turns any new JSON of the same kind into an optimised symbolic prompt.

---

## 1. Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

On Windows you can double-click `run_windows.bat`. On macOS or Linux, run `./run.sh`.

1. **Sidebar:** choose a backend.
   * *Simulated*: works offline with no API key. Use it only to learn the interface, because its numbers are synthetic.
   * *Anthropic / OpenAI*: paste an API key or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.
   * *OpenAI-compatible*: Ollama (`ollama serve`, base URL `http://localhost:11434/v1`), vLLM, LM Studio, OpenRouter, Groq and similar servers.
2. **Data tab:** generate a benchmark, or upload your own structures.
3. **Vocabulary lab:** design a vocabulary by hand, check the exact prompt, test it, and add it to the optimiser's seeds.
4. **Optimise:** run the search and watch the leaderboard update live.
5. **Results:** see held-out test accuracy with 95% CIs, McNemar p-values, accuracy against token cost, and the winning vocabulary. You can download `compiled_layer.json`, `results.json` and `report.md`.
6. **Translate:** apply a compiled layer to new JSON.

### Command line (for batch experiments)

```bash
python -m cosopt optimize --domain grid  --backend anthropic --model claude-sonnet-5-5 --generations 5
python -m cosopt optimize --domain graph --backend ollama --model qwen2.5:7b --meta-backend anthropic --meta-model claude-opus-5-5
python -m cosopt optimize --data examples/my_graph.json --backend openai --model gpt-4o-mini
python -m cosopt optimize --domain bricks --backend anthropic --random-proposer          # ablation: no LLM proposer
python -m cosopt translate --layer runs/<run>/compiled_layer.json --input examples/my_graph.json
python -m pytest -q
```

Every run writes `runs/<timestamp>-<domain>-<model>/` containing `config.json`, `calls.jsonl` (every prompt outcome), `candidates.jsonl`, `results.json`, `report.md` and `compiled_layer.json`. LLM calls at temperature 0 are cached in `.cosopt_cache/`, so a re-run with the same inputs costs nothing.

---

## 2. How it works

```
 structured JSON ──► schema inference ──► canonical task data + exact solver (ground truth)
                                              │
       ┌──────────────────────────────────────┘
       ▼
 seed vocabularies: CoT (NL) · hand-designed CoS · random CoS · your own
       │
       ▼   repeat for G generations
 ┌──────────────────────────────────────────────────────────────────────┐
 │ render prompt (1-shot exemplar written in the candidate's notation)  │
 │ solver LLM answers ──► exact verifier ──► accuracy, token cost       │
 │ LLM judge scores sampled traces (symbol consistency, step validity,  │
 │   faithfulness, concision) + critique + suggested notation change    │
 │ LLM proposer reads leaderboard + critiques ──► new vocabularies      │
 │ successive halving: screen on a subset, promote the best to full dev │
 └──────────────────────────────────────────────────────────────────────┘
       ▼
 best vocabulary ──► held-out test vs CoT and hand-designed CoS
                     (bootstrap 95% CI, exact McNemar) ──► compiled translation layer
```

**Fitness** = accuracy + w_judge · judge_score − w_tokens · (relative token cost). The defaults are 0.10 and 0.05. Accuracy always comes from exact verifiers. The judge only shapes the search, so it can't inflate the reported numbers.

**Search space** (`cosopt/vocab.py`). Each family has *categorical slots* (layout, trace style, coordinate or distance format, legend on or off) and *token slots* (any 1–4 character symbol). The LLM proposer can invent new tokens. The validator rejects collisions, whitespace and tokens that clash with entity names or numbers.

| Family | Task | Verifier |
|---|---|---|
| `grid` | shortest obstacle-free path; answer is a move sequence | simulate the moves, check it is valid, reaches the goal and has optimal length |
| `graph` | weighted shortest-path cost, directed or undirected | Dijkstra |
| `bricks` | Brick-World: which bricks to remove, in order, to take a target | stack reconstruction |

**Accepted upload formats** (`cosopt/schema.py`):
* grid matrices: strings or 0/1 lists, with `S`/`G` cells or `start`/`goal` keys
* edge lists, networkx node-link JSON and adjacency maps
* stacking relations (`on`) or `stacks`
* JSONL with one structure per line

If a structure has no query, queries are generated for it.

---

## 3. Research framing

**Problem.** CoS, which uses compact symbols in place of natural-language steps, beats CoT on spatial planning. Hu et al. (2023/2024) report up to +60.8 points on Brick World for ChatGPT, using roughly a third of the reasoning tokens. Their symbol sets were designed by hand, though, and hand design does not carry over reliably to new structural domains.

**Gap.** Related work compares *fixed* graph encodings (for example, human-interpretable vs. opaque symbolic encodings, ACL Findings 2026) or *selects* among a few symbolic languages per problem (Cheng, AAAI 2026). We did not find an automated, closed-loop optimiser that *generates* symbol vocabularies from the input structure using judge feedback. Before you commit to this claim, do a systematic literature search: arXiv plus ACL/NeurIPS/ICLR/COLM 2025–26, searching "symbolic prompt optimisation", "graph encoding search" and "representation search LLM".

**Research questions**
1. RQ1: Does an optimised CoS vocabulary beat (a) CoT and (b) hand-designed CoS on held-out instances, for each model?
2. RQ2: Does the LLM proposer beat random evolutionary search under the same call budget? Run with `--random-proposer`.
3. RQ3: Does the judge signal help? Set `--judge-samples 0` or vary `w_judge`.
4. RQ4: Do optimised vocabularies transfer across models (frontier vs. small open models) and across difficulty levels?
5. RQ5: Which vocabulary features matter? Analyse `candidates.jsonl`, for example with a regression of accuracy on slot values.

**Suggested experiment grid:** 3 families × 3 difficulties × at least 3 models (one frontier API model, one mid-size model, one small open model through Ollama) × 3 seeds × {full system, random proposer, no judge}. Report accuracy with CI, tokens, and McNemar p with a Holm correction across comparisons.

**Threats to validity**
* *Dev overfitting:* that is why the test split is held out. Report test numbers only.
* *Exemplar confound:* every condition gets one exemplar of the same structure.
* *Judge bias:* report the judge-free ablation.
* *Data contamination:* generated instances are fresh.
* *Simulated backend:* exists only for software testing. Never report its numbers.

---

## 4. How long does it take?

| Phase | Effort (one person) |
|---|---|
| Framework and interface (this repo) | done |
| Literature review and sharpening the gap claim | 1–2 weeks |
| Pilot runs on one family and one model; tune budgets and weights | 3–5 days |
| Full experiment grid (mostly waiting on API calls) | 1–2 weeks |
| Ablations, analysis, figures | 1–2 weeks |
| Paper writing (workshop: 4–6 pages; main conference: 8–9) | 2–3 weeks |
| **Total** | **about 6–10 weeks to a workshop-ready paper** |

**Compute per run** at the defaults (24 dev / 60 test, 6 seeds + 4 generations × 4 proposals): up to about 600 solver calls plus about 50 judge or proposer calls. That is about 0.5–1.5M tokens: a few dollars on a mid-tier API model, and free but slower on a local Ollama model. Wall-clock time is about 5–30 minutes depending on rate limits and the `Parallel requests` setting. The full grid above is roughly 150–250 runs.

---

## 5. Code map

```
app.py               Streamlit interface
cosopt/tasks.py      task generators + exact solvers
cosopt/schema.py     schema inference for arbitrary JSON
cosopt/vocab.py      search space, validation, mutation/crossover
cosopt/render.py     translation layer: JSON + vocabulary -> prompt; answer decoding + grading
cosopt/llm.py        Anthropic / OpenAI / OpenAI-compatible / simulated backends, cache, retries
cosopt/judge.py      LLM-as-a-judge (+ offline heuristic)
cosopt/proposer.py   LLM proposer (OPRO-style) + random evolutionary proposer
cosopt/optimizer.py  search loop, successive halving, test evaluation
cosopt/stats.py      bootstrap CIs, exact McNemar
cosopt/compiler.py   compiled translation layer (load + translate + grade)
cosopt/report.py     markdown report
tests/               pytest suite
```

## References
* Hu, Lu, Zhang, Song, Lam, Zhang. *Chain-of-Symbol Prompting for Spatial Reasoning in Large Language Models.* arXiv:2305.10276 (COLM 2024).
* Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
* Yang et al. *Large Language Models as Optimizers (OPRO).* ICLR 2024.
* Zhou et al. *Large Language Models Are Human-Level Prompt Engineers (APE).* ICLR 2023.
* Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
* *Colorful Talks with Graphs: Human-Interpretable Graph Encodings for LLMs.* arXiv:2602.10386 (Findings of ACL 2026).
* Cheng. *Empowering LLMs with Symbolic Representation and Reasoning.* AAAI 2026.
