---
title: Skill Invocation Environment
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Skill Invocation Environment

An OpenEnv RL environment that trains LLMs to make better decisions about **when to invoke procedural knowledge (skills)** during task-solving.

## Why This Matters

SkillsBench showed that AI agents fail to invoke available skills ~56% of the time, even when skills would significantly help. This environment creates a training ground for this specific problem.

### When Skills Are Irreplaceable

Skills are essential when the task requires knowledge that:
1. **Cannot be derived from general training data** (e.g., proprietary API authentication protocols)
2. **Has precise, non-obvious specifications** (e.g., binary format byte layouts, exact CLI commands)
3. **Would be impossible to guess correctly** (e.g., specific error code formats, deployment phase configurations)

## Context Cost Model

Skills aren't free — each loaded skill consumes context budget.
The environment rewards precision: agents that load only the skills
they need get higher rewards than agents that load everything.

### Actions

- `load(skill_id)` — Load full skill content into context (costs budget)
- `unload(skill_id)` — Remove skill from context (frees budget)
- `submit(answer)` — Submit solution (reward computed on loaded state at submit time)

The skill catalog (short descriptions) is returned in every observation, so agents always know what's available. The unload mechanic is key: agents can load a skill to read it, decide it's not useful, and unload it before submitting to avoid the bloat penalty.

### Reward Function

```
correctness  = 0.6  if answer is correct, else 0.0
precision    = 0.3 × (relevant loaded / total loaded)
recall       = 0.1 × (relevant loaded / total relevant)
bloat        = -0.15 per unnecessary skill loaded at submit time
total        = max(correctness + precision + recall + bloat, -1.0)
```

| Scenario | Correct? | Loaded | Relevant | Reward |
|----------|----------|--------|----------|--------|
| Right skill, correct answer | Yes | {A} | {A} | **1.0** |
| Right skill + 1 distractor | Yes | {A,B} | {A} | **0.7** |
| All 5 loaded, correct | Yes | {A,B,C,D,E} | {A} | **0.16** |
| No skills loaded, correct | Yes | {} | {A} | **0.6** |
| Right skill, wrong answer | No | {A} | {A} | **0.4** |

**Best policy: load exactly the right skill(s), solve correctly → 1.0**

## Quick Start

### Install

```bash
pip install -e .
```

### Run Locally (Direct)

```python
from skill_invocation_env.models import SkillInvocationAction
from skill_invocation_env.server.skill_invocation_env_environment import SkillInvocationEnvironment

env = SkillInvocationEnvironment()
obs = env.reset(seed=42)

print(f"Task: {obs.task_description}")
print(f"Skills: {[s['name'] for s in obs.skill_catalog]}")

# Load a skill (costs context)
obs = env.step(SkillInvocationAction(action_type="load", skill_id=obs.skill_catalog[0]["id"]))
print(f"Skill content: {obs.skill_content[:200]}...")
print(f"Context: {obs.context_budget_used}/{obs.context_budget_total}")

# Unload if not needed
obs = env.step(SkillInvocationAction(action_type="unload", skill_id=obs.loaded_skills[0]))

# Submit answer
obs = env.step(SkillInvocationAction(action_type="submit", answer="your solution here"))
print(f"Reward: {obs.reward}, Done: {obs.done}")
```

### Run Server

```bash
cd skill_invocation_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use Client

```python
from skill_invocation_env import SkillInvocationEnv, SkillInvocationAction

with SkillInvocationEnv(base_url="http://localhost:8000") as client:
    result = client.reset()
    print(f"Task: {result.observation.task_description}")

    # Load a skill
    skill_id = result.observation.skill_catalog[0]["id"]
    result = client.step(SkillInvocationAction(action_type="load", skill_id=skill_id))

    # Submit
    result = client.step(SkillInvocationAction(action_type="submit", answer="solution"))
    print(f"Reward: {result.reward}")
```

### Docker

```bash
docker build -t skill-invocation-env -f server/Dockerfile .
docker run -p 8000:8000 skill-invocation-env
```

## Task Domains

The environment includes 13 tasks (10 synthetic + 3 from SkillsBench) across 9 domains,
each with 5-8 skills in the catalog (1-2 relevant + 4-6 distractors):

| Domain | Skills | Tasks | Difficulty |
|--------|--------|-------|------------|
| Zephyr-3 API | Auth, Rate Limiting, Webhooks | 1 | Easy |
| NovaBin Format | File Spec, Compression | 2 | Easy, Medium |
| HelixLang | Error Handling, Modules, Concurrency | 1 | Easy |
| ArcDeploy | Canary Rollout, Service Mesh, Monitoring | 1 | Easy |
| CrystalQL | Temporal Queries, Index Optimization | 1 | Easy |
| VaultSync | Secret Rotation, Access Policies | 1 | Medium |
| FluxStream | Event Processing, Connectors, Schema | 1 | Medium |
| Cross-domain | CrystalQL + VaultSync | 1 | Hard |
| Cross-domain | ArcDeploy + FluxStream | 1 | Hard |
| Flood Detection* | Flood Detection, USGS Data, NWS Thresholds | 1 | Easy |
| Economics Detrending* | HP Filter, Pandas, Matplotlib | 1 | Medium |
| Dialogue Parsing* | Dialogue Graph, Graphviz, JSON Schema | 1 | Medium |

*Adapted from SkillsBench (see below).

## SkillsBench Integration

Three tasks are adapted from [SkillsBench](https://github.com/benchflow-ai/skillsbench) (Apache 2.0),
the first benchmark for evaluating how well AI agents use skills. SkillsBench proved that
agents fail to invoke skills ~56% of the time. Our environment provides the RL training
ground to fix this.

Adapted tasks use real SkillsBench skill content, distilled into our text-in/text-out
Gymnasium format with deterministic code execution verifiers.

## Procedural Task Generation

The environment includes a `TaskGenerator` that creates unlimited unique tasks at runtime,
preventing LLM memorization of fixed task content.

### Templates

| Template | What It Randomizes | Verifier |
|----------|--------------------|----------|
| `auth_protocol` | API name, hash algo (SHA-256/384/512/MD5), signing format, header format | HMAC exec |
| `binary_format` | Format name, magic bytes, endianness, flag names/bits | struct exec |

### Usage

```python
from skill_invocation_env.server.skill_invocation_env_environment import SkillInvocationEnvironment

# Procedural mode: every reset() generates a unique task
env = SkillInvocationEnvironment(use_procedural=True, procedural_seed=42)
obs = env.reset(seed=0)  # unique task from seed 0
obs = env.reset(seed=1)  # completely different task
```

## Testing

```bash
python test_env.py  # 34 tests
```

## Project Structure

```
skill_invocation_env/
├── __init__.py
├── models.py              # Pydantic Action/Observation/State
├── client.py              # SkillInvocationEnv(EnvClient)
├── task_bank.py           # 13 tasks + 27 skills + verifiers
├── task_generator.py      # Procedural task generator (2 templates)
├── README.md
├── openenv.yaml
├── pyproject.toml
├── train_demo.py          # Integration demo script
├── test_env.py            # Local test suite (34 tests)
└── server/
    ├── skill_invocation_env_environment.py  # Core Environment logic
    ├── app.py                               # FastAPI server
    ├── requirements.txt
    └── Dockerfile
```
