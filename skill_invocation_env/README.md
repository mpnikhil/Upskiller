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

## How It Works

Each episode:
1. Agent receives a **task description** + a **skill catalog** (short descriptions only)
2. Agent can **invoke skills** (up to 3) to read full procedural content
3. Agent **submits a solution**
4. Environment computes a **composite reward**

## Reward Function

```
reward = task_correct * 0.7
       + invocation_bonus * 0.2   (for each relevant skill invoked, normalized)
       - distractor_penalty * 0.1 (for each distractor skill invoked)
```

- Maximum reward: 0.9 (correct answer + all relevant skills invoked, no distractors)
- Minimum reward: -1.0 (floor)

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

# Invoke a skill
obs = env.step(SkillInvocationAction(action_type="invoke", skill_id=obs.skill_catalog[0]["id"]))
print(f"Skill content: {obs.skill_content[:200]}...")

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

    # Invoke a skill
    skill_id = result.observation.skill_catalog[0]["id"]
    result = client.step(SkillInvocationAction(action_type="invoke", skill_id=skill_id))

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

The environment includes 10 synthetic tasks across 6 fictional domains:

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

## Testing

```bash
python test_env.py
```

## Project Structure

```
skill_invocation_env/
├── __init__.py
├── models.py              # Pydantic Action/Observation/State
├── client.py              # SkillInvocationEnv(EnvClient)
├── task_bank.py           # 10 synthetic tasks + 18 skills + verifiers
├── README.md
├── openenv.yaml
├── pyproject.toml
├── train_demo.py          # Integration demo script
├── test_env.py            # Local test suite (10 tests)
└── server/
    ├── skill_invocation_env_environment.py  # Core Environment logic
    ├── app.py                               # FastAPI server
    ├── requirements.txt
    └── Dockerfile
```
