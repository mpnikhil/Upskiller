"""
Skill Invocation Environment Implementation.

Trains LLMs to decide WHEN to invoke procedural knowledge (skills) during
task-solving. Context cost model: each loaded skill costs context budget.

Reward has two distinct cost signals:
  - Context hygiene (bloat_penalty): penalizes irrelevant skills still loaded at
    submit time (-0.15 per skill).
  - Token efficiency (token_waste_penalty): penalizes skills that were ever loaded
    but turned out to be irrelevant, even if unloaded before submission (-0.05 per
    skill). This captures cumulative token waste across the episode.

Actions: load, unload, submit (plus "invoke" as backward-compat alias for load).
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import SkillInvocationAction, SkillInvocationObservation, SkillInvocationState
from task_bank import TASK_BANK, SKILL_BANK
from task_generator import TaskGenerator


DEFAULT_CONTEXT_BUDGET = 5


class SkillInvocationEnvironment(Environment):
    """
    RL environment for training skill invocation decisions.

    Episodes:
    1. reset() samples a task, assembles skill catalog (relevant + distractors)
    2. Agent can load and unload skills (within context budget)
    3. Agent submits a solution
    4. Reward = correctness + precision + recall - bloat - token_waste
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        use_procedural: bool = False,
        procedural_seed: int = 0,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
    ):
        super().__init__()
        self._state = SkillInvocationState(episode_id=str(uuid4()), step_count=0)
        self._current_task = None
        self._catalog_skill_ids: list[str] = []
        self._messages: list[str] = []
        self._use_procedural = use_procedural
        self._task_generator = TaskGenerator(seed=procedural_seed) if use_procedural else None
        self._episode_skills: dict = {}
        self._context_budget = context_budget
        # Per-instance RNG to avoid mutating global random state (concurrency-safe)
        self._rng = random.Random()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SkillInvocationObservation:
        """Sample a random task and assemble the skill catalog."""
        # Use a local RNG instance to avoid mutating global random state.
        # This is concurrency-safe: parallel rollouts won't clobber each other's seeds.
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        if self._use_procedural and self._task_generator:
            gen_seed = seed if seed is not None else self._rng.randint(0, 2**31)
            result = self._task_generator.generate_with_seed(gen_seed)
            task = result["task"]
            self._episode_skills = result["skills"]
        else:
            task = self._rng.choice(TASK_BANK)
            self._episode_skills = SKILL_BANK

        self._current_task = task

        # Build catalog: relevant + distractor skills, shuffled
        catalog_ids = list(task["relevant_skills"]) + list(task["distractor_skills"])
        self._rng.shuffle(catalog_ids)
        self._catalog_skill_ids = catalog_ids

        # Build catalog descriptions (short only, no full content)
        skill_catalog = []
        for sid in catalog_ids:
            skill = self._episode_skills[sid]
            skill_catalog.append({
                "id": sid,
                "name": skill["name"],
                "description": skill["short_description"],
            })

        # Initialize state
        eid = episode_id or str(uuid4())
        self._state = SkillInvocationState(
            episode_id=eid,
            step_count=0,
            task_id=task["id"],
            loaded_skills=[],
            skills_ever_loaded=[],
            skills_invoked=[],
            difficulty=task["difficulty"],
            done=False,
            context_budget_total=self._context_budget,
            remaining_invocations=self._context_budget,
        )
        self._messages = [f"Episode started. Task: {task['id']} ({task['difficulty']})"]

        return self._make_observation(
            skill_content=None,
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: SkillInvocationAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SkillInvocationObservation:
        """Process a load, unload, or submit action."""
        self._state.step_count += 1

        if self._state.done:
            self._messages.append("Episode already done. Call reset().")
            return self._make_observation(
                skill_content=None,
                verification_result="Episode already finished.",
                reward=0.0,
                done=True,
            )

        action_type = action.action_type

        # Backward compat: "invoke" is an alias for "load"
        if action_type == "invoke":
            action_type = "load"

        if action_type == "load":
            return self._handle_load(action)
        elif action_type == "unload":
            return self._handle_unload(action)
        elif action_type == "submit":
            return self._handle_submit(action)
        else:
            self._messages.append(f"Unknown action_type: {action.action_type}")
            return self._make_observation(
                skill_content=None,
                reward=0.0,
                done=False,
            )

    def _handle_load(self, action: SkillInvocationAction) -> SkillInvocationObservation:
        """Load a skill into context."""
        skill_id = action.skill_id

        if not skill_id:
            self._messages.append("load action requires skill_id")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        if skill_id not in self._episode_skills:
            self._messages.append(f"Unknown skill_id: {skill_id}")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        if skill_id not in self._catalog_skill_ids:
            self._messages.append(f"Skill {skill_id} not in current catalog.")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        # Already loaded — no-op, but still return content
        if skill_id in self._state.loaded_skills:
            full_content = self._episode_skills[skill_id]["full_content"]
            self._messages.append(f"Skill {skill_id} already loaded.")
            return self._make_observation(skill_content=full_content, reward=0.0, done=False)

        # Check context budget
        if len(self._state.loaded_skills) >= self._state.context_budget_total:
            self._messages.append(
                f"Context budget full ({self._state.context_budget_total} skills loaded). "
                "Unload a skill first."
            )
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        # Load skill
        self._state.loaded_skills.append(skill_id)
        if skill_id not in self._state.skills_ever_loaded:
            self._state.skills_ever_loaded.append(skill_id)
        # Backward compat
        self._state.skills_invoked = list(self._state.skills_ever_loaded)
        self._state.remaining_invocations = (
            self._state.context_budget_total - len(self._state.loaded_skills)
        )

        full_content = self._episode_skills[skill_id]["full_content"]
        skill_name = self._episode_skills[skill_id]["name"]
        self._messages.append(
            f"Loaded skill '{skill_name}' ({skill_id}). "
            f"Context: {len(self._state.loaded_skills)}/{self._state.context_budget_total}"
        )

        return self._make_observation(
            skill_content=full_content,
            reward=0.0,
            done=False,
        )

    def _handle_unload(self, action: SkillInvocationAction) -> SkillInvocationObservation:
        """Unload a skill from context."""
        skill_id = action.skill_id

        if not skill_id:
            self._messages.append("unload action requires skill_id")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        if skill_id not in self._state.loaded_skills:
            self._messages.append(f"Skill {skill_id} is not currently loaded.")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        self._state.loaded_skills.remove(skill_id)
        self._state.remaining_invocations = (
            self._state.context_budget_total - len(self._state.loaded_skills)
        )

        skill_name = self._episode_skills[skill_id]["name"]
        self._messages.append(
            f"Unloaded skill '{skill_name}' ({skill_id}). "
            f"Context: {len(self._state.loaded_skills)}/{self._state.context_budget_total}"
        )

        return self._make_observation(skill_content=None, reward=0.0, done=False)

    def _handle_submit(self, action: SkillInvocationAction) -> SkillInvocationObservation:
        """Handle a solution submission.

        Reward = correctness + precision + recall - bloat - token_waste.

        Two distinct cost signals:
          - bloat_penalty (-0.15 per skill): penalizes irrelevant skills still
            loaded at submit time (context hygiene).
          - token_waste_penalty (-0.05 per skill): penalizes skills that were ever
            loaded but turned out irrelevant, capturing cumulative token waste
            across the episode (token efficiency).
        """
        answer = action.answer or ""
        task = self._current_task

        # Run deterministic verifier
        try:
            task_correct = task["verifier"](answer)
        except Exception:
            task_correct = False

        # Compute reward
        loaded = set(self._state.loaded_skills)
        ever_loaded = set(self._state.skills_ever_loaded)
        relevant = set(task["relevant_skills"])

        # 1. Correctness: +0.6
        correctness = 0.6 if task_correct else 0.0

        # 2. Precision: what fraction of loaded skills are relevant?
        if len(loaded) > 0:
            precision = len(loaded & relevant) / len(loaded)
        else:
            precision = 0.0
        precision_bonus = 0.3 * precision

        # 3. Recall: did you load all relevant skills?
        if len(relevant) > 0:
            recall = len(loaded & relevant) / len(relevant)
        else:
            recall = 1.0
        recall_bonus = 0.1 * recall

        # 4. Bloat: penalty for unnecessary skills loaded at submit time
        unnecessary = loaded - relevant
        bloat_penalty = -0.15 * len(unnecessary)

        # 5. Token waste: penalty for skills ever loaded that were irrelevant
        wasted = ever_loaded - relevant
        token_waste_penalty = -0.05 * len(wasted)

        total_reward = correctness + precision_bonus + recall_bonus + bloat_penalty + token_waste_penalty
        total_reward = max(total_reward, -1.0)

        self._state.done = True
        verification_msg = (
            f"{'CORRECT' if task_correct else 'INCORRECT'}. "
            f"Reward: correctness={correctness:.2f}, "
            f"precision={precision_bonus:.2f}, recall={recall_bonus:.2f}, "
            f"bloat={bloat_penalty:.2f}, token_waste={token_waste_penalty:.2f}, "
            f"total={total_reward:.2f}"
        )
        self._messages.append(f"Submitted answer. {verification_msg}")

        return self._make_observation(
            skill_content=None,
            verification_result=verification_msg,
            reward=total_reward,
            done=True,
        )

    def _make_observation(
        self,
        skill_content: Optional[str],
        reward: float,
        done: bool,
        verification_result: Optional[str] = None,
    ) -> SkillInvocationObservation:
        """Build an observation from current state."""
        task = self._current_task
        catalog = []
        if task:
            for sid in self._catalog_skill_ids:
                if sid in self._episode_skills:
                    skill = self._episode_skills[sid]
                    catalog.append({
                        "id": sid,
                        "name": skill["name"],
                        "description": skill["short_description"],
                    })

        # Build loaded skill contents
        loaded_contents = {}
        for sid in self._state.loaded_skills:
            if sid in self._episode_skills:
                loaded_contents[sid] = self._episode_skills[sid]["full_content"]

        return SkillInvocationObservation(
            task_description=task["description"] if task else "",
            skill_catalog=catalog,
            difficulty=self._state.difficulty,
            loaded_skills=list(self._state.loaded_skills),
            loaded_skill_contents=loaded_contents,
            context_budget_used=len(self._state.loaded_skills),
            context_budget_total=self._state.context_budget_total,
            skill_content=skill_content,
            remaining_invocations=(
                self._state.context_budget_total - len(self._state.loaded_skills)
            ),
            verification_result=verification_result,
            skills_invoked=list(self._state.skills_ever_loaded),
            messages=list(self._messages),
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> SkillInvocationState:
        """Get current episode state."""
        return self._state
