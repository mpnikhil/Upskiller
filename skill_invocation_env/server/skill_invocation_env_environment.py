"""
Skill Invocation Environment Implementation.

Trains LLMs to decide WHEN to invoke procedural knowledge (skills) during
task-solving. The agent gets a task + skill catalog, decides which skills to
invoke for full procedural knowledge, then submits a solution.
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import SkillInvocationAction, SkillInvocationObservation, SkillInvocationState
from task_bank import TASK_BANK, SKILL_BANK


MAX_INVOCATIONS = 3


class SkillInvocationEnvironment(Environment):
    """
    RL environment for training skill invocation decisions.

    Episodes:
    1. reset() samples a task, assembles skill catalog (relevant + distractors)
    2. Agent can invoke skills (up to MAX_INVOCATIONS) to read full content
    3. Agent submits a solution
    4. Reward is computed based on task correctness + invocation quality
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = SkillInvocationState(episode_id=str(uuid4()), step_count=0)
        self._current_task = None
        self._catalog_skill_ids: list[str] = []
        self._messages: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SkillInvocationObservation:
        """Sample a random task and assemble the skill catalog."""
        if seed is not None:
            random.seed(seed)

        # Pick a random task
        task = random.choice(TASK_BANK)
        self._current_task = task

        # Build catalog: relevant + distractor skills, shuffled
        catalog_ids = list(task["relevant_skills"]) + list(task["distractor_skills"])
        random.shuffle(catalog_ids)
        self._catalog_skill_ids = catalog_ids

        # Build catalog descriptions (short only, no full content)
        skill_catalog = []
        for sid in catalog_ids:
            skill = SKILL_BANK[sid]
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
            skills_invoked=[],
            difficulty=task["difficulty"],
            done=False,
            remaining_invocations=MAX_INVOCATIONS,
        )
        self._messages = [f"Episode started. Task: {task['id']} ({task['difficulty']})"]

        return SkillInvocationObservation(
            task_description=task["description"],
            skill_catalog=skill_catalog,
            difficulty=task["difficulty"],
            skill_content=None,
            remaining_invocations=MAX_INVOCATIONS,
            verification_result=None,
            skills_invoked=[],
            messages=list(self._messages),
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: SkillInvocationAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SkillInvocationObservation:
        """Process an invoke or submit action."""
        self._state.step_count += 1

        if self._state.done:
            self._messages.append("Episode already done. Call reset().")
            return self._make_observation(
                skill_content=None,
                verification_result="Episode already finished.",
                reward=0.0,
                done=True,
            )

        if action.action_type == "invoke":
            return self._handle_invoke(action)
        elif action.action_type == "submit":
            return self._handle_submit(action)
        else:
            self._messages.append(f"Unknown action_type: {action.action_type}")
            return self._make_observation(
                skill_content=None,
                verification_result=None,
                reward=0.0,
                done=False,
            )

    def _handle_invoke(self, action: SkillInvocationAction) -> SkillInvocationObservation:
        """Handle a skill invocation action."""
        skill_id = action.skill_id

        if not skill_id:
            self._messages.append("invoke action requires skill_id")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        if self._state.remaining_invocations <= 0:
            self._messages.append("No remaining invocations. Submit your answer.")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        if skill_id not in SKILL_BANK:
            self._messages.append(f"Unknown skill_id: {skill_id}")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        if skill_id not in self._catalog_skill_ids:
            self._messages.append(f"Skill {skill_id} not in current catalog.")
            return self._make_observation(skill_content=None, reward=0.0, done=False)

        # Successful invocation
        self._state.remaining_invocations -= 1
        if skill_id not in self._state.skills_invoked:
            self._state.skills_invoked.append(skill_id)

        full_content = SKILL_BANK[skill_id]["full_content"]
        skill_name = SKILL_BANK[skill_id]["name"]
        self._messages.append(
            f"Invoked skill '{skill_name}' ({skill_id}). "
            f"Remaining invocations: {self._state.remaining_invocations}"
        )

        return self._make_observation(
            skill_content=full_content,
            reward=0.0,
            done=False,
        )

    def _handle_submit(self, action: SkillInvocationAction) -> SkillInvocationObservation:
        """Handle a solution submission. Compute composite reward."""
        answer = action.answer or ""
        task = self._current_task

        # Run deterministic verifier
        try:
            task_correct = task["verifier"](answer)
        except Exception:
            task_correct = False

        # Compute composite reward
        # 1. Task correctness: 0 or 1 * 0.7
        correctness_reward = 0.7 if task_correct else 0.0

        # 2. Invocation bonus: +0.2 for each relevant skill correctly invoked (normalized)
        relevant_skills = set(task["relevant_skills"])
        invoked_skills = set(self._state.skills_invoked)
        relevant_invoked = relevant_skills & invoked_skills
        if relevant_skills:
            invocation_bonus = 0.2 * (len(relevant_invoked) / len(relevant_skills))
        else:
            invocation_bonus = 0.0

        # 3. Distractor penalty: -0.1 for each distractor invoked
        distractor_skills = set(task["distractor_skills"])
        distractors_invoked = distractor_skills & invoked_skills
        distractor_penalty = -0.1 * len(distractors_invoked)

        total_reward = correctness_reward + invocation_bonus + distractor_penalty
        total_reward = max(total_reward, -1.0)  # Floor at -1.0

        self._state.done = True
        verification_msg = (
            f"{'CORRECT' if task_correct else 'INCORRECT'}. "
            f"Reward breakdown: correctness={correctness_reward:.2f}, "
            f"invocation_bonus={invocation_bonus:.2f}, "
            f"distractor_penalty={distractor_penalty:.2f}, "
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
            catalog_ids = list(task["relevant_skills"]) + list(task["distractor_skills"])
            for sid in self._catalog_skill_ids:
                if sid in SKILL_BANK:
                    skill = SKILL_BANK[sid]
                    catalog.append({
                        "id": sid,
                        "name": skill["name"],
                        "description": skill["short_description"],
                    })

        return SkillInvocationObservation(
            task_description=task["description"] if task else "",
            skill_catalog=catalog,
            difficulty=self._state.difficulty,
            skill_content=skill_content,
            remaining_invocations=self._state.remaining_invocations,
            verification_result=verification_result,
            skills_invoked=list(self._state.skills_invoked),
            messages=list(self._messages),
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> SkillInvocationState:
        """Get current episode state."""
        return self._state
