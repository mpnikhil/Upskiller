"""Skill Invocation Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import SkillInvocationAction, SkillInvocationObservation, SkillInvocationState


class SkillInvocationEnv(
    EnvClient[SkillInvocationAction, SkillInvocationObservation, SkillInvocationState]
):
    """
    Client for the Skill Invocation Environment.

    Example:
        >>> with SkillInvocationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...     result = client.step(SkillInvocationAction(
        ...         action_type="load", skill_id="skill_001"
        ...     ))
        ...     print(result.observation.skill_content[:100])
    """

    def _step_payload(self, action: SkillInvocationAction) -> Dict:
        """Convert action to JSON payload."""
        payload = {"action_type": action.action_type}
        if action.skill_id is not None:
            payload["skill_id"] = action.skill_id
        if action.answer is not None:
            payload["answer"] = action.answer
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SkillInvocationObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = SkillInvocationObservation(
            task_description=obs_data.get("task_description", ""),
            skill_catalog=obs_data.get("skill_catalog", []),
            difficulty=obs_data.get("difficulty", "easy"),
            loaded_skills=obs_data.get("loaded_skills", []),
            loaded_skill_contents=obs_data.get("loaded_skill_contents", {}),
            context_budget_used=obs_data.get("context_budget_used", 0),
            context_budget_total=obs_data.get("context_budget_total", 5),
            skill_content=obs_data.get("skill_content"),
            remaining_invocations=obs_data.get("remaining_invocations", 0),
            verification_result=obs_data.get("verification_result"),
            skills_invoked=obs_data.get("skills_invoked", []),
            messages=obs_data.get("messages", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SkillInvocationState:
        """Parse server response into State object."""
        return SkillInvocationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            loaded_skills=payload.get("loaded_skills", []),
            skills_ever_loaded=payload.get("skills_ever_loaded", []),
            skills_invoked=payload.get("skills_invoked", []),
            difficulty=payload.get("difficulty", "easy"),
            done=payload.get("done", False),
            context_budget_total=payload.get("context_budget_total", 5),
            remaining_invocations=payload.get("remaining_invocations", 5),
        )
