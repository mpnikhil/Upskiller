"""
Data models for the Skill Invocation Environment.

This environment trains LLMs to decide WHEN to invoke procedural knowledge (skills)
during task-solving. The agent receives a task + skill catalog, must decide which
skills to invoke for full procedural knowledge, then submits a solution.
"""

from typing import Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class SkillDescription(Action):
    """A short description of a skill visible in the catalog."""

    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(
        ..., description="1-2 sentence summary of what the skill covers"
    )


class SkillInvocationAction(Action):
    """Agent's action — either invoke a skill or submit a solution."""

    action_type: str = Field(
        ..., description='Either "invoke" to read a skill or "submit" to submit answer'
    )
    skill_id: Optional[str] = Field(
        default=None, description='Skill ID to invoke (required if action_type == "invoke")'
    )
    answer: Optional[str] = Field(
        default=None, description='Solution text (required if action_type == "submit")'
    )


class SkillInvocationObservation(Observation):
    """What the agent sees at each step."""

    task_description: str = Field(default="", description="The task to solve")
    skill_catalog: list[dict] = Field(
        default_factory=list,
        description="Available skills with id, name, and description",
    )
    difficulty: str = Field(default="easy", description="Task difficulty level")
    skill_content: Optional[str] = Field(
        default=None, description="Full skill content after invocation"
    )
    remaining_invocations: int = Field(
        default=3, description="How many more skills can be invoked"
    )
    verification_result: Optional[str] = Field(
        default=None, description="Result of answer verification"
    )
    skills_invoked: list[str] = Field(
        default_factory=list, description="IDs of skills already invoked"
    )
    messages: list[str] = Field(
        default_factory=list, description="Running log of actions/observations"
    )


class SkillInvocationState(State):
    """Internal episode state tracked server-side."""

    task_id: str = Field(default="", description="Current task ID")
    skills_invoked: list[str] = Field(
        default_factory=list, description="Skills invoked this episode"
    )
    difficulty: str = Field(default="easy", description="Task difficulty")
    done: bool = Field(default=False, description="Whether episode is finished")
    remaining_invocations: int = Field(default=3, description="Invocations left")
