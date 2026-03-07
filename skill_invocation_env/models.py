"""
Data models for the Skill Invocation Environment.

This environment trains LLMs to decide WHEN to invoke procedural knowledge (skills)
during task-solving. The agent receives a task + skill catalog, must decide which
skills to load for full procedural knowledge, then submits a solution.

Context cost model: each loaded skill costs context budget. The reward penalizes
bloat (unnecessary skills loaded at submit time) and rewards precision.
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
    """Agent's action — list, load, unload, or submit."""

    action_type: str = Field(
        ...,
        description=(
            '"load" to load a skill, "unload" to unload a skill, '
            'or "submit" to submit answer'
        ),
    )
    skill_id: Optional[str] = Field(
        default=None, description='Skill ID (required for load/unload)'
    )
    answer: Optional[str] = Field(
        default=None, description='Solution text (required for submit)', max_length=100000
    )


class SkillInvocationObservation(Observation):
    """What the agent sees at each step."""

    task_description: str = Field(default="", description="The task to solve")
    skill_catalog: list[dict] = Field(
        default_factory=list,
        description="Available skills with id, name, and description",
    )
    difficulty: str = Field(default="easy", description="Task difficulty level")

    # Skill context management
    loaded_skills: list[str] = Field(
        default_factory=list, description="IDs of currently loaded skills"
    )
    loaded_skill_contents: dict = Field(
        default_factory=dict,
        description="Mapping of skill_id -> full_content for loaded skills",
    )
    context_budget_used: int = Field(
        default=0, description="Number of skills currently loaded"
    )
    context_budget_total: int = Field(
        default=5, description="Max skills that can be loaded simultaneously"
    )

    # Backward compat
    skill_content: Optional[str] = Field(
        default=None, description="Full content of last loaded skill"
    )
    remaining_invocations: int = Field(
        default=5, description="Remaining context budget (backward compat)"
    )
    verification_result: Optional[str] = Field(
        default=None, description="Result of answer verification"
    )
    skills_invoked: list[str] = Field(
        default_factory=list, description="IDs of all skills ever loaded this episode"
    )
    messages: list[str] = Field(
        default_factory=list, description="Running log of actions/observations"
    )


class SkillInvocationState(State):
    """Internal episode state tracked server-side."""

    task_id: str = Field(default="", description="Current task ID")
    loaded_skills: list[str] = Field(
        default_factory=list, description="Currently loaded skills (in context)"
    )
    skills_ever_loaded: list[str] = Field(
        default_factory=list, description="All skills ever loaded this episode"
    )
    difficulty: str = Field(default="easy", description="Task difficulty")
    done: bool = Field(default=False, description="Whether episode is finished")
    context_budget_total: int = Field(default=5, description="Max simultaneous skills")

    # Backward compat
    skills_invoked: list[str] = Field(
        default_factory=list, description="Alias for skills_ever_loaded"
    )
    remaining_invocations: int = Field(
        default=5, description="Remaining context budget"
    )
