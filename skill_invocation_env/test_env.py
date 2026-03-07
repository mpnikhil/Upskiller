#!/usr/bin/env python3
"""
Local test script for the Skill Invocation Environment.

Tests the environment directly (no server) to verify:
- reset() works and returns proper observation
- invoke action returns skill content
- submit action computes rewards correctly
- edge cases (unknown skills, exhausted invocations, etc.)
"""

import sys
import os

# Add parent dir so imports work
sys.path.insert(0, os.path.dirname(__file__))

from models import SkillInvocationAction, SkillInvocationObservation, SkillInvocationState
from task_bank import TASK_BANK, SKILL_BANK
from server.skill_invocation_env_environment import SkillInvocationEnvironment


def test_reset():
    """Test that reset returns a valid observation."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    assert isinstance(obs, SkillInvocationObservation)
    assert obs.task_description != ""
    assert len(obs.skill_catalog) >= 3  # relevant + distractors
    assert obs.remaining_invocations == 3
    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.skill_content is None
    assert obs.skills_invoked == []
    assert len(obs.messages) > 0

    print("[PASS] test_reset")


def test_invoke_skill():
    """Test invoking a valid skill returns full content."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    # Get a skill from the catalog
    skill_id = obs.skill_catalog[0]["id"]
    action = SkillInvocationAction(action_type="invoke", skill_id=skill_id)
    obs2 = env.step(action)

    assert obs2.skill_content is not None
    assert len(obs2.skill_content) > 0
    assert obs2.remaining_invocations == 2
    assert skill_id in obs2.skills_invoked
    assert obs2.done is False
    assert obs2.reward == 0.0

    print("[PASS] test_invoke_skill")


def test_invoke_unknown_skill():
    """Test invoking a skill not in the catalog."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    action = SkillInvocationAction(action_type="invoke", skill_id="skill_999")
    obs = env.step(action)

    assert obs.skill_content is None
    assert obs.remaining_invocations == 3  # Not decremented

    print("[PASS] test_invoke_unknown_skill")


def test_exhausted_invocations():
    """Test that invocations are limited."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    catalog_ids = [s["id"] for s in obs.skill_catalog]

    # Use up all 3 invocations
    for i in range(3):
        idx = i % len(catalog_ids)
        action = SkillInvocationAction(action_type="invoke", skill_id=catalog_ids[idx])
        obs = env.step(action)

    assert obs.remaining_invocations == 0

    # 4th invocation should fail gracefully
    action = SkillInvocationAction(action_type="invoke", skill_id=catalog_ids[0])
    obs = env.step(action)
    assert obs.skill_content is None
    assert obs.remaining_invocations == 0

    print("[PASS] test_exhausted_invocations")


def test_submit_incorrect():
    """Test submitting an incorrect answer."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    action = SkillInvocationAction(action_type="submit", answer="I don't know")
    obs = env.step(action)

    assert obs.done is True
    assert obs.reward <= 0.0  # No correctness reward
    assert obs.verification_result is not None
    assert "INCORRECT" in obs.verification_result

    print("[PASS] test_submit_incorrect")


def test_submit_after_done():
    """Test that actions after done return done state."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    # Submit
    action = SkillInvocationAction(action_type="submit", answer="test")
    env.step(action)

    # Try another action
    action2 = SkillInvocationAction(action_type="invoke", skill_id="skill_001")
    obs = env.step(action2)
    assert obs.done is True

    print("[PASS] test_submit_after_done")


def test_correct_submission_task_001():
    """Test a correct answer for task_001 (Zephyr-3 auth)."""
    env = SkillInvocationEnvironment()

    # Find task_001 by resetting with different seeds until we get it
    for seed in range(100):
        obs = env.reset(seed=seed)
        state = env.state
        if state.task_id == "task_001":
            break
    else:
        print("[SKIP] test_correct_submission_task_001 - couldn't find task")
        return

    # Invoke the relevant skill
    action = SkillInvocationAction(action_type="invoke", skill_id="skill_001")
    obs = env.step(action)
    assert obs.skill_content is not None

    # Submit a correct-ish answer that passes the verifier
    correct_answer = """
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
"""
    action = SkillInvocationAction(action_type="submit", answer=correct_answer)
    obs = env.step(action)

    assert obs.done is True
    assert "CORRECT" in obs.verification_result
    assert obs.reward > 0.0
    # Should get: 0.7 (correct) + 0.2 (invoked relevant skill) = 0.9
    assert obs.reward >= 0.89, f"Expected reward >= 0.89, got {obs.reward}"

    print(f"[PASS] test_correct_submission_task_001 (reward={obs.reward})")


def test_distractor_penalty():
    """Test that invoking distractors reduces reward."""
    env = SkillInvocationEnvironment()

    # Find task_001
    for seed in range(100):
        obs = env.reset(seed=seed)
        state = env.state
        if state.task_id == "task_001":
            break
    else:
        print("[SKIP] test_distractor_penalty - couldn't find task")
        return

    # Invoke a distractor
    action = SkillInvocationAction(action_type="invoke", skill_id="skill_002")
    env.step(action)

    # Invoke the relevant skill too
    action = SkillInvocationAction(action_type="invoke", skill_id="skill_001")
    env.step(action)

    # Submit correct answer
    correct_answer = """
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
"""
    action = SkillInvocationAction(action_type="submit", answer=correct_answer)
    obs = env.step(action)

    # Should get: 0.7 (correct) + 0.2 (relevant) - 0.1 (distractor) = 0.8
    assert obs.reward < 0.9, f"Expected reward < 0.9 due to distractor, got {obs.reward}"
    assert obs.reward >= 0.7, f"Expected reward >= 0.7, got {obs.reward}"

    print(f"[PASS] test_distractor_penalty (reward={obs.reward})")


def test_state_property():
    """Test that state returns correct metadata."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    state = env.state
    assert isinstance(state, SkillInvocationState)
    assert state.episode_id is not None
    assert state.step_count == 0
    assert state.task_id != ""
    assert state.done is False

    # After a step
    skill_id = obs.skill_catalog[0]["id"]
    env.step(SkillInvocationAction(action_type="invoke", skill_id=skill_id))

    state = env.state
    assert state.step_count == 1

    print("[PASS] test_state_property")


def test_all_tasks_have_valid_skills():
    """Verify task bank integrity."""
    for task in TASK_BANK:
        for sid in task["relevant_skills"]:
            assert sid in SKILL_BANK, f"Task {task['id']}: missing relevant skill {sid}"
        for sid in task["distractor_skills"]:
            assert sid in SKILL_BANK, f"Task {task['id']}: missing distractor skill {sid}"
        # Verify no overlap between relevant and distractor
        overlap = set(task["relevant_skills"]) & set(task["distractor_skills"])
        assert len(overlap) == 0, f"Task {task['id']}: overlap between relevant and distractor: {overlap}"

    print(f"[PASS] test_all_tasks_have_valid_skills ({len(TASK_BANK)} tasks verified)")


if __name__ == "__main__":
    print("=" * 60)
    print("Skill Invocation Environment - Local Tests")
    print("=" * 60)

    tests = [
        test_reset,
        test_invoke_skill,
        test_invoke_unknown_skill,
        test_exhausted_invocations,
        test_submit_incorrect,
        test_submit_after_done,
        test_correct_submission_task_001,
        test_distractor_penalty,
        test_state_property,
        test_all_tasks_have_valid_skills,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
