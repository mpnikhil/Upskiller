#!/usr/bin/env python3
"""
Minimal TRL + OpenEnv integration demo for the Skill Invocation Environment.

This script demonstrates how to connect to the environment and run episodes.
It can be run in Google Colab with Unsloth for actual RL training.

Setup (Colab):
    !pip install unsloth openenv-core trl
    !pip install skill_invocation_env  # or install from local

Usage:
    # Against a local server:
    python train_demo.py --base-url http://localhost:8000

    # Against a HuggingFace Space:
    python train_demo.py --base-url https://YOUR-SPACE.hf.space
"""

import sys
import os

# For local testing without server, use direct environment
sys.path.insert(0, os.path.dirname(__file__))


def demo_direct():
    """Demo using the environment directly (no server needed)."""
    from models import SkillInvocationAction
    from server.skill_invocation_env_environment import SkillInvocationEnvironment

    print("=== Direct Environment Demo ===\n")

    env = SkillInvocationEnvironment()

    # Run 3 episodes
    for episode in range(3):
        obs = env.reset(seed=episode)
        print(f"--- Episode {episode + 1} ---")
        print(f"Task: {obs.task_description[:100]}...")
        print(f"Difficulty: {obs.difficulty}")
        print(f"Skills available: {[s['name'] for s in obs.skill_catalog]}")
        print(f"Context budget: {obs.context_budget_used}/{obs.context_budget_total}")

        # Strategy: load the first skill in catalog
        if obs.skill_catalog:
            skill = obs.skill_catalog[0]
            print(f"\nLoading skill: {skill['name']} ({skill['id']})")
            obs = env.step(SkillInvocationAction(
                action_type="load",
                skill_id=skill["id"],
            ))
            if obs.skill_content:
                print(f"Got skill content ({len(obs.skill_content)} chars)")
                print(f"Preview: {obs.skill_content[:150]}...")
                print(f"Context: {obs.context_budget_used}/{obs.context_budget_total}")

        # Submit a dummy answer
        print("\nSubmitting answer...")
        obs = env.step(SkillInvocationAction(
            action_type="submit",
            answer="This is a placeholder answer for demonstration.",
        ))
        print(f"Done: {obs.done}")
        print(f"Reward: {obs.reward}")
        print(f"Verification: {obs.verification_result}")
        print()

    print("Demo complete!")


def demo_client(base_url: str):
    """Demo using the WebSocket client against a running server."""
    from client import SkillInvocationEnv
    from models import SkillInvocationAction

    print(f"=== Client Demo (connecting to {base_url}) ===\n")

    with SkillInvocationEnv(base_url=base_url) as client:
        # Reset
        result = client.reset()
        obs = result.observation
        print(f"Task: {obs.task_description[:100]}...")
        print(f"Skills available: {[s['name'] for s in obs.skill_catalog]}")

        # Load first skill
        if obs.skill_catalog:
            skill = obs.skill_catalog[0]
            result = client.step(SkillInvocationAction(
                action_type="load",
                skill_id=skill["id"],
            ))
            print(f"\nLoaded '{skill['name']}'")
            if result.observation.skill_content:
                print(f"Content preview: {result.observation.skill_content[:200]}...")

        # Submit
        result = client.step(SkillInvocationAction(
            action_type="submit",
            answer="test answer",
        ))
        print(f"\nReward: {result.reward}")
        print(f"Done: {result.done}")
        print(f"Verification: {result.observation.verification_result}")

    print("\nClient demo complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skill Invocation Env Demo")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Server URL (if not provided, runs directly without server)",
    )
    args = parser.parse_args()

    if args.base_url:
        demo_client(args.base_url)
    else:
        demo_direct()
