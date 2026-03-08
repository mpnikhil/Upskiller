"""Test env connection patterns to find the WebSocket bug."""
import os
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

from skill_invocation_env.client import SkillInvocationEnv
from skill_invocation_env.models import SkillInvocationAction

ENV_URL = os.getenv("ENV_URL", "http://localhost:8001")

# Test 1: reuse single client
print("=== Test 1: Single client, multiple episodes ===")
env = SkillInvocationEnv(base_url=ENV_URL)
for i in range(5):
    result = env.reset(seed=i)
    print(f"Episode {i}: {result.observation.task_description[:60]}")
    action = SkillInvocationAction(action_type="submit", answer="test")
    result = env.step(action)
    print(f"  Reward: {result.reward}, Done: {result.done}")
env.close()

# Test 2: new client per episode, properly closed
print("\n=== Test 2: New client per episode (with close) ===")
for i in range(10):
    env = SkillInvocationEnv(base_url=ENV_URL)
    result = env.reset(seed=i)
    print(f"Client {i}: {result.observation.task_description[:60]}")
    action = SkillInvocationAction(action_type="submit", answer="test")
    result = env.step(action)
    print(f"  Reward: {result.reward}, Done: {result.done}")
    env.close()

print("\nALL OK")
