"""
GRPO Training v2 for Skill Invocation Environment.

Changes from v1:
- num_generations: 4 → 16 (better GRPO advantage estimation)
- LoRA: r=128, all projection layers (more capacity to learn new behavior)
- gradient_accumulation_steps: 8 → 2 (compensate for 4x more generations)
- NUM_EPISODES: 64 → 32 (fewer steps, each much higher quality)

Run on Northflank / Jupyter with GPU:
    python skill_invocation_env/train_demo_v2.py
"""

import os

import torch
import wandb
from datasets import Dataset
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from skill_invocation_env.client import SkillInvocationEnv
from skill_invocation_env.models import SkillInvocationAction

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-8B")
ENV_URL = os.getenv("ENV_URL", "https://mpnikhil-skill-invocation-env.hf.space")
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs/qwen-skill-env-v2")
HUB_REPO = os.getenv("HUB_REPO", "mpnikhil/Qwen2.5-3B-Skill-Invocation")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "32"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "16"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "4096"))

SYSTEM_PROMPT = """\
You solve tasks using a catalog of skills. Each skill has an ID, name, and description.

WORKFLOW:
1. Read the task carefully
2. Load ONLY the skills whose descriptions match the task (1-2 skills max)
3. Read the loaded skill content — it contains exact syntax, code, and configurations
4. Submit your answer using the ACTUAL code/syntax/config from the loaded skills, adapted to the task requirements

CRITICAL: Your submitted answer must contain the actual code, configuration, or implementation — NOT a description of what it should do. \
Copy and adapt the patterns from loaded skills directly.

IMPORTANT: You can only submit ONCE. After submitting you will see "Answer submitted and recorded. Episode complete." — this means the episode is over. Do not call any more tools after this."""


def format_observation(obs) -> str:
    """Formats the observation into a feedback string."""
    parts = [f"TASK: {obs.task_description}\n\nSKILL CATALOG:"]
    for s in obs.skill_catalog:
        parts.append(f"- [{s['id']}] {s['name']}: {s['description']}")

    if obs.loaded_skills:
        parts.append(f"\nCURRENTLY LOADED SKILLS: {', '.join(obs.loaded_skills)}")

    if obs.skill_content:
        parts.append(f"\nJUST LOADED SKILL CONTENT:\n{obs.skill_content}")

    if obs.loaded_skill_contents:
        just_loaded_id = None
        if obs.skill_content:
            for sid, content in obs.loaded_skill_contents.items():
                if content == obs.skill_content:
                    just_loaded_id = sid
                    break
        other_contents = {
            sid: content
            for sid, content in obs.loaded_skill_contents.items()
            if sid != just_loaded_id
        }
        if other_contents:
            parts.append("\nOTHER LOADED SKILL CONTENTS:")
            for sid, content in other_contents.items():
                parts.append(f"\n[{sid}]:\n{content}")

    if obs.verification_result:
        parts.append(f"\nVERIFICATION: {obs.verification_result}")

    if obs.messages:
        parts.append(f"\nSTATUS: {obs.messages[-1]}")

    parts.append(f"\nBUDGET USED: {obs.context_budget_used} / {obs.context_budget_total}")
    return "\n".join(parts)


# ── Environment wrapper (methods become tool calls) ───────────────────────────

class SkillEnv:
    """Environment for TRL's environment_factory. Public methods become tools."""

    def __init__(self):
        self.client = None
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = SkillInvocationEnv(base_url=ENV_URL, connect_timeout_s=60)
        result = self.client.reset()
        self.reward = 0.0
        self.done = False
        return format_observation(result.observation)

    def __del__(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def load_skill(self, skill_id: str) -> str:
        """Load a skill to read its contents. Costs context budget.

        Args:
            skill_id: The ID of the skill to load (e.g. 'skill_01')

        Returns:
            The loaded skill content.
        """
        if self.done:
            raise ValueError("Game over.")
        action = SkillInvocationAction(action_type="load", skill_id=skill_id)
        result = self.client.step(action)
        self.done = result.done
        self.reward = float(result.reward or 0.0)
        obs = result.observation
        content = obs.skill_content or "No content returned."
        return f"[{skill_id}] loaded (budget: {obs.context_budget_used}/{obs.context_budget_total}):\n{content}"

    def unload_skill(self, skill_id: str) -> str:
        """Unload a skill to free context budget.

        Args:
            skill_id: The ID of the skill to unload (e.g. 'skill_01')

        Returns:
            Confirmation of unload.
        """
        if self.done:
            raise ValueError("Game over.")
        action = SkillInvocationAction(action_type="unload", skill_id=skill_id)
        result = self.client.step(action)
        self.done = result.done
        self.reward = float(result.reward or 0.0)
        obs = result.observation
        return f"Unloaded {skill_id}. Budget: {obs.context_budget_used}/{obs.context_budget_total}"

    def submit(self, answer: str) -> str:
        """Submit your final solution to the task.

        Args:
            answer: Your solution to the task

        Returns:
            Verification result with your score.
        """
        if self.done:
            raise ValueError("Game over.")
        action = SkillInvocationAction(action_type="submit", answer=answer)
        result = self.client.step(action)
        self.done = result.done
        self.reward = float(result.reward or 0.0)
        obs = result.observation
        return "Answer submitted and recorded. Episode complete."


def reward_func(environments, **kwargs) -> list[float]:
    """Extract rewards from environment instances."""
    return [env.reward for env in environments]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting GRPO Training v2 with {MODEL_ID}")
    print(f"Environment: {ENV_URL}")
    print(f"Episodes: {NUM_EPISODES}, Generations per episode: {NUM_GENERATIONS}")

    wandb.init(
        project="skill-invocation-env",
        name=f"grpo-v2-{MODEL_ID.split('/')[-1]}-g{NUM_GENERATIONS}",
        config={
            "model_id": MODEL_ID,
            "env_url": ENV_URL,
            "num_episodes": NUM_EPISODES,
            "num_generations": NUM_GENERATIONS,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "learning_rate": 5e-6,
            "lora_r": 128,
            "version": "v2",
        },
    )

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Load the relevant skills and submit your solution."},
    ]
    dataset = Dataset.from_dict({
        "prompt": [prompt_messages for _ in range(NUM_EPISODES)]
    })

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=1,
        generation_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        max_tool_calling_iterations=8,
        logging_steps=1,
        save_steps=50,
        report_to="wandb",
        temperature=0.7,
        log_completions=True,
        num_completions_to_print=2,
        # Thinking enabled — model needs reasoning to synthesize skill content
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        environment_factory=SkillEnv,
        model_init_kwargs={"quantization_config": bnb_config},
    )

    trainer.train()

    print("Training complete! Pushing to hub...")
    if HF_TOKEN:
        trainer.push_to_hub(HUB_REPO, token=HF_TOKEN)
        print(f"Model pushed to https://huggingface.co/{HUB_REPO}")
    else:
        print("HF_TOKEN not set, skipping push. Model saved locally.")
        trainer.save_model(OUTPUT_DIR)
