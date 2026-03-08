"""
GRPO Training for Skill Invocation Environment.

Trains a model to decide which skills to load/unload before submitting a solution.
Uses TRL's GRPOTrainer with environment_factory for native multi-turn tool calling.

Prerequisites:
    pip install "transformers>=5.2.0" "trl>=0.26.0" "peft>=0.15.0" "datasets" "wandb"
    (No vLLM needed — uses HF generate())

Run on Northflank / Jupyter with GPU:
    python train_demo.py
"""

import os

import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from skill_invocation_env.client import SkillInvocationEnv
from skill_invocation_env.models import SkillInvocationAction

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B")
ENV_URL = os.getenv("ENV_URL", "https://mpnikhil-skill-invocation-env.hf.space")
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs/qwen-skill-env")
HUB_REPO = os.getenv("HUB_REPO", "mpnikhil/Qwen2.5-3B-Skill-Invocation")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "64"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "2048"))

SYSTEM_PROMPT = """\
You are given a task and a catalog of skills (procedural knowledge). \
Each skill has an ID, name, and description. You can load skills to read their full contents, \
which will help you solve the task correctly. Loading a skill costs context budget, so only load what you need. \
When ready, submit your solution.

Think step-by-step: identify which skills are relevant from their descriptions, load them, \
read the contents carefully, then submit an answer that uses the specific details from the loaded skills."""


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
        # Close previous connection if any, then open fresh one
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
            Updated environment state with the loaded skill content.
        """
        if self.done:
            raise ValueError("Episode is over.")
        action = SkillInvocationAction(action_type="load", skill_id=skill_id)
        result = self.client.step(action)
        self.done = result.done
        self.reward = float(result.reward or 0.0)
        return format_observation(result.observation)

    def unload_skill(self, skill_id: str) -> str:
        """Unload a skill to free context budget.

        Args:
            skill_id: The ID of the skill to unload (e.g. 'skill_01')

        Returns:
            Updated environment state after unloading.
        """
        if self.done:
            raise ValueError("Episode is over.")
        action = SkillInvocationAction(action_type="unload", skill_id=skill_id)
        result = self.client.step(action)
        self.done = result.done
        self.reward = float(result.reward or 0.0)
        return format_observation(result.observation)

    def submit(self, answer: str) -> str:
        """Submit your final solution to the task.

        Args:
            answer: Your solution to the task

        Returns:
            Verification result with your score.
        """
        if self.done:
            raise ValueError("Episode is over.")
        action = SkillInvocationAction(action_type="submit", answer=answer)
        result = self.client.step(action)
        self.done = result.done
        self.reward = float(result.reward or 0.0)
        return format_observation(result.observation)


def reward_func(environments, **kwargs) -> list[float]:
    """Extract rewards from environment instances."""
    return [env.reward for env in environments]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting GRPO Training with {MODEL_ID}")
    print(f"Environment: {ENV_URL}")
    print(f"Episodes: {NUM_EPISODES}, Generations per episode: {NUM_GENERATIONS}")

    wandb.init(
        project="skill-invocation-env",
        name=f"grpo-{MODEL_ID.split('/')[-1]}-ep{NUM_EPISODES}",
        config={
            "model_id": MODEL_ID,
            "env_url": ENV_URL,
            "num_episodes": NUM_EPISODES,
            "num_generations": NUM_GENERATIONS,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "learning_rate": 1e-6,
            "lora_r": 16,
        },
    )

    # All prompts identical — task variation comes from env.reset()
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
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=50,
        report_to="wandb",
        temperature=0.7,
        log_completions=True,
        num_completions_to_print=2,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        environment_factory=SkillEnv,
    )

    trainer.train()

    print("Training complete! Pushing to hub...")
    if HF_TOKEN:
        trainer.push_to_hub(HUB_REPO, token=HF_TOKEN)
        print(f"Model pushed to https://huggingface.co/{HUB_REPO}")
    else:
        print("HF_TOKEN not set, skipping push. Model saved locally.")
        trainer.save_model(OUTPUT_DIR)
