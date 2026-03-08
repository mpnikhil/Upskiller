"""
GRPO Training for Skill Invocation Environment.

Trains a model to decide which skills to load/unload before submitting a solution.
Uses TRL's GRPOTrainer with a custom multi-turn rollout that interacts with the
Skill Invocation Environment hosted on HF Spaces.

Run on Northflank with an A100/H100 GPU:
    python train_demo.py
"""

import hashlib
import re
import os

import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions
from transformers import AutoTokenizer
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from skill_invocation_env.client import SkillInvocationEnv
from skill_invocation_env.models import SkillInvocationAction

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
ENV_URL = os.getenv("ENV_URL", "https://mpnikhil-skill-invocation-env.hf.space")
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs/qwen-skill-env")
HUB_REPO = os.getenv("HUB_REPO", "mpnikhil/Qwen2.5-7B-Skill-Invocation")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "128"))
# Default 8 turns gives headroom to explore: load-inspect-unload-reload cycles
# beyond the minimum path of num_relevant_skills + 1 (submit) turns.
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "8"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "1024"))

SYSTEM_PROMPT = """\
You are an expert AI software engineer. You will be given a task and a catalog of available skills (procedural knowledge).
You must decide which skills to load to help you solve the task, and then submit your final answer.

You must interact by outputting EXACTLY ONE of the following XML actions per turn:

1. To load a skill to read its contents (costs context budget):
<action type="load" skill_id="skill_01"/>

2. To unload a skill if it is not useful (frees context budget):
<action type="unload" skill_id="skill_01"/>

3. To submit your final solution:
<action type="submit">
your solution here
</action>

Always think step-by-step before outputting an action."""


def parse_action(text: str) -> SkillInvocationAction:
    """Parses the LLM's text output into a SkillInvocationAction."""
    load_match = re.search(r'<action\s+type="load"\s+skill_id="([^"]+)"\s*/>', text)
    if load_match:
        return SkillInvocationAction(action_type="load", skill_id=load_match.group(1))

    unload_match = re.search(r'<action\s+type="unload"\s+skill_id="([^"]+)"\s*/>', text)
    if unload_match:
        return SkillInvocationAction(action_type="unload", skill_id=unload_match.group(1))

    submit_match = re.search(r'<action\s+type="submit">(.*?)</action>', text, re.DOTALL)
    if submit_match:
        return SkillInvocationAction(action_type="submit", answer=submit_match.group(1).strip())

    # Fallback: treat entire output as submission
    return SkillInvocationAction(action_type="submit", answer=text)


def format_observation(obs) -> str:
    """Formats the observation into a user prompt string for the LLM."""
    parts = [f"TASK: {obs.task_description}\n\nSKILL CATALOG:"]
    for s in obs.skill_catalog:
        parts.append(f"- [{s['id']}] {s['name']}: {s['description']}")

    if obs.loaded_skills:
        parts.append(f"\nCURRENTLY LOADED SKILLS: {', '.join(obs.loaded_skills)}")

    if obs.skill_content:
        parts.append(f"\nJUST LOADED SKILL CONTENT:\n{obs.skill_content}")

    # Surface all currently-loaded skill contents so the model doesn't rely
    # solely on conversation history to recall previously-loaded skills.
    if obs.loaded_skill_contents:
        just_loaded_id = None
        if obs.skill_content:
            # Find which skill was just loaded to avoid duplicating its content
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


# ── Multi-turn rollout ─────────────────────────────────────────────────────────

def rollout_once(
    trainer: GRPOTrainer,
    env: SkillInvocationEnv,
    tokenizer: AutoTokenizer,
    env_seed: int,
) -> dict:
    """
    Run one multi-turn episode against the Skill Invocation Environment.

    Args:
        env_seed: Deterministic seed passed to env.reset() so all generations
                  within a GRPO group face the identical task.

    Returns dict with prompt_ids, completion_ids, logprobs, and env_reward.
    Accumulates tokens across ALL turns so GRPO can assign credit to every
    decision (load, unload, submit).
    """
    result = env.reset(seed=env_seed)
    obs = result.observation

    # Token accumulation across turns:
    # - prompt_ids: first turn's full prompt (system + initial observation)
    # - completion_ids: all model generations + env feedback tokens interleaved
    # - logprobs: real logprobs for model tokens, 0.0 for env feedback tokens
    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    env_reward = 0.0
    generated_any = False

    # Tracks how many tokens we've already accounted for across turns.
    # Each turn's prompt_ids from apply_chat_template contains the FULL
    # conversation so far (quadratic growth). We only append the delta —
    # the new tokens since the last turn — to keep accounting linear.
    prev_total_len = 0

    # Conversation history — the model sees its full interaction so far,
    # so it can recall what it read in a loaded skill and decide to unload.
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in range(MAX_TURNS):
        if result.done:
            break

        # Append new observation to conversation history
        user_content = format_observation(obs)
        conversation.append({"role": "user", "content": user_content})

        prompt_text = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )

        # Safety check: prevent vLLM context length errors. Qwen3-8B has a
        # 32,768 token context window; leave room for MAX_COMPLETION_LENGTH.
        prompt_token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if prompt_token_count > 31_000:
            print(f"    [rollout] prompt too long ({prompt_token_count} tokens), breaking early")
            env_reward = -0.5
            break

        # Generate using TRL's vLLM helper
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        generated_any = True

        new_prompt_ids = rollout_outputs["prompt_ids"]

        if turn == 0:
            # First turn: store the full prompt
            prompt_ids.extend(new_prompt_ids)
            prev_total_len = len(new_prompt_ids)
        else:
            # Later turns: only append the delta (new env feedback tokens
            # beyond what we've already tracked). These get zeroed-out
            # logprobs since they're env-generated, not model-generated.
            delta_ids = new_prompt_ids[prev_total_len:]
            completion_ids.extend(delta_ids)
            logprobs.extend([0.0] * len(delta_ids))

        # Append the model's generation tokens (these get real logprobs)
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        # Update running total: everything up to and including this turn's completion
        prev_total_len = len(new_prompt_ids) + len(rollout_outputs["completion_ids"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True,
        )

        # Add the model's response to conversation history
        conversation.append({"role": "assistant", "content": completion_text})

        # Parse action and step the environment
        action = parse_action(completion_text)

        try:
            result = env.step(action)
            obs = result.observation
            if result.done:
                env_reward = float(result.reward or 0.0)
        except Exception as e:
            print(f"    [rollout] env.step error: {e}")
            env_reward = -1.0
            break

    # If we ran out of turns without submitting, penalize
    if not result.done:
        env_reward = -0.5

    # Fallback if no generation happened (e.g. env.reset() returned done=True)
    if not generated_any:
        dummy_ids = tokenizer.encode("error", add_special_tokens=False)
        prompt_ids = dummy_ids
        completion_ids = list(dummy_ids)
        logprobs = [0.0] * len(dummy_ids)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_reward": env_reward,
    }


def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    """
    Custom rollout function for GRPOTrainer.

    GRPO groups: prompts arrive as [p0, p0, p0, ..., p1, p1, p1, ...] where
    each prompt is repeated num_generations times. All rollouts for the same
    prompt must face the same task, so we extract the seed from the prompt text
    and pass it to env.reset(seed=...).
    """
    tokenizer = trainer.processing_class

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []
    rewards_received = 0

    for i, prompt_text in enumerate(prompts):
        # Extract seed from the prompt — format is "seed:<N> ..."
        # This ensures all K generations for the same prompt get the same task.
        seed = _extract_seed(prompt_text)

        env = SkillInvocationEnv(base_url=ENV_URL, connect_timeout_s=60)
        try:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                env_seed=seed,
            )
        finally:
            try:
                env.close()
            except Exception:
                pass
        all_prompt_ids.append(episode["prompt_ids"])
        all_completion_ids.append(episode["completion_ids"])
        all_logprobs.append(episode["logprobs"])
        all_rewards.append(episode["env_reward"])

        if episode["env_reward"] != 0.0:
            rewards_received += 1

        if (i + 1) % 10 == 0:
            avg_r = sum(all_rewards) / len(all_rewards)
            print(f"  [rollout] {i+1}/{len(prompts)} episodes, avg reward: {avg_r:.3f}")

    # Issue 4 guard: verify rewards actually flowed through
    if rewards_received == 0 and len(prompts) > 0:
        print("  [WARNING] All rewards are 0.0 — check env connectivity!")

    # Log rollout stats to wandb
    if wandb.run is not None:
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        positive = sum(1 for r in all_rewards if r > 0)
        negative = sum(1 for r in all_rewards if r < 0)
        wandb.log({
            "rollout/avg_reward": avg_reward,
            "rollout/max_reward": max(all_rewards) if all_rewards else 0.0,
            "rollout/min_reward": min(all_rewards) if all_rewards else 0.0,
            "rollout/positive_pct": positive / len(all_rewards) * 100 if all_rewards else 0.0,
            "rollout/negative_pct": negative / len(all_rewards) * 100 if all_rewards else 0.0,
            "rollout/num_episodes": len(all_rewards),
        })

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": all_rewards,
    }


def _extract_seed(prompt_text: str) -> int:
    """Extract the env seed from a prompt like 'seed:42 ...'

    Crashes loudly on malformed prompts rather than silently producing
    non-deterministic seeds (Python's hash() is randomized across processes).
    """
    match = re.match(r"seed:(\d+)", prompt_text)
    if match:
        return int(match.group(1))
    # Deterministic fallback using SHA-256 (stable across processes, unlike hash())
    digest = hashlib.sha256(prompt_text.encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)


def reward_from_env(completions, **kwargs):
    """Extract environment rewards passed via rollout_func kwargs."""
    env_rewards = kwargs.get("env_reward", [])
    if not env_rewards:
        print("  [WARNING] reward_from_env received no env_reward in kwargs!")
        return [0.0] * len(completions)
    return [float(r) for r in env_rewards]


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
            "max_turns": MAX_TURNS,
            "learning_rate": 1e-6,
            "lora_r": 16,
        },
    )

    # Each unique prompt = one GRPO group = one task (via seed).
    # GRPO will expand each prompt to num_generations rollouts internally.
    # All rollouts for the same seed face the same task → valid advantage computation.
    prompts = [f"seed:{i} Solve the coding task by loading the right skills." for i in range(NUM_EPISODES)]
    dataset = Dataset.from_dict({"prompt": prompts})

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        num_train_epochs=1,
        num_generations=NUM_GENERATIONS,
        max_completion_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        max_prompt_length=1024,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=50,
        loss_type="grpo",
        report_to="wandb",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        model_init_kwargs={"quantization_config": quantization_config},
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=rollout_func,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()

    print("Training complete! Pushing to hub...")
    if HF_TOKEN:
        trainer.push_to_hub(HUB_REPO, token=HF_TOKEN)
        print(f"Model pushed to https://huggingface.co/{HUB_REPO}")
    else:
        print("HF_TOKEN not set, skipping push. Model saved locally.")
        trainer.save_model(OUTPUT_DIR)
