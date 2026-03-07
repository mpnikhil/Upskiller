#!/usr/bin/env python3
"""
Local test script for the Skill Invocation Environment.

Tests the environment directly (no server) to verify:
- reset() works and returns proper observation
- list/load/unload/submit actions work correctly
- context budget enforcement
- precision/recall/bloat reward computation
- verifier tests for static and procedural tasks
"""

import sys
import os

# Add parent dir so imports work
sys.path.insert(0, os.path.dirname(__file__))

from models import SkillInvocationAction, SkillInvocationObservation, SkillInvocationState
from task_bank import TASK_BANK, SKILL_BANK
from server.skill_invocation_env_environment import SkillInvocationEnvironment
from task_generator import TaskGenerator


# ---------------------------------------------------------------------------
# Core environment tests
# ---------------------------------------------------------------------------


def test_reset():
    """Test that reset returns a valid observation."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    assert isinstance(obs, SkillInvocationObservation)
    assert obs.task_description != ""
    assert len(obs.skill_catalog) >= 5  # relevant + distractors (now 5-8)
    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.skill_content is None
    assert obs.loaded_skills == []
    assert obs.context_budget_used == 0
    assert obs.context_budget_total == 5
    assert len(obs.messages) > 0

    print("[PASS] test_reset")


def test_list_action():
    """Test that list action returns catalog without state change."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    action = SkillInvocationAction(action_type="list")
    obs2 = env.step(action)

    assert obs2.skill_content is None
    assert len(obs2.skill_catalog) == len(obs.skill_catalog)
    assert obs2.loaded_skills == []
    assert obs2.context_budget_used == 0
    assert obs2.done is False

    print("[PASS] test_list_action")


def test_load_skill():
    """Test loading a skill puts it in context."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    skill_id = obs.skill_catalog[0]["id"]
    action = SkillInvocationAction(action_type="load", skill_id=skill_id)
    obs2 = env.step(action)

    assert obs2.skill_content is not None
    assert len(obs2.skill_content) > 0
    assert skill_id in obs2.loaded_skills
    assert obs2.context_budget_used == 1
    assert skill_id in obs2.loaded_skill_contents
    assert obs2.done is False

    print("[PASS] test_load_skill")


def test_invoke_backward_compat():
    """Test that 'invoke' still works as alias for 'load'."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    skill_id = obs.skill_catalog[0]["id"]
    action = SkillInvocationAction(action_type="invoke", skill_id=skill_id)
    obs2 = env.step(action)

    assert obs2.skill_content is not None
    assert skill_id in obs2.loaded_skills
    assert obs2.context_budget_used == 1

    print("[PASS] test_invoke_backward_compat")


def test_unload_skill():
    """Test unloading a skill removes it from context."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    skill_id = obs.skill_catalog[0]["id"]

    # Load
    env.step(SkillInvocationAction(action_type="load", skill_id=skill_id))
    # Unload
    obs3 = env.step(SkillInvocationAction(action_type="unload", skill_id=skill_id))

    assert skill_id not in obs3.loaded_skills
    assert obs3.context_budget_used == 0
    assert obs3.skill_content is None
    # Should still be in skills_ever_loaded (history)
    assert skill_id in obs3.skills_invoked

    print("[PASS] test_unload_skill")


def test_load_already_loaded():
    """Loading same skill twice is a no-op (no double counting)."""
    env = SkillInvocationEnvironment()
    obs = env.reset(seed=42)

    skill_id = obs.skill_catalog[0]["id"]
    env.step(SkillInvocationAction(action_type="load", skill_id=skill_id))
    obs2 = env.step(SkillInvocationAction(action_type="load", skill_id=skill_id))

    assert obs2.context_budget_used == 1  # Not 2
    assert obs2.loaded_skills.count(skill_id) == 1
    assert obs2.skill_content is not None  # Still returns content

    print("[PASS] test_load_already_loaded")


def test_unload_not_loaded():
    """Unloading a skill that isn't loaded is a no-op."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    obs = env.step(SkillInvocationAction(action_type="unload", skill_id="skill_001"))
    assert obs.context_budget_used == 0

    print("[PASS] test_unload_not_loaded")


def test_context_budget():
    """Test that context budget is enforced."""
    env = SkillInvocationEnvironment(context_budget=3)
    obs = env.reset(seed=42)

    catalog_ids = [s["id"] for s in obs.skill_catalog]

    # Load 3 skills (budget full)
    for i in range(min(3, len(catalog_ids))):
        env.step(SkillInvocationAction(action_type="load", skill_id=catalog_ids[i]))

    obs = env.step(SkillInvocationAction(action_type="load", skill_id=catalog_ids[3]))
    # Should fail — budget is full
    assert obs.context_budget_used == 3
    assert catalog_ids[3] not in obs.loaded_skills

    # Unload one, then load should work
    env.step(SkillInvocationAction(action_type="unload", skill_id=catalog_ids[0]))
    obs2 = env.step(SkillInvocationAction(action_type="load", skill_id=catalog_ids[3]))
    assert catalog_ids[3] in obs2.loaded_skills
    assert obs2.context_budget_used == 3

    print("[PASS] test_context_budget")


def test_load_unknown_skill():
    """Test loading a skill not in the catalog."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    action = SkillInvocationAction(action_type="load", skill_id="skill_999")
    obs = env.step(action)

    assert obs.skill_content is None
    assert obs.context_budget_used == 0

    print("[PASS] test_load_unknown_skill")


def test_submit_incorrect():
    """Test submitting an incorrect answer."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    action = SkillInvocationAction(action_type="submit", answer="I don't know")
    obs = env.step(action)

    assert obs.done is True
    assert obs.reward <= 0.0
    assert obs.verification_result is not None
    assert "INCORRECT" in obs.verification_result

    print("[PASS] test_submit_incorrect")


def test_submit_after_done():
    """Test that actions after done return done state."""
    env = SkillInvocationEnvironment()
    env.reset(seed=42)

    env.step(SkillInvocationAction(action_type="submit", answer="test"))

    obs = env.step(SkillInvocationAction(action_type="load", skill_id="skill_001"))
    assert obs.done is True

    print("[PASS] test_submit_after_done")


def test_precision_reward():
    """Load only relevant skill, submit correct answer → max reward 1.0."""
    env = SkillInvocationEnvironment()

    for seed in range(100):
        obs = env.reset(seed=seed)
        state = env.state
        if state.task_id == "task_001":
            break
    else:
        print("[SKIP] test_precision_reward - couldn't find task_001")
        return

    # Load only relevant skill
    env.step(SkillInvocationAction(action_type="load", skill_id="skill_001"))

    correct_answer = """
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
"""
    obs = env.step(SkillInvocationAction(action_type="submit", answer=correct_answer))

    assert obs.done is True
    assert "CORRECT" in obs.verification_result
    # 0.6 correctness + 0.3 precision (1/1) + 0.1 recall (1/1) = 1.0
    assert abs(obs.reward - 1.0) < 0.01, f"Expected ~1.0, got {obs.reward}"

    print(f"[PASS] test_precision_reward (reward={obs.reward})")


def test_bloat_penalty():
    """Load all catalog skills, submit correct answer → reduced reward."""
    env = SkillInvocationEnvironment()

    for seed in range(100):
        obs = env.reset(seed=seed)
        state = env.state
        if state.task_id == "task_001":
            break
    else:
        print("[SKIP] test_bloat_penalty - couldn't find task_001")
        return

    # Load all catalog skills (relevant + distractors)
    for skill in obs.skill_catalog:
        env.step(SkillInvocationAction(action_type="load", skill_id=skill["id"]))

    correct_answer = """
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
"""
    obs = env.step(SkillInvocationAction(action_type="submit", answer=correct_answer))

    assert obs.done is True
    assert "CORRECT" in obs.verification_result
    # With 6 total skills loaded (1 relevant + 5 distractors):
    # 0.6 + 0.3*(1/6) + 0.1*(1/1) - 0.15*5 = 0.6 + 0.05 + 0.1 - 0.75 = 0.0
    # Reward should be much less than 1.0
    assert obs.reward < 0.5, f"Bloat should reduce reward, got {obs.reward}"

    print(f"[PASS] test_bloat_penalty (reward={obs.reward})")


def test_load_unload_no_bloat():
    """Load distractor, unload before submit → no bloat penalty."""
    env = SkillInvocationEnvironment()

    for seed in range(100):
        obs = env.reset(seed=seed)
        state = env.state
        if state.task_id == "task_001":
            break
    else:
        print("[SKIP] test_load_unload_no_bloat - couldn't find task_001")
        return

    # Load a distractor
    distractor_id = None
    for skill in obs.skill_catalog:
        if skill["id"] != "skill_001":
            distractor_id = skill["id"]
            break
    assert distractor_id is not None

    env.step(SkillInvocationAction(action_type="load", skill_id=distractor_id))
    # Unload it
    env.step(SkillInvocationAction(action_type="unload", skill_id=distractor_id))
    # Load relevant
    env.step(SkillInvocationAction(action_type="load", skill_id="skill_001"))

    correct_answer = """
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
"""
    obs = env.step(SkillInvocationAction(action_type="submit", answer=correct_answer))

    assert obs.done is True
    assert "CORRECT" in obs.verification_result
    # Only skill_001 loaded at submit → no bloat
    # 0.6 + 0.3 + 0.1 = 1.0
    assert abs(obs.reward - 1.0) < 0.01, f"Expected ~1.0 after unload, got {obs.reward}"

    print(f"[PASS] test_load_unload_no_bloat (reward={obs.reward})")


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
    assert state.loaded_skills == []
    assert state.context_budget_total == 5

    # After a step
    skill_id = obs.skill_catalog[0]["id"]
    env.step(SkillInvocationAction(action_type="load", skill_id=skill_id))

    state = env.state
    assert state.step_count == 1
    assert skill_id in state.loaded_skills

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
        assert len(overlap) == 0, f"Task {task['id']}: overlap: {overlap}"
        # Each task now should have at least 5 skills in catalog
        total = len(task["relevant_skills"]) + len(task["distractor_skills"])
        assert total >= 5, f"Task {task['id']}: only {total} skills in catalog"

    print(f"[PASS] test_all_tasks_have_valid_skills ({len(TASK_BANK)} tasks verified)")


# ---------------------------------------------------------------------------
# Verifier tests (unchanged — these test verifier correctness, not env logic)
# ---------------------------------------------------------------------------


def test_verifier_task001_correct_code_passes():
    """Verify task_001 exec verifier passes reference implementation from skill content."""
    task = next(t for t in TASK_BANK if t["id"] == "task_001")
    correct_code = '''
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
'''
    assert task["verifier"](correct_code), "Reference implementation should pass"
    print("[PASS] test_verifier_task001_correct_code_passes")


def test_verifier_task001_keywords_only_fails():
    """Verify task_001 exec verifier rejects keyword-stuffed garbage."""
    task = next(t for t in TASK_BANK if t["id"] == "task_001")
    garbage = "hmac sha256 x-zephyr-auth base64 zph encode_zephyr_auth"
    assert not task["verifier"](garbage), "Keyword-stuffed garbage should fail"
    print("[PASS] test_verifier_task001_keywords_only_fails")


def test_verifier_task001_wrong_format_fails():
    """Verify task_001 rejects code with wrong header format."""
    task = next(t for t in TASK_BANK if t["id"] == "task_001")
    wrong_code = '''
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.md5).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
'''
    assert not task["verifier"](wrong_code), "Wrong hash algorithm should fail"
    print("[PASS] test_verifier_task001_wrong_format_fails")


def test_verifier_task001_markdown_fenced():
    """Verify task_001 exec verifier handles markdown-fenced code."""
    task = next(t for t in TASK_BANK if t["id"] == "task_001")
    fenced = '''```python
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
```'''
    assert task["verifier"](fenced), "Markdown-fenced correct code should pass"
    print("[PASS] test_verifier_task001_markdown_fenced")


def test_verifier_task002_correct_passes():
    """Verify task_002 NovaBin header parser passes with correct implementation."""
    task = next(t for t in TASK_BANK if t["id"] == "task_002")
    correct_code = '''
import struct

def parse_novabin_header(data: bytes) -> dict:
    magic = data[0:4]
    assert magic == b'NOVB', f"Invalid magic: {magic}"
    version = struct.unpack('>H', data[4:6])[0]
    record_count = struct.unpack('>I', data[6:10])[0]
    flags = struct.unpack('>H', data[10:12])[0]
    checksum = struct.unpack('>I', data[12:16])[0]
    return {
        "version": version, "record_count": record_count,
        "compressed": bool(flags & 1), "encrypted": bool(flags & 2),
        "checksummed": bool(flags & 4), "checksum": checksum
    }
'''
    assert task["verifier"](correct_code), "Correct NovaBin parser should pass"
    print("[PASS] test_verifier_task002_correct_passes")


def test_verifier_task002_keywords_only_fails():
    """Verify task_002 rejects keyword-stuffed answer."""
    task = next(t for t in TASK_BANK if t["id"] == "task_002")
    garbage = "struct NOVB 0x4E4F5642 big-endian parse_novabin_header version record_count"
    assert not task["verifier"](garbage), "Keyword-stuffed answer should fail"
    print("[PASS] test_verifier_task002_keywords_only_fails")


def test_verifier_task003_structural():
    """Verify task_003 HelixLang structural verifier catches structure, not just keywords."""
    task = next(t for t in TASK_BANK if t["id"] == "task_003")

    good = '''
fn fetch_user(db: Database, user_id: str) -> result<User> {
    let conn = try! db.connect().with_context("step", "connecting to database")

    let user = match try! conn.query_user(user_id).with_context("step", "fetching user") {
        Ok(u) => u,
        Err(e) => {
            if e.retryable {
                return retry_with_backoff(|| conn.query_user(user_id), max=3, backoff=100ms)
            }
            helix.log.error(e)
            return Err(HelixError.wrap(e, "HLX-DATA-2001", "user fetch failed"))
        }
    }

    Ok(user)
}
'''
    assert task["verifier"](good), "Proper HelixLang pseudocode should pass"

    keywords_only = "HLX-DATA try! with_context retry backoff helix.log.error result Ok Err"
    assert not task["verifier"](keywords_only), "Keywords without structure should fail"

    print("[PASS] test_verifier_task003_structural")


def test_verifier_task004_yaml_structure():
    """Verify task_004 ArcDeploy YAML verifier checks structure."""
    task = next(t for t in TASK_BANK if t["id"] == "task_004")

    good_yaml = '''```yaml
canary:
  phases:
    - name: shadow
      traffic_pct: 0
      duration_min: 5
      metrics_gate: error_rate < 0.01
    - name: canary_1
      traffic_pct: 5
      duration_min: 10
      metrics_gate: p99_latency_ms < 200 AND error_rate < 0.005
    - name: canary_2
      traffic_pct: 25
      duration_min: 15
      metrics_gate: p99_latency_ms < 250 AND error_rate < 0.005
    - name: canary_3
      traffic_pct: 50
      duration_min: 20
      metrics_gate: p99_latency_ms < 300 AND error_rate < 0.01
    - name: full
      traffic_pct: 100
      duration_min: 0
  rollback:
    auto: true
    on_metric_breach: immediate
    cooldown_min: 30
```'''
    assert task["verifier"](good_yaml), "Valid ArcDeploy YAML should pass"

    keywords = "shadow canary_1 traffic_pct metrics_gate error_rate rollback auto: true"
    assert not task["verifier"](keywords), "Keywords-only should fail YAML verifier"

    print("[PASS] test_verifier_task004_yaml_structure")


def test_verifier_task008_record_parser():
    """Verify task_008 NovaBin record parser with exec verifier."""
    task = next(t for t in TASK_BANK if t["id"] == "task_008")

    correct_code = '''
import struct

def parse_novabin_record(data: bytes, offset: int) -> tuple:
    fields = {}
    field_count = struct.unpack('>H', data[offset:offset+2])[0]
    offset += 2

    for _ in range(field_count):
        type_tag = data[offset]
        offset += 1

        name_len = struct.unpack('>H', data[offset:offset+2])[0]
        offset += 2
        field_name = data[offset:offset+name_len].decode('utf-8')
        offset += name_len

        val_len = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        val_data = data[offset:offset+val_len]
        offset += val_len

        if type_tag == 0x01:  # int32
            fields[field_name] = struct.unpack('>i', val_data)[0]
        elif type_tag == 0x02:  # float64
            fields[field_name] = struct.unpack('>d', val_data)[0]
        elif type_tag == 0x03:  # string
            fields[field_name] = val_data.decode('utf-8')
        elif type_tag == 0x04:  # bool
            fields[field_name] = val_data[0] != 0

    return (fields, offset)
'''
    assert task["verifier"](correct_code), "Correct record parser should pass"

    keywords = "struct 0x01 0x02 0x03 0x04 uint16 utf-8 parse_novabin_record"
    assert not task["verifier"](keywords), "Keywords should fail exec verifier"

    print("[PASS] test_verifier_task008_record_parser")


# ---------------------------------------------------------------------------
# SkillsBench-adapted task tests
# ---------------------------------------------------------------------------

def test_sb_001_flood_detection_correct():
    """Verify task_sb_001 flood detection passes with correct implementation."""
    task = next(t for t in TASK_BANK if t["id"] == "task_sb_001")
    correct_code = '''
def detect_flood_days(daily_max_levels, flood_thresholds):
    result = {}
    for station_id, levels in daily_max_levels.items():
        if station_id not in flood_thresholds:
            continue
        threshold = flood_thresholds[station_id]
        flood_days = sum(1 for level in levels if level >= threshold)
        if flood_days > 0:
            result[station_id] = flood_days
    return result
'''
    assert task["verifier"](correct_code), "Correct flood detection should pass"

    garbage = "detect_flood_days daily_max_levels flood_thresholds threshold"
    assert not task["verifier"](garbage), "Keywords should fail"

    print("[PASS] test_sb_001_flood_detection_correct")


def test_sb_002_hp_filter_correct():
    """Verify task_sb_002 HP filter correlation passes with correct implementation."""
    try:
        import numpy  # noqa: F401
        from statsmodels.tsa.filters.hp_filter import hpfilter  # noqa: F401
    except ImportError:
        print("[SKIP] test_sb_002_hp_filter_correct - scipy/statsmodels not installed")
        return

    task = next(t for t in TASK_BANK if t["id"] == "task_sb_002")
    correct_code = '''
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

def hp_filter_correlation(series_a, series_b):
    log_a = np.log(series_a)
    log_b = np.log(series_b)
    cycle_a, _ = hpfilter(log_a, lamb=100)
    cycle_b, _ = hpfilter(log_b, lamb=100)
    corr = np.corrcoef(cycle_a, cycle_b)[0, 1]
    return round(float(corr), 5)
'''
    assert task["verifier"](correct_code), "Correct HP filter implementation should pass"

    garbage = "hp_filter_correlation numpy hpfilter corrcoef lamb=100"
    assert not task["verifier"](garbage), "Keywords should fail"

    print("[PASS] test_sb_002_hp_filter_correct")


def test_sb_003_dialogue_parser_correct():
    """Verify task_sb_003 dialogue parser passes with correct implementation."""
    task = next(t for t in TASK_BANK if t["id"] == "task_sb_003")
    correct_code = (
        'import re\n'
        '\n'
        'def parse_dialogue(script):\n'
        '    nodes = []\n'
        '    edges = []\n'
        '    lines = script.strip().split("\\n")\n'
        '    current_node_id = None\n'
        '    current_lines = []\n'
        '    def flush_node():\n'
        '        nonlocal current_node_id, current_lines\n'
        '        if current_node_id is None:\n'
        '            return\n'
        '        content_lines = [l.strip() for l in current_lines if l.strip()]\n'
        '        is_choice = any(re.match(r"^\\d+\\.", l) for l in content_lines)\n'
        '        if is_choice:\n'
        '            nodes.append({"id": current_node_id, "text": "", "speaker": "", "type": "choice"})\n'
        '            for l in content_lines:\n'
        '                m = re.match(r"^(\\d+\\.\\s*.+?)\\s*->\\s*(\\w+)$", l)\n'
        '                if m:\n'
        '                    edges.append({"from": current_node_id, "to": m.group(2), "text": m.group(1).strip()})\n'
        '        else:\n'
        '            speaker = ""\n'
        '            text = ""\n'
        '            target = None\n'
        '            for l in content_lines:\n'
        '                m = re.match(r"^(\\w[\\w\\s]*):\\s*(.+?)\\s*->\\s*(\\w+)$", l)\n'
        '                if m:\n'
        '                    speaker = m.group(1)\n'
        '                    text = m.group(2).strip()\n'
        '                    target = m.group(3)\n'
        '                else:\n'
        '                    m2 = re.match(r"^(\\w[\\w\\s]*):\\s*(.+)$", l)\n'
        '                    if m2:\n'
        '                        speaker = m2.group(1)\n'
        '                        text = m2.group(2).strip()\n'
        '            nodes.append({"id": current_node_id, "text": text, "speaker": speaker, "type": "line"})\n'
        '            if target:\n'
        '                edges.append({"from": current_node_id, "to": target, "text": ""})\n'
        '        current_node_id = None\n'
        '        current_lines = []\n'
        '    for line in lines:\n'
        '        m = re.match(r"^\\[(\\w+)\\]$", line.strip())\n'
        '        if m:\n'
        '            flush_node()\n'
        '            current_node_id = m.group(1)\n'
        '            current_lines = []\n'
        '        else:\n'
        '            current_lines.append(line)\n'
        '    flush_node()\n'
        '    return {"nodes": nodes, "edges": edges}\n'
    )
    assert task["verifier"](correct_code), "Correct dialogue parser should pass"

    garbage = "parse_dialogue nodes edges from to text speaker type"
    assert not task["verifier"](garbage), "Keywords should fail"

    print("[PASS] test_sb_003_dialogue_parser_correct")


# ---------------------------------------------------------------------------
# Procedural task generator tests
# ---------------------------------------------------------------------------


def test_procedural_auth_100_seeds():
    """Test auth protocol template produces valid, verifiable tasks for 100 seeds."""
    gen = TaskGenerator(seed=0)
    for seed in range(100):
        result = gen.generate_with_seed(seed, template="auth_protocol")
        task = result["task"]
        skills = result["skills"]

        assert task["id"].startswith("task_proc_auth_")
        assert task["source"] == "procedural"
        assert task["template"] == "auth_protocol"
        assert len(task["relevant_skills"]) == 1
        assert len(task["distractor_skills"]) >= 4

        for sid in task["relevant_skills"] + task["distractor_skills"]:
            assert sid in skills, f"Skill {sid} not in generated skills for seed {seed}"

        rel_skill = skills[task["relevant_skills"][0]]
        assert len(rel_skill["full_content"]) > 100

    print("[PASS] test_procedural_auth_100_seeds")


def test_procedural_binary_100_seeds():
    """Test binary format template produces valid tasks for 100 seeds."""
    gen = TaskGenerator(seed=0)
    for seed in range(100):
        result = gen.generate_with_seed(seed, template="binary_format")
        task = result["task"]
        skills = result["skills"]

        assert task["id"].startswith("task_proc_bin_")
        assert task["source"] == "procedural"
        assert len(task["relevant_skills"]) == 1
        assert len(task["distractor_skills"]) >= 4

        for sid in task["relevant_skills"] + task["distractor_skills"]:
            assert sid in skills

    print("[PASS] test_procedural_binary_100_seeds")


def test_procedural_deterministic():
    """Same seed produces identical tasks."""
    gen = TaskGenerator(seed=0)
    r1 = gen.generate_with_seed(42, template="auth_protocol")
    r2 = gen.generate_with_seed(42, template="auth_protocol")

    assert r1["task"]["id"] == r2["task"]["id"]
    assert r1["task"]["description"] == r2["task"]["description"]
    assert r1["task"]["relevant_skills"] == r2["task"]["relevant_skills"]
    assert r1["task"]["distractor_skills"] == r2["task"]["distractor_skills"]

    r3 = gen.generate_with_seed(42, template="binary_format")
    r4 = gen.generate_with_seed(42, template="binary_format")
    assert r3["task"]["id"] == r4["task"]["id"]
    assert r3["task"]["description"] == r4["task"]["description"]

    print("[PASS] test_procedural_deterministic")


def test_procedural_keyword_stuffing_fails():
    """Keyword-stuffed answers should fail procedural verifiers."""
    gen = TaskGenerator(seed=0)

    for seed in range(10):
        result = gen.generate_with_seed(seed, template="auth_protocol")
        task = result["task"]
        garbage = "HMAC SHA256 base64 signing API key authentication header"
        assert not task["verifier"](garbage), f"Keyword stuffing passed for auth seed {seed}"

        result = gen.generate_with_seed(seed, template="binary_format")
        task = result["task"]
        garbage = "struct unpack CRC32 magic bytes header version flags"
        assert not task["verifier"](garbage), f"Keyword stuffing passed for binary seed {seed}"

    print("[PASS] test_procedural_keyword_stuffing_fails")


def test_procedural_env_integration():
    """Test environment works with use_procedural=True."""
    env = SkillInvocationEnvironment(use_procedural=True, procedural_seed=42)
    obs = env.reset(seed=100)

    assert isinstance(obs, SkillInvocationObservation)
    assert obs.task_description != ""
    assert len(obs.skill_catalog) >= 5
    assert obs.context_budget_total == 5
    assert obs.done is False

    skill_id = obs.skill_catalog[0]["id"]
    obs2 = env.step(SkillInvocationAction(action_type="load", skill_id=skill_id))
    assert obs2.skill_content is not None
    assert len(obs2.skill_content) > 50
    assert obs2.context_budget_used == 1

    obs3 = env.step(SkillInvocationAction(action_type="submit", answer="test"))
    assert obs3.done is True
    assert obs3.reward is not None

    print("[PASS] test_procedural_env_integration")


def test_procedural_uniqueness():
    """Different seeds produce different tasks."""
    gen = TaskGenerator(seed=0)
    descriptions = set()
    for seed in range(50):
        result = gen.generate_with_seed(seed, template="auth_protocol")
        descriptions.add(result["task"]["description"])

    assert len(descriptions) >= 10, f"Only {len(descriptions)} unique tasks from 50 seeds"

    print("[PASS] test_procedural_uniqueness")


if __name__ == "__main__":
    print("=" * 60)
    print("Skill Invocation Environment - Local Tests")
    print("=" * 60)

    tests = [
        # Core environment tests
        test_reset,
        test_list_action,
        test_load_skill,
        test_invoke_backward_compat,
        test_unload_skill,
        test_load_already_loaded,
        test_unload_not_loaded,
        test_context_budget,
        test_load_unknown_skill,
        test_submit_incorrect,
        test_submit_after_done,
        test_precision_reward,
        test_bloat_penalty,
        test_load_unload_no_bloat,
        test_state_property,
        test_all_tasks_have_valid_skills,
        # Verifier tests
        test_verifier_task001_correct_code_passes,
        test_verifier_task001_keywords_only_fails,
        test_verifier_task001_wrong_format_fails,
        test_verifier_task001_markdown_fenced,
        test_verifier_task002_correct_passes,
        test_verifier_task002_keywords_only_fails,
        test_verifier_task003_structural,
        test_verifier_task004_yaml_structure,
        test_verifier_task008_record_parser,
        # SkillsBench tests
        test_sb_001_flood_detection_correct,
        test_sb_002_hp_filter_correct,
        test_sb_003_dialogue_parser_correct,
        # Procedural generator tests
        test_procedural_auth_100_seeds,
        test_procedural_binary_100_seeds,
        test_procedural_deterministic,
        test_procedural_keyword_stuffing_fails,
        test_procedural_env_integration,
        test_procedural_uniqueness,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
