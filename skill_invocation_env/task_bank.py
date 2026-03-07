"""
Task and Skill Bank for the Skill Invocation Environment.

Contains 10 synthetic task/skill pairs plus 3 tasks adapted from SkillsBench,
covering fictional APIs, data formats, coding standards, deployment procedures,
query languages, flood analysis, economics detrending, and dialogue parsing.

Verifiers use code execution and structural validation — not keyword matching.
"""

import binascii
import hashlib
import hmac
import base64
import re
import struct
from typing import Callable

try:
    import yaml as _yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Verifier helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    code = code.strip()
    # Match ```python ... ``` or ``` ... ```
    match = re.search(r'```(?:python)?\s*\n(.*?)```', code, re.DOTALL)
    if match:
        return match.group(1)
    # Also handle case where entire answer is fenced
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        return "\n".join(lines)
    return code


_SAFE_IMPORTS = "import hmac, hashlib, base64, struct, json, re, binascii, math"


def _exec_verifier(func_name: str, test_cases: list[dict]) -> Callable[[str], bool]:
    """
    Execute the agent's code, extract func_name, run test_cases.

    Each test_case: {"args": [...], "kwargs": {...}, "check": callable}
    where check(result) -> bool.
    """
    def verify(answer: str) -> bool:
        try:
            code = _strip_markdown_fences(answer)
            namespace: dict = {}
            exec(_SAFE_IMPORTS, namespace)
            exec(code, namespace)

            if func_name not in namespace:
                return False

            func = namespace[func_name]
            for tc in test_cases:
                result = func(*tc.get("args", []), **tc.get("kwargs", {}))
                if not tc["check"](result):
                    return False
            return True
        except Exception:
            return False
    return verify


def _structural_verifier(checks: list[Callable[[str], bool]]) -> Callable[[str], bool]:
    """All structural checks must pass."""
    def verify(answer: str) -> bool:
        return all(check(answer) for check in checks)
    return verify


def _multi_part_verifier(part_checks: list[Callable[[str], bool]]) -> Callable[[str], bool]:
    """Verify answer has multiple parts, each passing its check."""
    def verify(answer: str) -> bool:
        return all(check(answer) for check in part_checks)
    return verify


# ---------------------------------------------------------------------------
# Pre-computed test data for exec verifiers
# ---------------------------------------------------------------------------

# task_001: expected Zephyr-3 auth outputs
def _expected_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}


_ZEPHYR_TEST_CASES = [
    {
        "args": ["test_key_123", 1700000000],
        "check": lambda result: (
            isinstance(result, dict)
            and "X-Zephyr-Auth" in result
            and result == _expected_zephyr_auth("test_key_123", 1700000000)
        ),
    },
    {
        "args": ["another_key", 1700000001],
        "check": lambda result: (
            isinstance(result, dict)
            and "X-Zephyr-Auth" in result
            and result == _expected_zephyr_auth("another_key", 1700000001)
        ),
    },
    {
        "args": ["k", 0],
        "check": lambda result: (
            isinstance(result, dict)
            and "X-Zephyr-Auth" in result
            and result["X-Zephyr-Auth"].startswith("ZPH k:")
            and result["X-Zephyr-Auth"].endswith(":0")
        ),
    },
]

# task_002: NovaBin header test data
_NOVABIN_HEADER_RAW = (
    b'NOVB'                         # magic
    + struct.pack('>H', 0x0201)     # version 2.1
    + struct.pack('>I', 42)         # 42 records
    + struct.pack('>H', 0b101)      # compressed + checksummed (bits 0 and 2)
)
_NOVABIN_HEADER_CRC = binascii.crc32(_NOVABIN_HEADER_RAW) & 0xFFFFFFFF
_NOVABIN_HEADER = _NOVABIN_HEADER_RAW + struct.pack('>I', _NOVABIN_HEADER_CRC)

# Second test header: no flags, 1 record
_NOVABIN_HEADER2_RAW = (
    b'NOVB'
    + struct.pack('>H', 0x0201)
    + struct.pack('>I', 1)
    + struct.pack('>H', 0)  # no flags
)
_NOVABIN_HEADER2_CRC = binascii.crc32(_NOVABIN_HEADER2_RAW) & 0xFFFFFFFF
_NOVABIN_HEADER2 = _NOVABIN_HEADER2_RAW + struct.pack('>I', _NOVABIN_HEADER2_CRC)

_NOVABIN_HEADER_TEST_CASES = [
    {
        "args": [_NOVABIN_HEADER],
        "check": lambda r: (
            isinstance(r, dict)
            and r.get("version") in (0x0201, 513)
            and r.get("record_count") == 42
            and r.get("compressed") is True
            and r.get("encrypted") is False
            and r.get("checksummed") is True
        ),
    },
    {
        "args": [_NOVABIN_HEADER2],
        "check": lambda r: (
            isinstance(r, dict)
            and r.get("record_count") == 1
            and r.get("compressed") is False
            and r.get("encrypted") is False
            and r.get("checksummed") is False
        ),
    },
]

# task_008: NovaBin record test data
def _build_test_record() -> bytes:
    """Build a test record: 2 fields — int32 'age'=25, string 'name'='Alice'."""
    buf = bytearray()
    buf += struct.pack('>H', 2)  # field count = 2

    # Field 1: int32, name="age", value=25
    buf += bytes([0x01])  # type tag int32
    name1 = b"age"
    buf += struct.pack('>H', len(name1))
    buf += name1
    val1 = struct.pack('>i', 25)
    buf += struct.pack('>I', len(val1))
    buf += val1

    # Field 2: string, name="name", value="Alice"
    buf += bytes([0x03])  # type tag string
    name2 = b"name"
    buf += struct.pack('>H', len(name2))
    buf += name2
    val2 = b"Alice"
    buf += struct.pack('>I', len(val2))
    buf += val2

    return bytes(buf)


def _build_test_record_2() -> bytes:
    """Build a test record: 1 field — bool 'active'=True."""
    buf = bytearray()
    buf += struct.pack('>H', 1)

    buf += bytes([0x04])  # type tag bool
    name = b"active"
    buf += struct.pack('>H', len(name))
    buf += name
    val = bytes([1])  # True
    buf += struct.pack('>I', len(val))
    buf += val

    return bytes(buf)


_NOVABIN_RECORD = _build_test_record()
_NOVABIN_RECORD_2 = _build_test_record_2()

_NOVABIN_RECORD_TEST_CASES = [
    {
        "args": [_NOVABIN_RECORD, 0],
        "check": lambda r: (
            isinstance(r, tuple) and len(r) == 2
            and isinstance(r[0], dict)
            and r[0].get("age") == 25
            and r[0].get("name") == "Alice"
            and isinstance(r[1], int) and r[1] == len(_NOVABIN_RECORD)
        ),
    },
    {
        "args": [_NOVABIN_RECORD_2, 0],
        "check": lambda r: (
            isinstance(r, tuple) and len(r) == 2
            and isinstance(r[0], dict)
            and r[0].get("active") is True
            and isinstance(r[1], int) and r[1] == len(_NOVABIN_RECORD_2)
        ),
    },
]


# ---------------------------------------------------------------------------
# YAML-based verifier for ArcDeploy config
# ---------------------------------------------------------------------------

def _verify_arcdeploy_yaml(answer: str) -> bool:
    """Verify ArcDeploy canary config via YAML parsing + structural checks."""
    try:
        # Extract YAML block if wrapped in markdown fences
        yaml_text = answer
        if '```' in answer:
            blocks = re.findall(r'```(?:yaml)?\s*\n(.*?)```', answer, re.DOTALL)
            if blocks:
                yaml_text = blocks[0]

        if HAS_YAML:
            config = _yaml.safe_load(yaml_text)
            if not isinstance(config, dict):
                return False

            canary = config.get("canary", {})
            phases = canary.get("phases", [])

            if len(phases) < 5:
                return False

            # First phase must be shadow with 0% traffic
            if phases[0].get("name") != "shadow" or phases[0].get("traffic_pct") != 0:
                return False

            # Last phase must be 100%
            if phases[-1].get("traffic_pct") != 100:
                return False

            # Traffic must be monotonically increasing
            traffic = [p.get("traffic_pct", 0) for p in phases]
            if traffic != sorted(traffic):
                return False

            # Must have metrics gates on non-100% phases
            for p in phases:
                if p.get("traffic_pct", 0) < 100 and "metrics_gate" not in p:
                    return False

            # Must have rollback config with auto: true
            rollback = canary.get("rollback", {})
            if not rollback.get("auto"):
                return False

            return True
        else:
            # Fallback: structural regex checks if yaml not installed
            return _arcdeploy_structural_fallback(answer)
    except Exception:
        return False


def _arcdeploy_structural_fallback(answer: str) -> bool:
    """Structural fallback for ArcDeploy when yaml isn't available."""
    checks = [
        # Has shadow phase with traffic_pct: 0
        lambda a: bool(re.search(r'name:\s*shadow', a)) and bool(re.search(r'traffic_pct:\s*0\b', a)),
        # Has at least canary_1 phase
        lambda a: bool(re.search(r'name:\s*canary_1', a)),
        # Has full phase with traffic_pct: 100
        lambda a: bool(re.search(r'traffic_pct:\s*100', a)),
        # Has metrics gates with error_rate
        lambda a: bool(re.search(r'metrics_gate:.*error_rate', a)),
        # Has rollback with auto: true
        lambda a: bool(re.search(r'auto:\s*true', a, re.IGNORECASE)),
        # Has at least 5 phase entries
        lambda a: len(re.findall(r'-\s*name:', a)) >= 5,
    ]
    return all(c(answer) for c in checks)


# ---------------------------------------------------------------------------
# Skill Bank
# ---------------------------------------------------------------------------

SKILL_BANK: dict[str, dict] = {
    # --- Zephyr-3 API domain ---
    "skill_001": {
        "name": "Zephyr-3 API Authentication",
        "short_description": "Authentication protocol and header format for Zephyr-3 API endpoints.",
        "full_content": """## Zephyr-3 API Authentication Protocol

All Zephyr-3 API requests require HMAC-SHA256 authentication. The auth header must be constructed as follows:

1. Create a signing string: `{api_key}:{timestamp_ms}` where timestamp_ms is Unix epoch in milliseconds
2. Compute HMAC-SHA256 of the signing string using the api_key as the secret key
3. Base64-encode the HMAC digest
4. Set the header `X-Zephyr-Auth` to the format: `ZPH {api_key}:{base64_digest}:{timestamp_ms}`

Example:
```python
import hmac, hashlib, base64

def encode_zephyr_auth(api_key: str, timestamp: int) -> dict:
    signing_string = f"{api_key}:{timestamp}"
    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.sha256).digest()
    b64 = base64.b64encode(digest).decode()
    return {"X-Zephyr-Auth": f"ZPH {api_key}:{b64}:{timestamp}"}
```

Note: Timestamps must be in milliseconds. Requests with timestamps older than 30 seconds are rejected.
""",
    },
    "skill_002": {
        "name": "Zephyr-3 Rate Limiting",
        "short_description": "Rate limiting rules and backoff strategies for Zephyr-3 API.",
        "full_content": "Zephyr-3 uses a token bucket rate limiter with 100 requests per minute per API key. When rate limited (HTTP 429), use exponential backoff starting at 1 second with a max of 32 seconds. Include the X-Zephyr-Retry-After header value if present.",
    },
    "skill_003": {
        "name": "Zephyr-3 Webhook Configuration",
        "short_description": "Setting up and managing webhook endpoints in Zephyr-3.",
        "full_content": "Webhooks in Zephyr-3 are configured via the /v1/webhooks endpoint. Each webhook requires a target URL, event types array, and an optional secret for signature verification. Events are delivered with at-least-once semantics.",
    },

    # --- NovaBin binary format domain ---
    "skill_004": {
        "name": "NovaBin File Format Specification",
        "short_description": "Binary format spec for NovaBin data files used in NovaDB exports.",
        "full_content": """## NovaBin Binary Format v2.1

NovaBin is a compact binary format for serializing structured records.

### Header (16 bytes):
- Bytes 0-3: Magic number `0x4E4F5642` ("NOVB")
- Bytes 4-5: Version (uint16, big-endian) — current is 0x0201
- Bytes 6-9: Record count (uint32, big-endian)
- Bytes 10-11: Flags (uint16) — bit 0: compressed, bit 1: encrypted, bit 2: checksummed
- Bytes 12-15: Header checksum (CRC32 of bytes 0-11)

### Record format:
- 2 bytes: field count (uint16, big-endian)
- For each field:
  - 1 byte: type tag (0x01=int32, 0x02=float64, 0x03=string, 0x04=bool)
  - 2 bytes: name length (uint16)
  - N bytes: field name (UTF-8)
  - 4 bytes: value length (uint32)
  - M bytes: value data

### Parsing example:
```python
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
```
""",
    },
    "skill_005": {
        "name": "NovaBin Compression Options",
        "short_description": "Compression algorithms supported by NovaBin format.",
        "full_content": "NovaBin supports LZ4 (default) and Zstandard compression. When flag bit 0 is set, the record payload is compressed. The first byte of compressed data indicates the algorithm: 0x01 for LZ4, 0x02 for Zstandard.",
    },

    # --- HelixLang domain ---
    "skill_006": {
        "name": "HelixLang Error Handling Conventions",
        "short_description": "Error handling patterns and error code structure in HelixLang.",
        "full_content": """## HelixLang Error Handling Standard v3

HelixLang uses a Result monad pattern with structured error codes.

### Error Code Format:
`HLX-{CATEGORY}-{CODE}` where:
- CATEGORY is one of: IO, NET, AUTH, DATA, SYS
- CODE is a 4-digit number

### Result Type:
```
result<T> = Ok(T) | Err(HelixError)

class HelixError:
    code: str          # e.g., "HLX-NET-4012"
    message: str       # Human-readable description
    context: map       # Key-value pairs with debug info
    retryable: bool    # Whether operation can be retried
    chain: HelixError? # Wrapped inner error
```

### Mandatory Error Handling Rules:
1. All functions returning result<T> must be called with the `try!` operator or explicitly matched
2. Errors must be propagated with context: `try! operation().with_context("step", "fetching user")`
3. Error chains must preserve the original error: `Err(HelixError.wrap(original, "HLX-DATA-2001", "transform failed"))`
4. Retryable errors (retryable=true) should use the `retry_with_backoff` helper: max 3 attempts, exponential backoff 100ms/200ms/400ms
5. All error handlers must log via `helix.log.error(err)` before returning or re-raising

### Standard Error Codes:
- HLX-IO-1001: File not found
- HLX-IO-1002: Permission denied
- HLX-NET-4001: Connection timeout
- HLX-NET-4012: SSL certificate invalid
- HLX-AUTH-3001: Token expired
- HLX-AUTH-3002: Insufficient permissions
- HLX-DATA-2001: Deserialization failed
- HLX-DATA-2002: Schema validation error
- HLX-SYS-5001: Out of memory
- HLX-SYS-5002: Thread pool exhausted
""",
    },
    "skill_007": {
        "name": "HelixLang Module System",
        "short_description": "Module imports, visibility rules, and package management in HelixLang.",
        "full_content": "HelixLang uses a hierarchical module system. Modules are declared with `mod name { }` and imported with `use path::to::module`. Public items use the `pub` keyword. Circular dependencies are detected at compile time.",
    },
    "skill_008": {
        "name": "HelixLang Concurrency Primitives",
        "short_description": "Async/await patterns and thread safety in HelixLang.",
        "full_content": "HelixLang provides green threads via `spawn { }` blocks. Channels are typed: `chan<T>` for unbuffered, `chan<T, N>` for buffered. Mutexes use `lock(resource) { }` syntax. The runtime uses work-stealing scheduling.",
    },

    # --- ArcDeploy domain ---
    "skill_009": {
        "name": "ArcDeploy Canary Rollout Procedure",
        "short_description": "Step-by-step canary deployment process using ArcDeploy CLI.",
        "full_content": """## ArcDeploy Canary Rollout v4.2

ArcDeploy uses a 5-phase canary deployment with automatic rollback.

### Required Configuration (arc-deploy.yaml):
```yaml
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
```

### CLI Commands:
```bash
# Initialize deployment
arc deploy init --service my-svc --version v2.1.0 --config arc-deploy.yaml

# Start canary (enters shadow phase)
arc deploy start --wait-for-gate

# Advance to next phase (or auto-advance if --auto flag used)
arc deploy advance

# Check current status
arc deploy status --json

# Manual rollback
arc deploy rollback --reason "elevated error rate"

# Complete deployment (skip remaining phases)
arc deploy promote --force
```

### Rollback Triggers:
- Any metrics_gate failure triggers immediate rollback
- Manual `arc deploy rollback` at any phase
- Health check failures (3 consecutive) trigger rollback
- Rollback restores previous version and sets cooldown_min lockout
""",
    },
    "skill_010": {
        "name": "ArcDeploy Service Mesh Integration",
        "short_description": "Configuring ArcDeploy with service mesh for traffic routing.",
        "full_content": "ArcDeploy integrates with Istio and Linkerd for traffic splitting. Use `arc mesh configure --provider istio` to set up. Traffic rules are managed via VirtualService resources that ArcDeploy generates automatically during canary phases.",
    },
    "skill_011": {
        "name": "ArcDeploy Monitoring Dashboard",
        "short_description": "Setting up monitoring dashboards for ArcDeploy deployments.",
        "full_content": "ArcDeploy includes built-in Grafana dashboard templates. Run `arc monitor setup` to deploy. Default panels show: canary vs baseline error rates, latency percentiles, traffic split ratio, and rollback events.",
    },

    # --- CrystalQL domain ---
    "skill_012": {
        "name": "CrystalQL Temporal Query Syntax",
        "short_description": "Writing time-travel and temporal range queries in CrystalQL.",
        "full_content": """## CrystalQL Temporal Queries

CrystalQL extends SQL with temporal operators for querying historical data.

### Time-Travel Queries:
```sql
-- Query data as it existed at a specific point in time
SELECT * FROM users AS OF TIMESTAMP '2024-01-15T10:30:00Z';

-- Query data as of a relative time
SELECT * FROM orders AS OF INTERVAL '2 hours ago';

-- Query a range of historical states
SELECT * FROM inventory
BETWEEN TIMESTAMP '2024-01-01' AND TIMESTAMP '2024-01-31'
VERSIONED;
```

### Temporal Joins:
```sql
-- Join tables at consistent historical points
SELECT u.name, o.total
FROM users u
TEMPORAL JOIN orders o ON u.id = o.user_id
AS OF TIMESTAMP '2024-06-01T00:00:00Z';
```

### Temporal Aggregations:
```sql
-- Aggregate over time windows
SELECT
    product_id,
    TEMPORAL_AVG(price, INTERVAL '1 day') as avg_daily_price,
    TEMPORAL_MAX(stock, INTERVAL '1 week') as max_weekly_stock
FROM products
BETWEEN TIMESTAMP '2024-01-01' AND TIMESTAMP '2024-03-01'
GROUP BY product_id
WINDOW TUMBLING(INTERVAL '1 day');
```

### Important Syntax Rules:
1. `AS OF` requires ISO 8601 timestamps or INTERVAL expressions
2. `BETWEEN...AND...VERSIONED` returns all versions of each row
3. `TEMPORAL JOIN` ensures both sides use the same temporal point
4. `TEMPORAL_AVG`, `TEMPORAL_MAX`, `TEMPORAL_MIN` are temporal aggregate functions
5. `WINDOW TUMBLING(interval)` creates non-overlapping time windows
6. `WINDOW SLIDING(interval, step)` creates overlapping windows with step size
""",
    },
    "skill_013": {
        "name": "CrystalQL Index Optimization",
        "short_description": "Creating and tuning indexes for CrystalQL databases.",
        "full_content": "CrystalQL supports B-tree, Hash, and Temporal indexes. Create temporal indexes with `CREATE TEMPORAL INDEX idx ON table(column) RETENTION 90 DAYS`. Use `EXPLAIN TEMPORAL` to analyze temporal query plans.",
    },

    # --- VaultSync domain ---
    "skill_014": {
        "name": "VaultSync Secret Rotation Protocol",
        "short_description": "Automated secret rotation workflow in VaultSync.",
        "full_content": """## VaultSync Secret Rotation Protocol v2

VaultSync automates credential rotation with zero-downtime guarantees.

### Rotation Lifecycle:
1. **PREPARE**: Generate new credential, store as `pending` version
   ```
   vault-sync rotate prepare --secret db/prod/password --generator alphanumeric:32
   ```

2. **DUAL-WRITE**: Both old and new credentials are active (grace period)
   ```
   vault-sync rotate activate --secret db/prod/password --grace-period 300s
   ```
   During grace period, both versions are valid. Applications using VaultSync SDK
   automatically pick up the new version.

3. **VERIFY**: Confirm new credential works
   ```
   vault-sync rotate verify --secret db/prod/password --probe-endpoint https://db.internal/health
   ```

4. **COMMIT**: Revoke old credential
   ```
   vault-sync rotate commit --secret db/prod/password
   ```

5. **ROLLBACK** (if needed): Revert to previous credential
   ```
   vault-sync rotate rollback --secret db/prod/password --reason "verification failed"
   ```

### SDK Integration:
```python
from vaultsync import SecretClient

client = SecretClient(vault_url="https://vault.internal")

# Auto-refreshing secret reference
db_password = client.secret("db/prod/password")

# Always returns current active version
current = db_password.get()

# Register rotation callback
@db_password.on_rotate
def handle_rotation(new_value, old_value):
    reconnect_database(new_value)
```

### Rotation Policy (vault-sync.yaml):
```yaml
secrets:
  - path: db/prod/password
    rotation_interval: 7d
    generator: alphanumeric:32
    grace_period: 300s
    verify:
      type: http_probe
      endpoint: https://db.internal/health
    notify:
      - slack:#ops-alerts
```
""",
    },
    "skill_015": {
        "name": "VaultSync Access Policies",
        "short_description": "Configuring RBAC access policies for VaultSync secrets.",
        "full_content": "VaultSync uses path-based RBAC policies. Policies are written in HCL format: `path \"secret/data/*\" { capabilities = [\"read\"] }`. Policies are attached to service identities via `vault-sync policy attach`.",
    },

    # --- FluxStream domain ---
    "skill_016": {
        "name": "FluxStream Event Processing Pipeline",
        "short_description": "Building real-time event processing pipelines with FluxStream DSL.",
        "full_content": """## FluxStream Pipeline DSL v3.0

FluxStream uses a declarative DSL for building event processing pipelines.

### Pipeline Definition:
```flux
pipeline user_activity {
    source kafka("user-events", group="analytics") {
        format: json
        watermark: event_time, delay=10s
    }

    // Filter and transform
    |> filter(event.type IN ["click", "purchase", "signup"])
    |> map({
        user_id: event.user_id,
        action: event.type,
        timestamp: event.event_time,
        value: CASE event.type
            WHEN "purchase" THEN event.amount
            ELSE 1.0
        END
    })

    // Windowed aggregation
    |> window(tumbling=5m, on=timestamp) {
        group_by: [user_id]
        aggregate: {
            action_count: count(*),
            total_value: sum(value),
            unique_actions: count_distinct(action)
        }
    }

    // Route to multiple sinks
    |> branch {
        high_value: total_value > 100.0
            -> sink postgres("analytics_db", table="high_value_users")
        default:
            -> sink kafka("user-summaries")
    }

    error_handler {
        on deserialize_error: dead_letter("dlq-user-events")
        on processing_error: retry(max=3, backoff=exponential(100ms))
        on sink_error: circuit_breaker(threshold=5, reset=60s)
    }
}
```

### Key Operators:
- `|> filter(predicate)` — Filter events
- `|> map(expression)` — Transform events
- `|> window(type=duration)` — Windowed aggregation
- `|> branch { condition -> sink }` — Conditional routing
- `|> join(other_stream, on=key, within=duration)` — Stream joins
- `|> deduplicate(key, within=duration)` — Remove duplicates

### CLI:
```bash
flux deploy pipeline.flux --env production
flux status user_activity
flux metrics user_activity --window 1h
```
""",
    },
    "skill_017": {
        "name": "FluxStream Connector Configuration",
        "short_description": "Configuring source and sink connectors in FluxStream.",
        "full_content": "FluxStream supports Kafka, PostgreSQL, Redis, and S3 connectors. Configure via `flux connector add --type kafka --config broker=localhost:9092`. Each connector has health checks and auto-reconnection with configurable backoff.",
    },
    "skill_018": {
        "name": "FluxStream Schema Registry",
        "short_description": "Managing event schemas and compatibility in FluxStream.",
        "full_content": "FluxStream integrates with a schema registry for Avro/Protobuf/JSON Schema. Register schemas with `flux schema register --file event.avsc`. Compatibility modes: BACKWARD, FORWARD, FULL. Breaking changes are blocked by default.",
    },
}


# ---------------------------------------------------------------------------
# Task Bank
# ---------------------------------------------------------------------------

TASK_BANK: list[dict] = [
    # --- Task 1: Zephyr-3 Auth (Easy) --- exec verifier
    {
        "id": "task_001",
        "difficulty": "easy",
        "description": (
            "Write a Python function called `encode_zephyr_auth` that generates an "
            "authentication header for the Zephyr-3 API. The function should take "
            "`api_key` (str) and `timestamp` (int) as arguments and return a dict "
            "with the header."
        ),
        "relevant_skills": ["skill_001"],
        "distractor_skills": ["skill_002", "skill_003", "skill_007", "skill_013", "skill_017"],
        "verifier": _exec_verifier("encode_zephyr_auth", _ZEPHYR_TEST_CASES),
    },
    # --- Task 2: NovaBin Header Parser (Easy) --- exec verifier
    {
        "id": "task_002",
        "difficulty": "easy",
        "description": (
            "Write a Python function called `parse_novabin_header` that takes a "
            "`bytes` object (at least 16 bytes) and returns a dict with keys: "
            "'version', 'record_count', 'compressed', 'encrypted', 'checksummed', "
            "and 'checksum'. Parse according to the NovaBin file format specification."
        ),
        "relevant_skills": ["skill_004"],
        "distractor_skills": ["skill_005", "skill_017", "skill_003", "skill_008", "skill_013"],
        "verifier": _exec_verifier("parse_novabin_header", _NOVABIN_HEADER_TEST_CASES),
    },
    # --- Task 3: HelixLang Error Handling (Easy) --- structural verifier
    {
        "id": "task_003",
        "difficulty": "easy",
        "description": (
            "Write a code snippet in pseudocode or HelixLang style that demonstrates "
            "proper error handling for a function that fetches a user from a database, "
            "following HelixLang Error Handling Conventions. Include: the correct error "
            "code format, context propagation, retry logic for retryable errors, and "
            "error logging."
        ),
        "relevant_skills": ["skill_006"],
        "distractor_skills": ["skill_007", "skill_008", "skill_002", "skill_015", "skill_018"],
        "verifier": _structural_verifier([
            # Has a correctly-formatted error code (HLX-CATEGORY-NNNN)
            lambda a: bool(re.search(r'HLX-(IO|NET|AUTH|DATA|SYS)-\d{4}', a)),
            # Uses try! operator in a call context (not just the word)
            lambda a: bool(re.search(r'try!\s*\w+', a)),
            # Has context propagation with with_context call
            lambda a: bool(re.search(r'with_context\s*\(', a)),
            # Has retry logic referencing backoff
            lambda a: bool(re.search(r'retry.*backoff|retry_with_backoff', a, re.IGNORECASE)),
            # Has error logging via helix.log.error
            lambda a: bool(re.search(r'helix\.log\.error\s*\(', a)),
            # Has Result monad pattern (Ok or Err)
            lambda a: bool(re.search(r'result\s*<|Ok\s*\(|Err\s*\(', a)),
        ]),
    },
    # --- Task 4: ArcDeploy Canary Config (Easy) --- YAML verifier
    {
        "id": "task_004",
        "difficulty": "easy",
        "description": (
            "Write an `arc-deploy.yaml` configuration file for a canary deployment "
            "of service 'payment-svc' version 3.0.0. The deployment should have the "
            "standard 5-phase canary rollout with appropriate traffic percentages, "
            "durations, and metrics gates. Include automatic rollback configuration."
        ),
        "relevant_skills": ["skill_009"],
        "distractor_skills": ["skill_010", "skill_011", "skill_005", "skill_017", "skill_003"],
        "verifier": _verify_arcdeploy_yaml,
    },
    # --- Task 5: CrystalQL Temporal Query (Easy) --- structural verifier
    {
        "id": "task_005",
        "difficulty": "easy",
        "description": (
            "Write a CrystalQL query that retrieves the average daily price and "
            "maximum weekly stock for each product between January 1 and March 1, "
            "2024. Use temporal aggregation functions and a tumbling window of 1 day."
        ),
        "relevant_skills": ["skill_012"],
        "distractor_skills": ["skill_013", "skill_005", "skill_002", "skill_008", "skill_017"],
        "verifier": _structural_verifier([
            # Uses TEMPORAL_AVG function with parenthesized args
            lambda a: bool(re.search(r'TEMPORAL_AVG\s*\(', a, re.IGNORECASE)),
            # Uses TEMPORAL_MAX function with parenthesized args
            lambda a: bool(re.search(r'TEMPORAL_MAX\s*\(', a, re.IGNORECASE)),
            # Has BETWEEN clause with two timestamps (2024 dates)
            lambda a: bool(re.search(
                r'BETWEEN\s+TIMESTAMP\s+[\'"]2024-01-01.*AND\s+TIMESTAMP\s+[\'"]2024-03-01',
                a, re.IGNORECASE | re.DOTALL,
            )),
            # Has WINDOW TUMBLING with an interval
            lambda a: bool(re.search(r'WINDOW\s+TUMBLING\s*\(', a, re.IGNORECASE)),
            # Has GROUP BY product_id
            lambda a: bool(re.search(r'GROUP\s+BY\s+product_id', a, re.IGNORECASE)),
            # Uses INTERVAL for the window duration
            lambda a: bool(re.search(r"INTERVAL\s+['\"]?1\s*day", a, re.IGNORECASE)),
        ]),
    },
    # --- Task 6: VaultSync Rotation Script (Medium) --- structural verifier
    {
        "id": "task_006",
        "difficulty": "medium",
        "description": (
            "Write a shell script that performs a complete secret rotation for "
            "the database credential at path 'db/prod/password' using VaultSync CLI. "
            "The script should: prepare a new 32-char alphanumeric credential, activate "
            "with a 5-minute grace period, verify via health endpoint, and commit. "
            "Include error handling that triggers rollback if verification fails."
        ),
        "relevant_skills": ["skill_014"],
        "distractor_skills": ["skill_015", "skill_003", "skill_010", "skill_007", "skill_018"],
        "verifier": _structural_verifier([
            # Commands appear in correct lifecycle order: prepare < activate < verify < commit
            lambda a: (
                all(cmd in a for cmd in [
                    'vault-sync rotate prepare',
                    'vault-sync rotate activate',
                    'vault-sync rotate verify',
                    'vault-sync rotate commit',
                ])
                and a.index('rotate prepare') < a.index('rotate activate')
                < a.index('rotate verify') < a.index('rotate commit')
            ),
            # Has rollback command
            lambda a: 'vault-sync rotate rollback' in a,
            # Has grace-period flag
            lambda a: bool(re.search(r'--grace-period', a)),
            # Has probe-endpoint flag
            lambda a: bool(re.search(r'--probe-endpoint', a)),
            # Has conditional logic for error handling
            lambda a: bool(re.search(r'\bif\b|\bthen\b|\$\?|&&\s*vault-sync|\|\|', a)),
            # References the correct secret path
            lambda a: 'db/prod/password' in a,
        ]),
    },
    # --- Task 7: FluxStream Pipeline (Medium) --- structural verifier
    {
        "id": "task_007",
        "difficulty": "medium",
        "description": (
            "Write a FluxStream pipeline definition called 'order_analytics' that: "
            "1) reads from a Kafka topic 'order-events' in JSON format with a 10-second "
            "watermark delay, 2) filters for 'completed' and 'refunded' orders, "
            "3) performs a 5-minute tumbling window aggregation grouped by product_id "
            "computing count and sum of amount, 4) routes high-value results (total > 500) "
            "to PostgreSQL and everything else to a Kafka sink topic, and 5) includes "
            "proper error handling with dead letter queue and retry logic."
        ),
        "relevant_skills": ["skill_016"],
        "distractor_skills": ["skill_017", "skill_018", "skill_002", "skill_010", "skill_013"],
        "verifier": _structural_verifier([
            # Pipeline declaration with correct name
            lambda a: bool(re.search(r'pipeline\s+order_analytics\s*\{', a)),
            # Source is kafka with correct topic
            lambda a: bool(re.search(r'source\s+kafka\s*\(\s*["\']order-events', a)),
            # Has filter referencing completed and refunded
            lambda a: bool(re.search(r'filter\s*\(', a)) and 'completed' in a and 'refunded' in a,
            # Has tumbling window with 5m
            lambda a: bool(re.search(r'window\s*\(\s*tumbling\s*=\s*5m', a)),
            # Has group_by with product_id
            lambda a: bool(re.search(r'group_by.*product_id', a)),
            # Has branch with conditional routing to sink
            lambda a: bool(re.search(r'branch\s*\{', a)) and 'sink' in a,
            # Has error_handler block with dead_letter
            lambda a: bool(re.search(r'error_handler\s*\{', a)) and 'dead_letter' in a,
            # Has aggregation (count and sum)
            lambda a: bool(re.search(r'\bcount\b', a, re.IGNORECASE)) and bool(re.search(r'\bsum\b', a, re.IGNORECASE)),
        ]),
    },
    # --- Task 8: NovaBin Record Parser (Medium) --- exec verifier
    {
        "id": "task_008",
        "difficulty": "medium",
        "description": (
            "Write a Python function called `parse_novabin_record` that parses a single "
            "NovaBin record from a bytes buffer starting at a given offset. The function "
            "should take `data: bytes` and `offset: int` and return a tuple of "
            "(dict_of_fields, new_offset). Handle all four field types: int32, float64, "
            "string, and bool according to the NovaBin format specification."
        ),
        "relevant_skills": ["skill_004"],
        "distractor_skills": ["skill_005", "skill_013", "skill_001", "skill_015", "skill_018"],
        "verifier": _exec_verifier("parse_novabin_record", _NOVABIN_RECORD_TEST_CASES),
    },
    # --- Task 9: CrystalQL + VaultSync Integration (Hard) --- multi-part verifier
    {
        "id": "task_009",
        "difficulty": "hard",
        "description": (
            "Design a system that combines CrystalQL temporal queries with VaultSync "
            "secret rotation. Write: 1) A CrystalQL temporal join query that retrieves "
            "user orders with prices as they were at order time, using the TEMPORAL JOIN "
            "and AS OF syntax. 2) A VaultSync rotation policy YAML for the database "
            "credential used by this query service, with 7-day rotation, health probe "
            "verification, and Slack notifications. 3) A Python integration snippet using "
            "VaultSync SDK that auto-refreshes the database connection on secret rotation."
        ),
        "relevant_skills": ["skill_012", "skill_014"],
        "distractor_skills": ["skill_013", "skill_015", "skill_010", "skill_002", "skill_017"],
        "verifier": _multi_part_verifier([
            # Part 1: CrystalQL temporal join query
            lambda a: bool(re.search(r'TEMPORAL\s+JOIN', a, re.IGNORECASE)),
            lambda a: bool(re.search(r'AS\s+OF\s+(TIMESTAMP|INTERVAL)', a, re.IGNORECASE)),
            lambda a: bool(re.search(r'\bON\b.*\bid\b', a, re.IGNORECASE)),
            # Part 2: VaultSync rotation policy YAML
            lambda a: bool(re.search(r'rotation_interval:\s*7d', a)),
            lambda a: bool(re.search(r'grace_period:', a)),
            lambda a: bool(re.search(r'http_probe|probe.*endpoint', a, re.IGNORECASE)),
            lambda a: bool(re.search(r'slack:', a, re.IGNORECASE)),
            # Part 3: Python SDK with on_rotate callback
            lambda a: bool(re.search(r'SecretClient|secret\s*\(', a)),
            lambda a: bool(re.search(r'on_rotate|\.on_rotate', a)),
        ]),
    },
    # --- Task 10: Full ArcDeploy + FluxStream Monitoring (Hard) --- multi-part verifier
    {
        "id": "task_010",
        "difficulty": "hard",
        "description": (
            "Write a complete deployment and monitoring setup that: 1) Defines an "
            "ArcDeploy canary rollout configuration for a streaming service with 5 phases "
            "including shadow testing, with metrics gates checking both latency and error "
            "rates. 2) Writes a FluxStream pipeline called 'deploy_monitor' that consumes "
            "deployment metric events, computes real-time error rates and latency "
            "percentiles in 1-minute tumbling windows, and routes alerts (error_rate > 0.01) "
            "to a dedicated alert sink. Include CLI commands for both tools to start "
            "the deployment and deploy the monitoring pipeline."
        ),
        "relevant_skills": ["skill_009", "skill_016"],
        "distractor_skills": ["skill_010", "skill_011", "skill_018", "skill_003", "skill_013"],
        "verifier": _multi_part_verifier([
            # Part 1: ArcDeploy config — shadow phase + metrics gates
            lambda a: bool(re.search(r'name:\s*shadow', a)),
            lambda a: len(re.findall(r'-\s*name:', a)) >= 5,
            lambda a: bool(re.search(r'metrics_gate:.*latency', a, re.IGNORECASE)),
            lambda a: bool(re.search(r'metrics_gate:.*error_rate', a, re.IGNORECASE)),
            lambda a: bool(re.search(r'traffic_pct:\s*100', a)),
            # Part 2: FluxStream pipeline with correct name
            lambda a: bool(re.search(r'pipeline\s+deploy_monitor\s*\{', a)),
            lambda a: bool(re.search(r'source\s+kafka\s*\(', a)),
            lambda a: bool(re.search(r'window\s*\(\s*tumbling\s*=\s*1m', a)),
            lambda a: bool(re.search(r'branch\s*\{', a)) and bool(re.search(r'error_rate', a)),
            # Part 3: CLI commands for both tools
            lambda a: bool(re.search(r'arc\s+deploy\s+(init|start)', a)),
            lambda a: bool(re.search(r'flux\s+deploy', a)),
        ]),
    },
]


# ---------------------------------------------------------------------------
# Tasks adapted from SkillsBench (https://github.com/benchflow-ai/skillsbench)
# Apache 2.0 License. Skills content used with attribution.
# Adapted for text-in/text-out OpenEnv Gymnasium-style environment.
# ---------------------------------------------------------------------------

# ---- SkillsBench Skills ----

SKILL_BANK["skill_sb_001"] = {
    "name": "Flood Detection Guide",
    "short_description": "Detect flood events by comparing water levels to thresholds, aggregating data, and classifying severity.",
    "full_content": """## Flood Detection Guide

### Overview

Flood detection involves comparing observed water levels against established flood stage thresholds. This guide covers how to process water level data and identify flood events.

### Flood Stage Definition

According to the National Weather Service, flood stage is the water level at which overflow of the natural banks begins to cause damage. A flood event occurs when:
```
water_level >= flood_stage_threshold
```

### Aggregating Instantaneous Data to Daily

USGS instantaneous data is recorded at ~15-minute intervals. For flood detection, aggregate to daily maximum:
```python
# df is DataFrame from nwis.get_iv() with datetime index
# gage_col is the column name containing water levels
daily_max = df[gage_col].resample('D').max()
```

### Why Daily Maximum?

| Aggregation | Use Case |
|-------------|----------|
| `max()` | Flood detection - captures peak water level |
| `mean()` | Long-term trends - may miss short flood peaks |
| `min()` | Low flow analysis |

### Detecting Flood Days

Compare daily maximum water level against flood threshold:
```python
flood_threshold = <threshold_from_nws>  # feet

# Count days with flooding
flood_days = (daily_max >= flood_threshold).sum()

# Get specific dates with flooding
flood_dates = daily_max[daily_max >= flood_threshold].index.tolist()
```

### Processing Multiple Stations
```python
flood_results = []

for site_id, site_data in all_data.items():
    daily_max = site_data['water_levels'].resample('D').max()
    threshold = thresholds[site_id]['flood']

    days_above = int((daily_max >= threshold).sum())

    if days_above > 0:
        flood_results.append({
            'station_id': site_id,
            'flood_days': days_above
        })

# Sort by flood days descending
flood_results.sort(key=lambda x: x['flood_days'], reverse=True)
```

### Flood Severity Classification

If multiple threshold levels are available:
```python
def classify_flood(water_level, thresholds):
    if water_level >= thresholds['major']:
        return 'major'
    elif water_level >= thresholds['moderate']:
        return 'moderate'
    elif water_level >= thresholds['flood']:
        return 'minor'
    elif water_level >= thresholds['action']:
        return 'action'
    else:
        return 'normal'
```

### Best Practices

- Use daily maximum for flood detection to capture peaks
- Ensure water level and threshold use same units (typically feet)
- Only report stations with at least 1 flood day
- Sort results by flood severity or duration for prioritization
""",
}

SKILL_BANK["skill_sb_002"] = {
    "name": "USGS Data Download Guide",
    "short_description": "Download water level data from USGS using the dataretrieval Python package.",
    "full_content": """## USGS Data Download Guide

The `dataretrieval` package allows downloading water level data from USGS.

```python
from dataretrieval import nwis

# Get instantaneous values (15-min intervals)
df, meta = nwis.get_iv(
    sites='<station_id>',
    start='<start_date>',
    end='<end_date>',
    parameterCd='00065'
)
```

Parameter codes: 00065 = Gage height (feet), 00060 = Discharge (cfs).
Station IDs are 8-digit strings with leading zeros (e.g., '04119000').
""",
}

SKILL_BANK["skill_sb_003"] = {
    "name": "NWS Flood Threshold Lookup",
    "short_description": "Look up flood stage thresholds from National Weather Service for USGS stations.",
    "full_content": """## NWS Flood Thresholds

The NWS maintains flood stage thresholds. Download from:
```
https://water.noaa.gov/resources/downloads/reports/nwps_all_gauges_report.csv
```

Key columns: `usgs id`, `flood stage`, `moderate flood stage`, `major flood stage`.
Use the `flood stage` column as threshold. Filter out -9999 values.
""",
}

SKILL_BANK["skill_sb_004"] = {
    "name": "Time Series Detrending for Macroeconomic Analysis",
    "short_description": "HP filter detrending techniques for business cycle analysis of economic time series.",
    "full_content": """## Time Series Detrending for Macroeconomic Analysis

This skill provides guidance on decomposing economic time series into trend and cyclical components, a fundamental technique in business cycle analysis.

### The Hodrick-Prescott (HP) Filter

The HP filter decomposes a time series into a trend component and a cyclical component. Given a time series y_t, it finds the trend tau_t that minimizes:

  sum((y_t - tau_t)^2) + lambda * sum([(tau_{t+1} - tau_t) - (tau_t - tau_{t-1})]^2)

### Choosing Lambda

**Critical**: The choice of lambda depends on data frequency:

| Data Frequency | Recommended lambda | Rationale |
|----------------|-------------------|-----------|
| Annual         | **100**           | Standard for yearly data |
| Quarterly      | **1600**          | Hodrick-Prescott (1997) recommendation |
| Monthly        | **14400**         | Ravn-Uhlig (2002) adjustment |

**Common mistake**: Using lambda=1600 (quarterly default) for annual data.

### Python Implementation

```python
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np

# Apply HP filter
# Returns: (cyclical_component, trend_component)
cycle, trend = hpfilter(data, lamb=100)  # For annual data
```

**Important**: The function parameter is `lamb` (not `lambda`, which is a Python keyword).

### Log Transformation

For most macroeconomic aggregates (GDP, consumption, investment), apply the natural logarithm **before** filtering:

```python
import numpy as np

log_series = np.log(real_series)
cycle, trend = hpfilter(log_series, lamb=100)

# cycle now represents percentage deviations from trend
```

### Complete Workflow: Business Cycle Correlation

```python
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

# Log transformation
ln_consumption = np.log(real_consumption)
ln_investment = np.log(real_investment)

# HP filter with lambda=100 for annual data
cycle_c, trend_c = hpfilter(ln_consumption, lamb=100)
cycle_i, trend_i = hpfilter(ln_investment, lamb=100)

# Compute correlation of cyclical components
correlation = np.corrcoef(cycle_c, cycle_i)[0, 1]
print(f"Business cycle correlation: {correlation:.5f}")
```

### Dependencies
```bash
pip install statsmodels pandas numpy
```

The HP filter is in `statsmodels.tsa.filters.hp_filter`.
""",
}

SKILL_BANK["skill_sb_005"] = {
    "name": "Pandas DataFrame Operations",
    "short_description": "Common pandas operations for data manipulation, merging, and aggregation.",
    "full_content": """## Pandas DataFrame Operations

Common pandas patterns: `pd.read_csv()`, `df.merge()`, `df.groupby().agg()`, `df.pivot_table()`. Use `pd.to_numeric(errors='coerce')` for type conversion. Handle missing data with `df.fillna()` or `df.dropna()`.
""",
}

SKILL_BANK["skill_sb_006"] = {
    "name": "Matplotlib Time Series Visualization",
    "short_description": "Plotting time series data and trends with matplotlib.",
    "full_content": """## Matplotlib Time Series Visualization

Plot time series with `plt.plot(dates, values)`. Add trend lines, annotations, and formatting. Use `fig, axes = plt.subplots()` for multi-panel plots. Save with `plt.savefig('output.png', dpi=150)`.
""",
}

SKILL_BANK["skill_sb_007"] = {
    "name": "Dialogue Graph Library",
    "short_description": "Build, validate, and serialize dialogue graphs for branching narratives using Graph/Node/Edge classes.",
    "full_content": """## Dialogue Graph Skill

This skill provides a `dialogue_graph` module to build valid dialogue trees/graphs.

### When to use
- **Script Parsers**: Converting text to structured data
- **Dialogue Editors**: Building tools to edit conversation flow
- **Game Logic**: Traversing a dialogue tree

### How to use

```python
from dialogue_graph import Graph, Node, Edge
```

#### The Graph Class
```python
graph = Graph()
```

#### Adding Nodes
```python
# Regular dialogue line
graph.add_node(Node(id="Start", speaker="Guard", text="Halt!", type="line"))

# Choice hub
graph.add_node(Node(id="Choices", type="choice"))
```

#### Adding Edges
```python
# Simple transition
graph.add_edge(Edge(source="Start", target="Choices"))

# Choice transition (with text)
graph.add_edge(Edge(source="Choices", target="End", text="1. Run away"))
```

#### Export
```python
data = graph.to_dict()
# returns {"nodes": [...], "edges": [...]}
```

#### Validation
```python
errors = graph.validate()
# Returns list of strings, e.g., ["Edge 'Start'->'Unk' points to missing node 'Unk'"]
```

#### Loading from JSON
```python
graph = Graph.from_dict({'nodes': [...], 'edges': [...]})
```

### Node format
Each node is a dict with: `id` (str), `text` (str), `speaker` (str), `type` ("line" or "choice")

### Edge format
Each edge is a dict with: `from` (str), `to` (str), `text` (str)

### Parsing Script Format
The standard script format:
```
[SceneName]
Speaker: Dialogue text. -> NextScene

[ChoiceScene]
1. Choice text -> TargetScene
2. Another choice -> OtherTarget
```

Rules:
- Lines starting with `[Name]` define scene nodes
- If a scene has `Speaker: Text -> Target`, it is type="line" with that speaker
- If a scene has numbered options (`1. Text -> Target`), it is type="choice"
- The arrow `->` defines an edge from the current scene to the target
- Choice edges include the choice text (e.g., "1. Choice text")
- For line nodes, the edge text is empty or the dialogue text
""",
}

SKILL_BANK["skill_sb_008"] = {
    "name": "Graphviz DOT Format",
    "short_description": "Generate graph visualizations using Graphviz DOT language.",
    "full_content": """## Graphviz DOT Format

Create directed graphs: `digraph G { A -> B; }`. Set node shapes with `[shape=diamond]`. Colors with `[color=blue]`. Use `rankdir=LR` for left-to-right layout. Render with `dot -Tpng graph.dot -o graph.png`.
""",
}

SKILL_BANK["skill_sb_009"] = {
    "name": "JSON Schema Validation",
    "short_description": "Validate JSON data structures against schemas using jsonschema library.",
    "full_content": """## JSON Schema Validation

Use `jsonschema.validate(data, schema)` to check JSON structure. Define schemas with `type`, `properties`, `required`, `items`. Install: `pip install jsonschema`. Catches `ValidationError` on failure.
""",
}

# ---- SkillsBench-adapted Test Data ----

_SB_FLOOD_TEST_DATA = {
    "daily_max_levels": {
        "04124000": [15.2, 16.1, 15.8, 14.9, 15.5, 16.0, 15.3],
        "04101800": [22.1, 23.5, 22.8, 24.1, 23.0, 21.9, 22.5],
        "04097500": [10.5, 11.2, 10.8, 11.1, 10.3, 9.8, 9.5],
        "04166500": [8.2, 8.5, 7.9, 7.8, 8.0, 7.5, 7.3],
        "04999999": [5.0, 4.8, 5.1, 4.9, 5.0, 4.7, 4.6],
    },
    "flood_thresholds": {
        "04124000": 15.0,
        "04101800": 22.0,
        "04097500": 10.5,
        "04166500": 8.1,
        "04999999": 6.0,
    },
}
_SB_FLOOD_EXPECTED = {"04124000": 6, "04101800": 6, "04097500": 4, "04166500": 2}

# Pre-computed HP filter test data (deterministic, seed=42)
# Generated with: np.random.seed(42); n=50; trend=linspace(10,12,n);
# cycle=0.05*sin(linspace(0,4*pi,n)); noise_a/b=0.01*randn(n);
# series = exp(trend + [1|0.8]*cycle + noise)
_SB_HP_SERIES_A = [
    22136.146542207392, 23204.845943423705, 24652.79143953327,
    26172.47083057089, 27002.39677592748, 28273.313580880793,
    30051.396100901988, 31012.171288310117, 31766.445340494465,
    33182.78309985593, 33887.64705958506, 34894.26280732681,
    36149.52370863675, 36385.350397792274, 37514.59011964166,
    39113.01687213975, 40212.14327686968, 42196.670687212085,
    43297.26210449467, 44892.592298244555, 48303.64908257102,
    49801.15821216017, 52508.42643350753, 54489.575008037566,
    57974.245708500996, 61568.844301395045, 64117.839034819415,
    68576.37200943612, 71392.08801561705, 75092.589894759,
    78263.12445218801, 83583.20317625241, 85213.13757625608,
    87323.09714571184, 91894.08758161688, 92786.56912352098,
    96861.20170394408, 97480.13273428561, 100909.88166437816,
    105506.58025009745, 109428.19143887387, 112516.26919621794,
    116352.87005507768, 120827.19859243499, 124641.85636295384,
    131502.23328916836, 138435.59870506602, 147916.42415706386,
    154807.41309002193, 159910.50571503676,
]
_SB_HP_SERIES_B = [
    22097.96583697866, 23089.002181075368, 24209.299982962977,
    25755.231345447573, 27113.336616067252, 28331.828906180173,
    29041.61942185999, 30382.41537568266, 31739.18190358319,
    33080.971269261834, 33697.971409879494, 34882.03889214109,
    35642.5482028925, 36717.31912567404, 38645.99707499506,
    40129.947797790956, 40926.54825685394, 42889.14931129554,
    44289.153022361395, 45687.121649986715, 48212.03616086788,
    51085.192717671445, 52768.858896074795, 56354.155802272886,
    56856.12876304818, 61928.90266241154, 64673.42412352394,
    67709.05244322174, 71328.26917352251, 73157.81859575034,
    77790.94757943293, 81527.23479593106, 85685.60547068984,
    87078.29744266604, 89823.2654818476, 93051.57680422119,
    97361.69347175729, 99800.53110013365, 102039.82705754046,
    106419.91537764092, 109536.45736907546, 114426.94522822299,
    116805.80714641871, 122008.49722830504, 127217.1275907411,
    131648.90322204158, 140460.9679752306, 147465.45111253537,
    154676.11432606063, 162373.437098354,
]
_SB_HP_EXPECTED_CORR = 0.62487

_SB_DIALOGUE_TEST_SCRIPT = """[GateScene]
Guard: Halt! State your name. -> NameChoice

[NameChoice]
1. I am Sir Aldric. -> KnightPath
2. Just a traveler. -> TravelerPath

[KnightPath]
Captain: Welcome, knight! -> End

[TravelerPath]
Guard: Move along then. -> End

[End]
Narrator: The story continues...
"""

# ---- SkillsBench-adapted Tasks ----

TASK_BANK.extend([
    # --- task_sb_001: Flood Risk Analysis (Easy) --- exec verifier
    {
        "id": "task_sb_001",
        "difficulty": "easy",
        "source": "skillsbench",
        "description": (
            "Write a Python function called `detect_flood_days` that takes two arguments: "
            "`daily_max_levels` (dict mapping station_id to list of daily maximum water "
            "levels in feet) and `flood_thresholds` (dict mapping station_id to flood stage "
            "threshold in feet). Return a dict mapping station_id to the number of days "
            "where water level >= threshold. Only include stations with at least 1 flood day."
        ),
        "relevant_skills": ["skill_sb_001"],
        "distractor_skills": ["skill_sb_002", "skill_sb_003", "skill_sb_005", "skill_sb_008", "skill_sb_009"],
        "verifier": _exec_verifier("detect_flood_days", [
            {
                "args": [
                    _SB_FLOOD_TEST_DATA["daily_max_levels"],
                    _SB_FLOOD_TEST_DATA["flood_thresholds"],
                ],
                "check": lambda r: (
                    isinstance(r, dict)
                    and r == _SB_FLOOD_EXPECTED
                ),
            },
            # Edge case: empty input
            {
                "args": [{}, {}],
                "check": lambda r: isinstance(r, dict) and len(r) == 0,
            },
            # Edge case: station with exactly threshold value (should count)
            {
                "args": [
                    {"ST1": [5.0, 4.9, 5.0]},
                    {"ST1": 5.0},
                ],
                "check": lambda r: isinstance(r, dict) and r.get("ST1") == 2,
            },
        ]),
    },
    # --- task_sb_002: Economics HP Filter Correlation (Medium) --- exec verifier
    {
        "id": "task_sb_002",
        "difficulty": "medium",
        "source": "skillsbench",
        "description": (
            "Write a Python function called `hp_filter_correlation` that takes two arguments: "
            "`series_a` (list of floats) and `series_b` (list of floats) representing two "
            "economic time series. The function should: 1) Take the natural log of each "
            "series, 2) Apply the Hodrick-Prescott filter with lambda=100 to extract "
            "cyclical components, 3) Compute and return the Pearson correlation coefficient "
            "between the two cyclical components as a float rounded to 5 decimal places. "
            "You may use numpy and scipy (statsmodels.api.tsa.filters.hpfilter is available)."
        ),
        "relevant_skills": ["skill_sb_004"],
        "distractor_skills": ["skill_sb_005", "skill_sb_006", "skill_sb_002", "skill_sb_009", "skill_sb_003"],
        "verifier": _exec_verifier("hp_filter_correlation", [
            {
                "args": [_SB_HP_SERIES_A, _SB_HP_SERIES_B],
                "check": lambda r: (
                    isinstance(r, (int, float))
                    and abs(float(r) - _SB_HP_EXPECTED_CORR) < 0.01
                ),
            },
        ]),
    },
    # --- task_sb_003: Dialogue Parser (Medium) --- exec verifier
    {
        "id": "task_sb_003",
        "difficulty": "medium",
        "source": "skillsbench",
        "description": (
            "Write a Python function called `parse_dialogue` that takes a dialogue script "
            "string and returns a dict with 'nodes' (list of dicts with 'id', 'text', "
            "'speaker', 'type') and 'edges' (list of dicts with 'from', 'to', 'text'). "
            "Parse this format:\n\n"
            "```\n"
            "[SceneName]\n"
            "Speaker: Dialogue text. -> NextScene\n"
            "\n"
            "[ChoiceScene]\n"
            "1. Choice text -> TargetScene\n"
            "2. Another choice -> OtherTarget\n"
            "```\n\n"
            "Lines with `[Name]` define scene nodes (type='line' if has speaker dialogue, "
            "type='choice' if has numbered options). Arrow `->` defines edges. For line "
            "nodes, text is the dialogue text and speaker is the speaker name. For choice "
            "nodes, text and speaker are empty strings. Edge text is the choice text for "
            "choices or empty string for line transitions."
        ),
        "relevant_skills": ["skill_sb_007"],
        "distractor_skills": ["skill_sb_008", "skill_sb_009", "skill_sb_002", "skill_sb_005", "skill_sb_006"],
        "verifier": _exec_verifier("parse_dialogue", [
            {
                "args": [_SB_DIALOGUE_TEST_SCRIPT],
                "check": lambda r: (
                    isinstance(r, dict)
                    and "nodes" in r and "edges" in r
                    and isinstance(r["nodes"], list) and isinstance(r["edges"], list)
                    # Must have 5 nodes: GateScene, NameChoice, KnightPath, TravelerPath, End
                    and len(r["nodes"]) == 5
                    # Check specific nodes exist with correct data
                    and any(
                        n.get("id") == "GateScene"
                        and n.get("speaker") == "Guard"
                        and "Halt" in n.get("text", "")
                        and n.get("type") == "line"
                        for n in r["nodes"]
                    )
                    and any(
                        n.get("id") == "NameChoice"
                        and n.get("type") == "choice"
                        for n in r["nodes"]
                    )
                    and any(
                        n.get("id") == "End"
                        for n in r["nodes"]
                    )
                    # Must have 5 edges
                    and len(r["edges"]) == 5
                    # Check specific edges
                    and any(
                        e.get("from") == "GateScene" and e.get("to") == "NameChoice"
                        for e in r["edges"]
                    )
                    and any(
                        e.get("from") == "NameChoice" and e.get("to") == "KnightPath"
                        for e in r["edges"]
                    )
                    and any(
                        e.get("from") == "NameChoice" and e.get("to") == "TravelerPath"
                        for e in r["edges"]
                    )
                    and any(
                        e.get("from") == "KnightPath" and e.get("to") == "End"
                        for e in r["edges"]
                    )
                    and any(
                        e.get("from") == "TravelerPath" and e.get("to") == "End"
                        for e in r["edges"]
                    )
                ),
            },
        ]),
    },
])
