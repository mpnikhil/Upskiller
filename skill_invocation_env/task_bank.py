"""
Task and Skill Bank for the Skill Invocation Environment.

Contains 10 synthetic task/skill pairs covering fictional APIs, data formats,
coding standards, deployment procedures, and query languages. Each task is
genuinely impossible without reading the corresponding skill content.
"""

from typing import Callable


def _make_verifier(required: list[str], forbidden: list[str] | None = None) -> Callable[[str], bool]:
    """Create a keyword-based verifier."""
    def verify(answer: str) -> bool:
        lower = answer.lower()
        if not all(kw.lower() in lower for kw in required):
            return False
        if forbidden and any(kw.lower() in lower for kw in forbidden):
            return False
        return True
    return verify


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


TASK_BANK: list[dict] = [
    # --- Task 1: Zephyr-3 Auth (Easy) ---
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
        "distractor_skills": ["skill_002", "skill_003"],
        "verifier": _make_verifier(
            required=["hmac", "sha256", "x-zephyr-auth", "base64", "zph"],
        ),
    },
    # --- Task 2: NovaBin Parser (Easy) ---
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
        "distractor_skills": ["skill_005", "skill_017"],
        "verifier": _make_verifier(
            required=["struct", "novb", "0x4e4f5642", "big-endian"],
            forbidden=None,
        ),
    },
    # --- Task 3: HelixLang Error Handling (Easy) ---
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
        "distractor_skills": ["skill_007", "skill_008"],
        "verifier": _make_verifier(
            required=["hlx-", "try!", "with_context", "retry", "helix.log.error"],
        ),
    },
    # --- Task 4: ArcDeploy Canary Config (Easy) ---
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
        "distractor_skills": ["skill_010", "skill_011"],
        "verifier": _make_verifier(
            required=["shadow", "canary_1", "traffic_pct", "metrics_gate", "error_rate", "rollback", "auto: true"],
        ),
    },
    # --- Task 5: CrystalQL Temporal Query (Easy) ---
    {
        "id": "task_005",
        "difficulty": "easy",
        "description": (
            "Write a CrystalQL query that retrieves the average daily price and "
            "maximum weekly stock for each product between January 1 and March 1, "
            "2024. Use temporal aggregation functions and a tumbling window of 1 day."
        ),
        "relevant_skills": ["skill_012"],
        "distractor_skills": ["skill_013", "skill_005"],
        "verifier": _make_verifier(
            required=["temporal_avg", "temporal_max", "between", "window tumbling", "interval"],
        ),
    },
    # --- Task 6: VaultSync Rotation Script (Medium) ---
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
        "distractor_skills": ["skill_015", "skill_003"],
        "verifier": _make_verifier(
            required=[
                "vault-sync rotate prepare",
                "vault-sync rotate activate",
                "vault-sync rotate verify",
                "vault-sync rotate commit",
                "vault-sync rotate rollback",
                "grace-period",
                "probe-endpoint",
            ],
        ),
    },
    # --- Task 7: FluxStream Pipeline (Medium) ---
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
        "distractor_skills": ["skill_017", "skill_018"],
        "verifier": _make_verifier(
            required=[
                "pipeline",
                "source kafka",
                "filter",
                "window",
                "tumbling",
                "branch",
                "sink",
                "error_handler",
                "dead_letter",
            ],
        ),
    },
    # --- Task 8: NovaBin Record Parser (Medium) ---
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
        "distractor_skills": ["skill_005", "skill_013"],
        "verifier": _make_verifier(
            required=["struct", "0x01", "0x02", "0x03", "0x04", "uint16", "utf-8"],
        ),
    },
    # --- Task 9: CrystalQL + VaultSync Integration (Hard) ---
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
        "distractor_skills": ["skill_013", "skill_015", "skill_010"],
        "verifier": _make_verifier(
            required=[
                "temporal join",
                "as of",
                "vault-sync",
                "rotation_interval",
                "on_rotate",
                "secret",
                "grace_period",
            ],
        ),
    },
    # --- Task 10: Full ArcDeploy + FluxStream Monitoring (Hard) ---
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
        "distractor_skills": ["skill_010", "skill_011", "skill_018"],
        "verifier": _make_verifier(
            required=[
                "arc deploy",
                "canary",
                "shadow",
                "metrics_gate",
                "pipeline deploy_monitor",
                "source kafka",
                "window",
                "error_rate",
                "branch",
                "flux deploy",
            ],
        ),
    },
]
