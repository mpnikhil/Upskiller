"""
Procedural Task Generator for the Skill Invocation Environment.

Generates unlimited unique tasks at runtime using seeded randomization,
preventing LLM memorization of fixed task content. Each template produces
a task dict compatible with TASK_BANK format, plus any generated skills.

Templates:
1. Auth Protocol — randomized API name, hash algo, signing format, header format
2. Binary Format — randomized format name, magic bytes, endianness, header fields
"""

import hashlib
import hmac
import base64
import random
import struct
import binascii
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    import re
    code = code.strip()
    match = re.search(r'```(?:python)?\s*\n(.*?)```', code, re.DOTALL)
    if match:
        return match.group(1)
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        return "\n".join(lines)
    return code


_SAFE_IMPORTS = "import hmac, hashlib, base64, struct, json, re, binascii, math"


def _exec_verifier(func_name: str, test_cases: list[dict]) -> Callable[[str], bool]:
    """Execute agent code, extract func_name, run test_cases."""
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


# ---------------------------------------------------------------------------
# Distractor skill pool for procedural tasks
# ---------------------------------------------------------------------------

_DISTRACTOR_SKILLS = [
    {
        "id": "skill_proc_dist_001",
        "name": "Rate Limiting Strategies",
        "short_description": "Common rate limiting algorithms: token bucket, sliding window, leaky bucket.",
        "full_content": (
            "# Rate Limiting Strategies\n\n"
            "## Token Bucket\nMaintain a bucket of tokens that refills at rate R. "
            "Each request consumes one token. Reject when empty.\n\n"
            "## Sliding Window\nTrack request timestamps in a window of W seconds. "
            "Reject when count exceeds threshold T.\n\n"
            "## Leaky Bucket\nQueue requests and process at constant rate. "
            "Reject when queue is full."
        ),
    },
    {
        "id": "skill_proc_dist_002",
        "name": "Webhook Configuration",
        "short_description": "How to set up and manage webhook endpoints for event notifications.",
        "full_content": (
            "# Webhook Configuration\n\n"
            "Register an endpoint URL via POST /webhooks with event types. "
            "Verify signatures using HMAC-SHA256 of the payload with your webhook secret. "
            "Respond with 200 within 5s or the webhook will be retried 3 times with exponential backoff."
        ),
    },
    {
        "id": "skill_proc_dist_003",
        "name": "Data Compression Algorithms",
        "short_description": "Overview of LZ4, Zstd, and DEFLATE compression for binary data.",
        "full_content": (
            "# Data Compression\n\n"
            "## LZ4\nFast compression, moderate ratio. Use for real-time streaming.\n\n"
            "## Zstd\nHigh ratio with configurable levels (1-22). Good for storage.\n\n"
            "## DEFLATE\nWidely compatible (gzip/zip). Use for interchange formats."
        ),
    },
    {
        "id": "skill_proc_dist_004",
        "name": "Service Mesh Routing",
        "short_description": "Traffic splitting, circuit breaking, and retry policies for microservices.",
        "full_content": (
            "# Service Mesh Routing\n\n"
            "Configure traffic splitting with weight-based routing. "
            "Set circuit breakers with failure thresholds and recovery windows. "
            "Retry policies: max 3 retries with exponential backoff, only on 5xx errors."
        ),
    },
]


# ---------------------------------------------------------------------------
# Template 1: Auth Protocol
# ---------------------------------------------------------------------------

_API_NAMES = [
    "Zephyr", "Nebula", "Quantum", "Prism", "Helix",
    "Vortex", "Apex", "Nimbus", "Zenith", "Orion",
    "Pulse", "Flux", "Stratos", "Cipher", "Forge",
    "Atlas", "Beacon", "Crest", "Drift", "Echo",
]

_HASH_ALGOS = [
    ("sha256", "SHA-256", hashlib.sha256),
    ("sha384", "SHA-384", hashlib.sha384),
    ("sha512", "SHA-512", hashlib.sha512),
    ("md5", "MD5", hashlib.md5),
]

_SIGNING_FORMATS = [
    # (format_name, format_template, builder)
    (
        "key:timestamp",
        "{api_key}:{timestamp}",
        lambda api_key, timestamp, **_: f"{api_key}:{timestamp}",
    ),
    (
        "timestamp.key",
        "{timestamp}.{api_key}",
        lambda api_key, timestamp, **_: f"{timestamp}.{api_key}",
    ),
    (
        "key|timestamp|method",
        "{api_key}|{timestamp}|{method}",
        lambda api_key, timestamp, method="GET", **_: f"{api_key}|{timestamp}|{method}",
    ),
    (
        "method:key:timestamp",
        "{method}:{api_key}:{timestamp}",
        lambda api_key, timestamp, method="GET", **_: f"{method}:{api_key}:{timestamp}",
    ),
]

_HEADER_FORMATS = [
    # (header_name, prefix, format_builder)
    (
        "X-{api}-Auth",
        "{prefix}",
        lambda prefix, api_key, sig, timestamp, **_: f"{prefix} {api_key}:{sig}:{timestamp}",
    ),
    (
        "Authorization",
        "Bearer",
        lambda prefix, api_key, sig, timestamp, **_: f"Bearer {sig}",
    ),
    (
        "X-{api}-Signature",
        "{prefix}",
        lambda prefix, api_key, sig, timestamp, **_: f"{prefix} sig={sig},key={api_key},ts={timestamp}",
    ),
    (
        "X-{api}-Token",
        "{prefix}",
        lambda prefix, api_key, sig, timestamp, **_: f"{prefix} {timestamp}:{sig}",
    ),
]


def _gen_auth_protocol(rng: random.Random, seed: int) -> dict:
    """Generate an auth protocol task with randomized parameters."""
    api_name = rng.choice(_API_NAMES)
    api_version = f"{rng.randint(1, 9)}.{rng.randint(0, 9)}"
    api_prefix = api_name[:3].upper()

    hash_id, hash_name, hash_func = rng.choice(_HASH_ALGOS)

    signing_fmt_name, signing_fmt_template, signing_builder = rng.choice(_SIGNING_FORMATS)

    # Does this format use a method parameter?
    uses_method = "method" in signing_fmt_template

    header_template_name, header_prefix_template, header_builder = rng.choice(_HEADER_FORMATS)
    header_name = header_template_name.replace("{api}", api_name)
    header_prefix = header_prefix_template.replace("{prefix}", f"{api_prefix}")

    # Determine function signature based on whether method is needed
    if uses_method:
        func_sig = "api_key: str, timestamp: int, method: str = 'GET'"
        func_name = "generate_auth_header"
    else:
        func_sig = "api_key: str, timestamp: int"
        func_name = "generate_auth_header"

    # Build the signing string description
    signing_desc = signing_fmt_template.replace("{api_key}", "API_KEY").replace(
        "{timestamp}", "TIMESTAMP"
    ).replace("{method}", "METHOD")

    # Build expected computation function
    def compute_expected(api_key, timestamp, method="GET"):
        signing_string = signing_builder(
            api_key=api_key, timestamp=timestamp, method=method
        )
        digest = hmac.new(
            api_key.encode(), signing_string.encode(), hash_func
        ).digest()
        sig = base64.b64encode(digest).decode()
        return {
            header_name: header_builder(
                prefix=header_prefix, api_key=api_key,
                sig=sig, timestamp=timestamp, method=method,
            )
        }

    # Build test cases
    test_keys = [
        (f"test_key_{seed}", 1700000000 + seed),
        (f"another_key_{seed}", 1700000001 + seed),
        (f"k{seed}", seed),
    ]
    if uses_method:
        test_keys_with_method = [
            (k, t, rng.choice(["GET", "POST", "PUT", "DELETE"]))
            for k, t in test_keys
        ]
    else:
        test_keys_with_method = [(k, t, "GET") for k, t in test_keys]

    test_cases = []
    for api_key, timestamp, method in test_keys_with_method:
        expected = compute_expected(api_key, timestamp, method)
        if uses_method:
            test_cases.append({
                "args": [api_key, timestamp, method],
                "check": lambda result, exp=expected: (
                    isinstance(result, dict) and result == exp
                ),
            })
        else:
            test_cases.append({
                "args": [api_key, timestamp],
                "check": lambda result, exp=expected: (
                    isinstance(result, dict) and result == exp
                ),
            })

    verifier = _exec_verifier(func_name, test_cases)

    # Skill content (the procedural knowledge the agent needs)
    if uses_method:
        signing_step = (
            f"2. Build signing string: `{signing_desc}` "
            f"(concatenate the API key, timestamp, and HTTP method)"
        )
    else:
        signing_step = (
            f"2. Build signing string: `{signing_desc}` "
            f"(concatenate the API key and timestamp)"
        )

    skill_content = (
        f"# {api_name} API v{api_version} Authentication\n\n"
        f"## Auth Header Generation\n\n"
        f"To authenticate requests to the {api_name} API:\n\n"
        f"1. Obtain your API key from the dashboard\n"
        f"{signing_step}\n"
        f"3. Compute HMAC-{hash_name} of the signing string, "
        f"using the API key as the HMAC key\n"
        f"4. Base64-encode the raw digest bytes\n"
        f"5. Set header `{header_name}` to: "
        f"`{header_builder(prefix=header_prefix, api_key='<KEY>', sig='<SIG>', timestamp='<TS>', method='<METHOD>')}`\n\n"
        f"## Example\n\n"
        f"```python\n"
        f"import hmac, hashlib, base64\n\n"
        f"def {func_name}({func_sig}):\n"
        f"    signing_string = f\"{signing_fmt_template}\"\n"
        f"    digest = hmac.new(api_key.encode(), signing_string.encode(), hashlib.{hash_id}).digest()\n"
        f"    sig = base64.b64encode(digest).decode()\n"
        f"    return {{'{header_name}': ..."
        f"}}\n"
        f"```\n\n"
        f"**Important**: Use `hmac.new()` (not `hashlib` directly) with `hashlib.{hash_id}` as the digest algorithm.\n"
        f"The HMAC key is always the API key encoded as UTF-8.\n"
    )

    # Task description
    if uses_method:
        task_desc = (
            f"Write a Python function `{func_name}({func_sig})` that generates "
            f"the authentication header for the {api_name} API v{api_version}.\n\n"
            f"The function should:\n"
            f"1. Build the signing string by combining the API key, timestamp, and HTTP method "
            f"in the format: `{signing_desc}`\n"
            f"2. Compute the HMAC-{hash_name} digest using the API key as the HMAC key\n"
            f"3. Base64-encode the raw digest\n"
            f"4. Return a dict with a single key `{header_name}` containing the formatted header value\n\n"
            f"You will need to invoke the relevant skill to learn the exact header format and signing protocol."
        )
    else:
        task_desc = (
            f"Write a Python function `{func_name}({func_sig})` that generates "
            f"the authentication header for the {api_name} API v{api_version}.\n\n"
            f"The function should:\n"
            f"1. Build the signing string by combining the API key and timestamp "
            f"in the format: `{signing_desc}`\n"
            f"2. Compute the HMAC-{hash_name} digest using the API key as the HMAC key\n"
            f"3. Base64-encode the raw digest\n"
            f"4. Return a dict with a single key `{header_name}` containing the formatted header value\n\n"
            f"You will need to invoke the relevant skill to learn the exact header format and signing protocol."
        )

    # Skill entry
    skill_id = f"skill_proc_auth_{seed}"
    skill = {
        "id": skill_id,
        "name": f"{api_name} API Authentication",
        "short_description": (
            f"Authentication protocol for the {api_name} API v{api_version}. "
            f"Covers signing, header format, and HMAC computation."
        ),
        "full_content": skill_content,
    }

    # Pick 2 distractor skills
    distractor_ids = [d["id"] for d in rng.sample(_DISTRACTOR_SKILLS, 2)]

    task = {
        "id": f"task_proc_auth_{seed}",
        "description": task_desc,
        "difficulty": "easy",
        "relevant_skills": [skill_id],
        "distractor_skills": distractor_ids,
        "verifier": verifier,
        "source": "procedural",
        "template": "auth_protocol",
    }

    # Generated skills dict (relevant + distractors)
    generated_skills = {skill_id: skill}
    for d in _DISTRACTOR_SKILLS:
        generated_skills[d["id"]] = d

    return {"task": task, "skills": generated_skills}


# ---------------------------------------------------------------------------
# Template 2: Binary Format
# ---------------------------------------------------------------------------

_FORMAT_NAMES = [
    "NovaBin", "HexPack", "DataForge", "ByteStream", "PacketX",
    "BinFrame", "CrystalPack", "FluxBinary", "QuantumPack", "NexusBin",
    "VectorPack", "PulseBin", "ArchivX", "StreamPack", "CoreBin",
    "SignalPack", "MatrixBin", "GridPack", "TensorBin", "WavePack",
]

_MAGIC_BYTES_OPTIONS = [
    (b"NOVB", "NOVB"), (b"HXPK", "HXPK"), (b"DFGE", "DFGE"),
    (b"BYST", "BYST"), (b"PKTX", "PKTX"), (b"BNFR", "BNFR"),
    (b"CRPK", "CRPK"), (b"FLXB", "FLXB"), (b"QPAK", "QPAK"),
    (b"NXBN", "NXBN"),
]

_FLAG_SETS = [
    # (flag_names, bit_positions)
    (["compressed", "encrypted", "checksummed"], [0, 1, 2]),
    (["compressed", "signed", "indexed"], [0, 1, 2]),
    (["encrypted", "compressed", "verified"], [0, 1, 2]),
    (["indexed", "compressed", "encrypted", "signed"], [0, 1, 2, 3]),
]


def _gen_binary_format(rng: random.Random, seed: int) -> dict:
    """Generate a binary format parsing task with randomized parameters."""
    format_name = rng.choice(_FORMAT_NAMES)
    magic_bytes, magic_str = rng.choice(_MAGIC_BYTES_OPTIONS)
    endian = rng.choice(["big", "little"])
    endian_char = ">" if endian == "big" else "<"

    # Version format: major.minor packed as 16-bit
    version_major = rng.randint(1, 5)
    version_minor = rng.randint(0, 9)
    version_packed = (version_major << 8) | version_minor

    # Flag configuration
    flag_names, flag_bits = rng.choice(_FLAG_SETS)

    # Choose header fields order (always: magic, version, record_count, flags, crc32)
    func_name = "parse_header"

    # Build test headers
    def build_header(record_count: int, flag_values: dict) -> bytes:
        buf = bytearray()
        buf += magic_bytes
        buf += struct.pack(f"{endian_char}H", version_packed)
        buf += struct.pack(f"{endian_char}I", record_count)
        flag_int = 0
        for fname, fbit in zip(flag_names, flag_bits):
            if flag_values.get(fname, False):
                flag_int |= (1 << fbit)
        buf += struct.pack(f"{endian_char}H", flag_int)
        # CRC32 of everything so far
        crc = binascii.crc32(bytes(buf)) & 0xFFFFFFFF
        buf += struct.pack(f"{endian_char}I", crc)
        return bytes(buf)

    def expected_parse(record_count: int, flag_values: dict) -> dict:
        result = {
            "version": version_packed,
            "record_count": record_count,
        }
        for fname in flag_names:
            result[fname] = flag_values.get(fname, False)
        return result

    # Test case 1: some flags set
    flags1 = {}
    for fname in flag_names:
        flags1[fname] = rng.choice([True, False])
    # Ensure at least one flag is True
    flags1[flag_names[0]] = True
    header1 = build_header(42 + seed % 100, flags1)
    expected1 = expected_parse(42 + seed % 100, flags1)

    # Test case 2: no flags
    flags2 = {fname: False for fname in flag_names}
    header2 = build_header(1, flags2)
    expected2 = expected_parse(1, flags2)

    # Test case 3: all flags set
    flags3 = {fname: True for fname in flag_names}
    header3 = build_header(1000 + seed, flags3)
    expected3 = expected_parse(1000 + seed, flags3)

    test_cases = [
        {
            "args": [header1],
            "check": lambda r, exp=expected1: (
                isinstance(r, dict)
                and r.get("record_count") == exp["record_count"]
                and all(r.get(fn) == exp[fn] for fn in exp if fn not in ("version",))
            ),
        },
        {
            "args": [header2],
            "check": lambda r, exp=expected2: (
                isinstance(r, dict)
                and r.get("record_count") == exp["record_count"]
                and all(r.get(fn) is False for fn in flag_names)
            ),
        },
        {
            "args": [header3],
            "check": lambda r, exp=expected3: (
                isinstance(r, dict)
                and r.get("record_count") == exp["record_count"]
                and all(r.get(fn) is True for fn in flag_names)
            ),
        },
    ]

    verifier = _exec_verifier(func_name, test_cases)

    # Flag description for skill content
    flag_desc_lines = []
    for fname, fbit in zip(flag_names, flag_bits):
        flag_desc_lines.append(f"  - Bit {fbit}: `{fname}`")
    flag_desc = "\n".join(flag_desc_lines)

    header_size = 4 + 2 + 4 + 2 + 4  # magic + version + record_count + flags + crc32

    skill_content = (
        f"# {format_name} Binary Format Specification\n\n"
        f"## Header Layout ({header_size} bytes)\n\n"
        f"| Offset | Size | Field | Description |\n"
        f"|--------|------|-------|-------------|\n"
        f"| 0 | 4 | Magic | `{magic_str}` (ASCII) |\n"
        f"| 4 | 2 | Version | {endian}-endian uint16, packed as (major << 8) | minor |\n"
        f"| 6 | 4 | Record Count | {endian}-endian uint32 |\n"
        f"| 10 | 2 | Flags | {endian}-endian uint16, bitfield |\n"
        f"| 12 | 4 | CRC32 | {endian}-endian uint32, CRC32 of bytes 0-11 |\n\n"
        f"## Flags Bitfield\n\n"
        f"{flag_desc}\n\n"
        f"## Byte Order\n\n"
        f"All multi-byte fields are **{endian}-endian** "
        f"(struct format: `'{endian_char}'`).\n\n"
        f"## Validation\n\n"
        f"1. Check magic bytes match `{magic_str}`\n"
        f"2. Compute CRC32 of bytes 0..11 and compare with stored CRC32 at offset 12\n"
        f"3. If CRC mismatch, raise ValueError\n\n"
        f"## Parsing Example\n\n"
        f"```python\n"
        f"import struct, binascii\n\n"
        f"def {func_name}(data: bytes) -> dict:\n"
        f"    magic = data[0:4]\n"
        f"    assert magic == b'{magic_str}'\n"
        f"    version = struct.unpack('{endian_char}H', data[4:6])[0]\n"
        f"    record_count = struct.unpack('{endian_char}I', data[6:10])[0]\n"
        f"    flags = struct.unpack('{endian_char}H', data[10:12])[0]\n"
        f"    crc_stored = struct.unpack('{endian_char}I', data[12:16])[0]\n"
        f"    crc_computed = binascii.crc32(data[0:12]) & 0xFFFFFFFF\n"
        f"    if crc_stored != crc_computed:\n"
        f"        raise ValueError('CRC mismatch')\n"
        f"    return {{\n"
        f"        'version': version,\n"
        f"        'record_count': record_count,\n"
        f"        ...  # extract flags from bitfield\n"
        f"    }}\n"
        f"```\n"
    )

    task_desc = (
        f"Write a Python function `{func_name}(data: bytes) -> dict` that parses "
        f"a {format_name} binary file header.\n\n"
        f"The function should:\n"
        f"1. Validate the 4-byte magic number\n"
        f"2. Parse the version (uint16), record count (uint32), and flags (uint16)\n"
        f"3. Validate the CRC32 checksum\n"
        f"4. Return a dict with keys: `version`, `record_count`, "
        + ", ".join(f"`{fn}`" for fn in flag_names) + " (booleans from bitfield)\n\n"
        f"The exact byte layout, endianness, and flag bit positions are specified "
        f"in the {format_name} format skill. You must invoke it to get the details."
    )

    skill_id = f"skill_proc_bin_{seed}"
    skill = {
        "id": skill_id,
        "name": f"{format_name} Format Specification",
        "short_description": (
            f"Binary header format for {format_name} files. "
            f"Defines magic bytes, field layout, flags, and CRC32 validation."
        ),
        "full_content": skill_content,
    }

    distractor_ids = [d["id"] for d in rng.sample(_DISTRACTOR_SKILLS, 2)]

    task = {
        "id": f"task_proc_bin_{seed}",
        "description": task_desc,
        "difficulty": "easy",
        "relevant_skills": [skill_id],
        "distractor_skills": distractor_ids,
        "verifier": verifier,
        "source": "procedural",
        "template": "binary_format",
    }

    generated_skills = {skill_id: skill}
    for d in _DISTRACTOR_SKILLS:
        generated_skills[d["id"]] = d

    return {"task": task, "skills": generated_skills}


# ---------------------------------------------------------------------------
# TaskGenerator
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "auth_protocol": _gen_auth_protocol,
    "binary_format": _gen_binary_format,
}


class TaskGenerator:
    """
    Procedural task generator with seeded randomization.

    Usage:
        gen = TaskGenerator(seed=42)
        result = gen.generate()  # returns {"task": ..., "skills": ...}
        result = gen.generate(template="auth_protocol")  # specific template
    """

    def __init__(self, seed: int = 0):
        self._base_seed = seed
        self._counter = 0

    def generate(self, template: str | None = None) -> dict:
        """
        Generate a task. Returns {"task": dict, "skills": dict}.

        Args:
            template: Optional template name. If None, picks randomly.
        """
        episode_seed = self._base_seed * 10000 + self._counter
        self._counter += 1
        rng = random.Random(episode_seed)

        if template is None:
            template = rng.choice(list(_TEMPLATES.keys()))

        if template not in _TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Available: {list(_TEMPLATES.keys())}")

        return _TEMPLATES[template](rng, episode_seed)

    def generate_with_seed(self, seed: int, template: str | None = None) -> dict:
        """
        Generate a task with an explicit seed (deterministic).

        Args:
            seed: Exact seed to use for this generation.
            template: Optional template name.
        """
        rng = random.Random(seed)

        if template is None:
            template = rng.choice(list(_TEMPLATES.keys()))

        if template not in _TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Available: {list(_TEMPLATES.keys())}")

        return _TEMPLATES[template](rng, seed)

    @property
    def available_templates(self) -> list[str]:
        return list(_TEMPLATES.keys())
