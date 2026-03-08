"""
FastAPI application for the Skill Invocation Environment.

Exposes the SkillInvocationEnvironment over HTTP and WebSocket endpoints.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core>=0.2.1"
    ) from e

from models import SkillInvocationAction, SkillInvocationObservation
from .skill_invocation_env_environment import SkillInvocationEnvironment

app = create_app(
    SkillInvocationEnvironment,
    SkillInvocationAction,
    SkillInvocationObservation,
    env_name="skill_invocation_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
