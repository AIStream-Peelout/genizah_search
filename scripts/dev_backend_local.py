"""Run a second backend instance locally against the running docker-compose services.

This lets backend changes be exercised end-to-end (for example with
``scripts/run_agentic_rag_eval.py --api-base-url http://127.0.0.1:8010``)
without rebuilding or restarting the production backend container.

Neo4j credentials are read from the running backend container at start so
they never need to be written to disk. Every setting can be overridden with the
corresponding environment variable.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/dev_backend_local.py
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def read_container_env(container: str, name: str) -> str:
    """Read one environment variable from a running docker container.

    :param container: Container name.
    :param name: Environment variable name.
    :returns: The value, or an empty string when unavailable.
    :rtype: str
    """
    result = subprocess.run(
        ["docker", "exec", container, "sh", "-c", f'printf %s "${name}"'],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def main() -> None:
    """Configure the environment and start uvicorn on the dev port."""
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    container = os.getenv("DEV_BACKEND_CONTAINER", "genizah_search-backend-1")
    defaults = {
        "NEO4J_USER": read_container_env(container, "NEO4J_USER"),
        "NEO4J_PASSWORD": read_container_env(container, "NEO4J_PASSWORD"),
        "NEO4J_URI": "bolt://127.0.0.1:7681",
        "LLM_STUDIO_URL": "http://127.0.0.1:1234",
        "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8001",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(project_root), os.environ.get("PYTHONPATH", "")] if part
    )
    sys.path.insert(0, str(project_root))

    import uvicorn

    uvicorn.run(
        "src.backend.app:app",
        host="127.0.0.1",
        port=int(os.getenv("DEV_BACKEND_PORT", "8010")),
    )


if __name__ == "__main__":
    main()
