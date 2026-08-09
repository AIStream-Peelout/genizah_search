"""Tests for the repository's tracked ``.env`` guard."""

from scripts.check_no_tracked_env import forbidden_env_files


def test_rejects_runtime_environment_files() -> None:
    """Reject root and nested runtime environment files."""
    assert forbidden_env_files([".env", "src/backend/.env.production"]) == [
        ".env",
        "src/backend/.env.production",
    ]


def test_allows_example_environment_files() -> None:
    """Allow committed placeholder templates and ordinary source files."""
    assert forbidden_env_files(
        [".env.example", "src/backend/.env.example", "src/backend/app.py"]
    ) == []
