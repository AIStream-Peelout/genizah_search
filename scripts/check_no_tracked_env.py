"""Fail when Git tracks a runtime ``.env`` credential file."""

from pathlib import PurePosixPath
import subprocess
from typing import Sequence


ALLOWED_ENV_FILENAMES = {".env.example"}


def tracked_files() -> list[str]:
    """Return all paths currently present in Git's index.

    :returns: Repository-relative tracked paths.
    :rtype: list[str]
    """
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]


def forbidden_env_files(paths: Sequence[str]) -> list[str]:
    """Select tracked runtime environment files from a path collection.

    :param paths: Repository-relative paths to inspect.
    :returns: Sorted paths whose basename is ``.env`` or begins with
        ``.env.`` other than the approved example template.
    :rtype: list[str]
    """
    forbidden: list[str] = []
    for path in paths:
        filename = PurePosixPath(path).name
        if filename in ALLOWED_ENV_FILENAMES:
            continue
        if filename == ".env" or filename.startswith(".env."):
            forbidden.append(path)
    return sorted(forbidden)


def main() -> int:
    """Run the tracked-environment-file guard.

    :returns: Zero when the index is safe, otherwise one.
    :rtype: int
    """
    forbidden = forbidden_env_files(tracked_files())
    if not forbidden:
        print("OK: Git is not tracking runtime .env files.")
        return 0

    print("ERROR: Git must never track runtime .env files:")
    for path in forbidden:
        print(f"  - {path}")
    print("Remove them from the index with: git rm --cached <path>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
