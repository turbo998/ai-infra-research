"""Persistent, owner-matched research lock shared by Git worktrees."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_NAME = "ai-infra-research.lock"
OWNER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class LockError(ValueError):
    """A fail-closed lock protocol error."""


def common_git_dir(repo: Path) -> Path:
    """Locate the absolute shared Git directory without changing Git configuration."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LockError("git_common_dir_unavailable") from exc
    if result.returncode or not result.stdout.strip():
        raise LockError("git_common_dir_unavailable")
    path = Path(result.stdout.strip())
    if not path.is_absolute() or not path.is_dir():
        raise LockError("git_common_dir_invalid")
    return path


def checked_owner(owner: str | None) -> str:
    """Require an opaque printable token, never a path or email address."""
    if not isinstance(owner, str) or not OWNER_PATTERN.fullmatch(owner):
        raise LockError("owner_must_be_safe_opaque_token")
    return owner


def read_lock(path: Path) -> dict:
    """Read existing ownership; malformed or partially written files stay locked."""
    if path.is_symlink():
        raise LockError("lock_invalid_or_unreadable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockError("lock_invalid_or_unreadable") from exc
    if not isinstance(value, dict):
        raise LockError("lock_invalid_or_unreadable")
    checked_owner(value.get("owner"))
    try:
        timestamp = datetime.fromisoformat(value["acquired_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LockError("lock_invalid_or_unreadable") from exc
    if timestamp.tzinfo is None:
        raise LockError("lock_invalid_or_unreadable")
    return value


def acquire(path: Path, owner: str) -> dict:
    """Create a persistent exclusive file; never take over an existing lock."""
    owner = checked_owner(owner)
    value = {"owner": owner, "acquired_at": datetime.now(timezone.utc).isoformat()}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        try:
            previous_owner = read_lock(path)["owner"]
        except LockError:
            previous_owner = "unavailable"
        raise LockError(f"lock_exists owner={previous_owner}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return {"locked": True, "owner": owner}


def status(path: Path) -> dict:
    """Report only safe ownership, never automatically release a lock."""
    if not path.exists() and not path.is_symlink():
        return {"locked": False, "owner": None}
    return {"locked": True, "owner": read_lock(path)["owner"]}


def release(path: Path, owner: str) -> dict:
    """Remove a lock only after matching the supplied owner token."""
    owner = checked_owner(owner)
    if not path.exists() and not path.is_symlink():
        raise LockError("lock_not_found")
    if read_lock(path)["owner"] != owner:
        raise LockError("owner_mismatch")
    path.unlink()
    return {"locked": False, "owner": None}


def main(argv: list[str] | None = None) -> int:
    """Run acquire/status/release against the shared Git directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "status", "release"))
    parser.add_argument("--owner")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        path = common_git_dir(args.repo) / LOCK_NAME
        if args.action == "status":
            result = status(path)
        elif args.action == "acquire":
            result = acquire(path, checked_owner(args.owner))
        else:
            result = release(path, checked_owner(args.owner))
    except LockError as exc:
        print(f"Research lock failed: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("Research lock failed: lock_io_error", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
