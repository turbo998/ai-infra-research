from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import json
import subprocess
from unittest.mock import patch

from scripts import research_lock as lock
from tests.fixture_support import FixtureCase


class ResearchLockTests(FixtureCase):
    def setUp(self) -> None:
        super().setUp()
        self.path = self.root / lock.LOCK_NAME

    def test_matching_persistent_owner_acquire_status_release(self) -> None:
        self.assertEqual(lock.status(self.path), {"locked": False, "owner": None})
        self.assertEqual(lock.acquire(self.path, "session-one"), {"locked": True, "owner": "session-one"})
        value = json.loads(self.path.read_text())
        self.assertIsNotNone(datetime.fromisoformat(value["acquired_at"]).tzinfo)
        self.assertEqual(lock.status(self.path), {"locked": True, "owner": "session-one"})
        self.assertEqual(lock.release(self.path, "session-one"), {"locked": False, "owner": None})
        self.assertFalse(self.path.exists())

    def test_repeated_acquire_never_steals_even_same_owner_or_old_age(self) -> None:
        lock.acquire(self.path, "session-one")
        contents = json.loads(self.path.read_text())
        contents["acquired_at"] = "2000-01-01T00:00:00+00:00"
        self.path.write_text(json.dumps(contents))
        before = self.path.read_bytes()
        for owner in ("session-one", "session-two"):
            with self.assertRaisesRegex(lock.LockError, "lock_exists owner=session-one"):
                lock.acquire(self.path, owner)
        self.assertEqual(self.path.read_bytes(), before)

    def test_release_mismatch_and_repeated_release(self) -> None:
        lock.acquire(self.path, "session-one")
        with self.assertRaisesRegex(lock.LockError, "owner_mismatch"):
            lock.release(self.path, "session-two")
        self.assertTrue(self.path.exists())
        lock.release(self.path, "session-one")
        with self.assertRaisesRegex(lock.LockError, "lock_not_found"):
            lock.release(self.path, "session-one")

    def test_invalid_existing_lock_fails_closed_and_redacts(self) -> None:
        self.path.write_text('{"owner": "unsafe private value",')
        before = self.path.read_bytes()
        with self.assertRaisesRegex(lock.LockError, "owner=unavailable") as error:
            lock.acquire(self.path, "session-one")
        self.assertNotIn("unsafe private value", str(error.exception))
        with self.assertRaises(lock.LockError):
            lock.status(self.path)
        with self.assertRaises(lock.LockError):
            lock.release(self.path, "session-one")
        self.assertEqual(self.path.read_bytes(), before)

    def test_unsafe_owner_tokens_rejected(self) -> None:
        for owner in ("", "person@example.org", "path/value", "two tokens", "a" * 129, None):
            with self.subTest(owner=owner), self.assertRaises(lock.LockError):
                lock.acquire(self.path, owner)
        self.assertFalse(self.path.exists())

    def test_concurrent_exclusive_creation_has_exactly_one_owner(self) -> None:
        def attempt(owner: str) -> str | None:
            try:
                lock.acquire(self.path, owner)
            except lock.LockError:
                return None
            return owner

        with ThreadPoolExecutor(max_workers=2) as executor:
            owners = list(executor.map(attempt, ("session-one", "session-two")))
        winner = [owner for owner in owners if owner is not None]
        self.assertEqual(len(winner), 1)
        self.assertEqual(lock.status(self.path)["owner"], winner[0])
        lock.release(self.path, winner[0])

    def test_actual_git_common_dir_without_commit(self) -> None:
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True, capture_output=True, timeout=10)
        common = lock.common_git_dir(self.root)
        self.assertEqual(common.resolve(), (self.root / ".git").resolve())
        lock.acquire(common / lock.LOCK_NAME, "fixture-owner")
        self.assertTrue((self.root / ".git" / lock.LOCK_NAME).is_file())
        lock.release(common / lock.LOCK_NAME, "fixture-owner")

    def test_two_worktrees_share_common_directory_protocol(self) -> None:
        common = self.root / "common-git"
        common.mkdir()
        result = subprocess.CompletedProcess([], 0, stdout=str(common) + "\n", stderr="")
        with patch.object(lock.subprocess, "run", return_value=result), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(lock.main(["acquire", "--owner", "first", "--repo", str(self.root / "tree-one")]), 0)
            self.assertEqual(lock.main(["acquire", "--owner", "second", "--repo", str(self.root / "tree-two")]), 1)
            self.assertEqual(lock.main(["release", "--owner", "second", "--repo", str(self.root / "tree-two")]), 1)
            self.assertEqual(lock.main(["release", "--owner", "first", "--repo", str(self.root / "tree-two")]), 0)

    def test_git_failure_does_not_echo_output(self) -> None:
        result = subprocess.CompletedProcess([], 1, stdout="", stderr="private diagnostic")
        error = io.StringIO()
        with patch.object(lock.subprocess, "run", return_value=result), redirect_stderr(error):
            self.assertEqual(lock.main(["status", "--repo", str(self.root)]), 1)
        self.assertIn("git_common_dir_unavailable", error.getvalue())
        self.assertNotIn("private diagnostic", error.getvalue())

    def test_status_cli_only_safe_owner(self) -> None:
        result = subprocess.CompletedProcess([], 0, stdout=str(self.root) + "\n", stderr="")
        lock.acquire(self.path, "fixture-owner")
        output = io.StringIO()
        with patch.object(lock.subprocess, "run", return_value=result), redirect_stdout(output):
            self.assertEqual(lock.main(["status"]), 0)
        self.assertEqual(json.loads(output.getvalue()), {"locked": True, "owner": "fixture-owner"})
        self.assertNotIn(str(self.root), output.getvalue())
