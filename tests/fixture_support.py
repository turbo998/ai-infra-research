"""Original disposable fixtures confined to the owned tests directory."""

import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4


FIXTURE_ROOT = Path(__file__).resolve().parent / "_fixtures"


class FixtureCase(unittest.TestCase):
    """Create and clean isolated in-repository fixture directories."""

    def setUp(self) -> None:
        self.root = FIXTURE_ROOT / uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.remove_fixture)

    def remove_fixture(self) -> None:
        """Remove only this test's exact fixture and the empty container."""
        shutil.rmtree(self.root)
        if FIXTURE_ROOT.exists() and not any(FIXTURE_ROOT.iterdir()):
            FIXTURE_ROOT.rmdir()

    def write(self, relative: str, content: str) -> Path:
        """Write original test data inside this fixture."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, relative: str, value: object) -> Path:
        """Write original JSON test data inside this fixture."""
        return self.write(relative, json.dumps(value))


def ledgers() -> dict[str, dict]:
    """Return a valid minimal snapshot with a completed and an unfinished question."""
    return {
        "sources.json": {
            "schema_version": 1,
            "sources": [{
                "id": "src-one", "title": "Original synthetic source",
                "url": "https://docs.kernel.org/", "kind": "documentation",
                "version": None, "published_at": None, "verified_at": "2026-10-12T00:00:00+00:00",
                "read_scope": "synthetic fixture", "access_status": "partial", "limitations": "",
            }],
        },
        "evidence.json": {
            "schema_version": 1,
            "claims": [{
                "id": "claim-one", "claim": "Synthetic structural claim",
                "source_ids": ["src-one"], "status": "specification",
                "artifacts": ["docs/result.md"], "limitations": "",
                "verified_at": "2026-10-12", "verifier": "offline-unit-test",
            }],
        },
        "queue.json": {
            "schema_version": 1,
            "items": [
                {
                    "id": "q-one", "title": "Original question", "component": "memory",
                    "priority": 1, "depends_on": [], "status": "done",
                    "artifacts": ["docs/result.md"], "evidence_ids": ["claim-one"], "next_step": "",
                },
                {
                    "id": "q-two", "title": "Pending question", "component": "memory",
                    "priority": 2, "depends_on": ["q-one"], "status": "pending",
                    "artifacts": [], "evidence_ids": [], "next_step": "Continue offline",
                },
            ],
        },
        "progress.json": {
            "schema_version": 1, "last_run_date": "2026-10-12",
            "last_completed_question": "q-one", "active_branch": "research/test",
            "active_pr": None, "research_artifacts": ["docs/result.md"],
            "learning_mastery": "not-assessed", "blocked": [], "monitor_report": None,
        },
    }
