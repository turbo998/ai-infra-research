from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAgenticWorkspaceDependencyCachingUvHardlinkNixGitWorktreeSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "agentic_workspace_dependency_caching_uv_hardlink_nix_git_worktree_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["workspace_ready_sec"], 1.0)
        self.assertLess(data["tiers"][3]["disk_footprint_gb_per_workspace"], 0.1)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "agentic-workspace-dependency-caching-uv-hardlink-nix-git-worktree-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "agentic-workspace-dependency-caching-uv-hardlink-nix-git-worktree-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "agentic-workspace-dependency-caching-uv-hardlink-nix-git-worktree-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "agentic-workspace-dependency-caching-uv-hardlink-nix-git-worktree-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "agentic-workspace-dependency-caching-uv-hardlink-nix-git-worktree-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
