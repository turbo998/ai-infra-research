from contextlib import redirect_stderr, redirect_stdout
import io
import json

from scripts import validate_research as validator
from tests.fixture_support import FixtureCase, ledgers


class ValidatorTests(FixtureCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = ledgers()
        self.write("docs/result.md", "An original synthetic artifact.\n")
        self.write("docs/daily/2026-10-12.md", "[Result](../result.md#section)\n")
        self.save()

    def save(self) -> None:
        for name, value in self.data.items():
            self.write_json("research/" + name, value)

    def invalid(self, message: str) -> None:
        self.save()
        with self.assertRaisesRegex(validator.ValidationError, message):
            validator.validate(self.root)

    def test_valid_pending_and_blocked_sources(self) -> None:
        self.data["sources.json"]["sources"][0]["access_status"] = "blocked"
        self.data["queue.json"]["items"][1]["status"] = "blocked"
        self.save()
        self.assertEqual(validator.validate(self.root), {"sources": 1, "claims": 1, "questions": 2})

    def test_cli_nonzero_stderr_on_missing_json(self) -> None:
        (self.root / "research" / "sources.json").unlink()
        stream = io.StringIO()
        with redirect_stderr(stream):
            self.assertEqual(validator.main(["--root", str(self.root)]), 1)
        self.assertIn("Validation failed", stream.getvalue())
        self.assertNotIn(str(self.root), stream.getvalue())

    def test_cli_success(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(validator.main(["--root", str(self.root)]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "valid")

    def test_bad_json_and_duplicate_keys(self) -> None:
        for content in ('{"bad":', '{"schema_version":1,"schema_version":1,"sources":[]}'):
            with self.subTest(content=content):
                self.write("research/sources.json", content)
                with self.assertRaises(validator.ValidationError):
                    validator.validate(self.root)

    def test_nonfinite_json(self) -> None:
        self.write("research/sources.json", '{"schema_version": 1, "sources": [], "extra": NaN}')
        with self.assertRaisesRegex(validator.ValidationError, "non-finite"):
            validator.validate(self.root)

    def test_duplicate_and_invalid_ids(self) -> None:
        source = self.data["sources.json"]["sources"][0]
        self.data["sources.json"]["sources"].append(dict(source))
        self.invalid("duplicate")
        self.data["sources.json"]["sources"] = [source]
        source["id"] = "unsafe identifier"
        self.invalid("stable ID")

    def test_bad_schema_version(self) -> None:
        self.data["sources.json"]["schema_version"] = True
        self.invalid("schema_version")

    def test_missing_required_source_fields(self) -> None:
        del self.data["sources.json"]["sources"][0]["published_at"]
        self.invalid("missing nullable")

    def test_bad_source_url_and_timestamp(self) -> None:
        source = self.data["sources.json"]["sources"][0]
        for url in ("file:local", "https://localhost/spec", "https://127.0.0.1/", "https://user@example.org/"):
            source["url"] = url
            self.invalid("source.url")
        source["url"] = "https://docs.kernel.org/"
        source["verified_at"] = "yesterday"
        self.invalid("ISO")

    def test_unknown_evidence_source_and_queue_refs(self) -> None:
        claim = self.data["evidence.json"]["claims"][0]
        claim["source_ids"] = ["missing"]
        self.invalid("unknown reference")
        claim["source_ids"] = ["src-one"]
        item = self.data["queue.json"]["items"][1]
        item["depends_on"] = ["missing"]
        self.invalid("unknown reference")
        item["depends_on"] = ["q-one"]
        item["evidence_ids"] = ["missing"]
        self.invalid("unknown reference")

    def test_cycle_and_self_dependency(self) -> None:
        items = self.data["queue.json"]["items"]
        items[0]["status"] = "pending"
        items[0]["depends_on"] = ["q-two"]
        self.invalid("cyclic")
        items[0]["depends_on"] = ["q-one"]
        self.invalid("cyclic")

    def test_done_requires_all_dependencies_not_any(self) -> None:
        items = self.data["queue.json"]["items"]
        items.append({
            **items[0], "id": "q-three", "depends_on": ["q-one", "q-two"],
        })
        self.invalid("all prerequisites done")
        items[2]["depends_on"] = []
        items[2]["artifacts"] = []
        self.invalid("requires artifacts")
        items[2]["artifacts"] = ["docs/result.md"]
        items[2]["evidence_ids"] = []
        self.invalid("requires artifacts")

    def test_artifact_missing_absolute_and_escape(self) -> None:
        for artifact in ("missing.md", "../outside.md", "/absolute.md", "docs\\result.md", "./docs/result.md"):
            self.data["evidence.json"]["claims"][0]["artifacts"] = [artifact]
            self.invalid("artifacts")

    def test_unsupported_evidence_states_and_source_access(self) -> None:
        claim = self.data["evidence.json"]["claims"][0]
        for state in ("proven", [], None):
            claim["status"] = state
            self.invalid("unsupported evidence")
        claim["status"] = "specification"
        self.data["sources.json"]["sources"][0]["access_status"] = []
        self.invalid("unsupported state")

    def test_nonown_requires_sources_including_unverified(self) -> None:
        claim = self.data["evidence.json"]["claims"][0]
        claim["source_ids"] = []
        for state in ("specification", "unverified"):
            claim["status"] = state
            self.invalid("requires sources")

    def test_offline_requires_executable_and_results(self) -> None:
        claim = self.data["evidence.json"]["claims"][0]
        claim.update(status="own-offline", source_ids=[])
        self.invalid("requires experiment")
        self.write("experiments/original/run.py", 'print("synthetic")\n')
        self.write_json("experiments/original/results.json", {"synthetic": True})
        claim["artifacts"] = ["experiments/original/run.py", "experiments/original/results.json"]
        self.save()
        validator.validate(self.root)

    def test_hardware_requires_environment_and_raw_results(self) -> None:
        claim = self.data["evidence.json"]["claims"][0]
        claim.update(status="own-hardware", source_ids=[])
        self.invalid("environment and raw-results")
        self.write_json("experiments/original/environment.json", {"synthetic": True})
        self.write("experiments/original/raw-results.csv", "synthetic,value\ntrue,1\n")
        claim["artifacts"] = ["experiments/original/environment.json", "experiments/original/raw-results.csv"]
        self.save()
        validator.validate(self.root)

    def test_progress_refs_mastery_and_date(self) -> None:
        progress = self.data["progress.json"]
        progress["last_completed_question"] = "q-two"
        self.invalid("completed queue reference")
        progress["last_completed_question"] = "q-one"
        progress["learning_mastery"] = "mastered"
        self.invalid("not-assessed")
        progress["learning_mastery"] = "not-assessed"
        progress["last_run_date"] = "2026-02-30"
        self.invalid("ISO date")
        progress["last_run_date"] = "2026-10-11"
        self.invalid("latest daily")
        progress["last_run_date"] = "2026-10-12"
        progress["monitor_report"] = "missing.json"
        self.invalid("monitor_report")

    def test_initial_progress_can_have_no_completed_question(self) -> None:
        self.data["queue.json"]["items"][0]["status"] = "pending"
        self.data["progress.json"]["last_completed_question"] = ""
        self.save()
        validator.validate(self.root)

    def test_daily_filename_and_duplicate_day(self) -> None:
        self.write("docs/daily/not-a-day.md", "Synthetic\n")
        self.invalid("filenames")
        (self.root / "docs/daily/not-a-day.md").unlink()
        self.write("docs/daily/subdir/2026-10-12.md", "Synthetic\n")
        self.invalid("duplicate day")

    def test_local_links_images_references_and_fragments(self) -> None:
        self.write(
            "README.md",
            "[Result](docs/result.md#heading)\n![Image](docs/result.md)\n"
            "[reference][r]\n[r]: docs/result.md#heading\n"
            "[URL](https://example.org/not-downloaded)\n[Anchor](#heading)\n"
            "[Root](/docs/result.md)\n[Encoded](docs/%72esult.md)\n"
            "```markdown\n[example](nonexistent)\n```\n",
        )
        validator.validate(self.root)

    def test_missing_links_and_traversal(self) -> None:
        for target in ("missing.md", "../outside.md", "%2e%2e/outside.md", "file:docs/result.md"):
            self.write("README.md", f"[Bad]({target})\n")
            self.invalid("invalid markdown link")
        self.write("README.md", "[reference][r]\n[r]: missing.md\n")
        self.invalid("invalid markdown link")

    def test_publication_redacts_private_path_and_credential(self) -> None:
        private = "C:" + "\\" + "Users" + "\\" + "synthetic-person" + "\\" + "private"
        self.write("README.md", private)
        with self.assertRaises(validator.ValidationError) as caught:
            validator.validate(self.root)
        self.assertNotIn(private, str(caught.exception))
        self.assertIn("local private path", str(caught.exception))
        candidate = "synthetic" + "Opaque" + "Credential"
        self.write("README.md", "API_KEY" + ' = "' + candidate + '"')
        with self.assertRaises(validator.ValidationError) as caught:
            validator.validate(self.root)
        self.assertNotIn(candidate, str(caught.exception))
        self.assertIn("credential assignment", str(caught.exception))

    def test_routine_secret_documentation_is_not_flagged(self) -> None:
        self.write(
            "README.md",
            'Do not publish secrets. API_KEY="your-placeholder-key"\n'
            "Use environment variables and managed identities.\n",
        )
        validator.validate(self.root)

    def test_unquoted_environment_assignment_is_redacted(self) -> None:
        candidate = "synthetic" + "Opaque" + "Credential"
        self.write(".env", "API_KEY" + "=" + candidate + "\n")
        with self.assertRaises(validator.ValidationError) as error:
            validator.validate(self.root)
        self.assertIn("credential assignment", str(error.exception))
        self.assertNotIn(candidate, str(error.exception))
