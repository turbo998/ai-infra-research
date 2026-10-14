"""Check ledger structure, references, local links, and obvious publication hazards."""

import argparse
from datetime import date, datetime
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._-]*\Z")
EVIDENCE_STATES = {
    "announcement", "specification", "implementation", "author-benchmark",
    "own-offline", "own-hardware", "unverified",
}
PRIVATE_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s<>\"']+"
    r"|/(?:home|Users)/[A-Za-z0-9_.-]+/|\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_$.-]+\\)"
)
CREDENTIAL_ASSIGNMENT = re.compile(
    r"""(?ix)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|
    client[_-]?secret|secret[_-]?key|aws[_-]?secret[_-]?access[_-]?key)
    ["']?\s*[:=]\s*["']([A-Za-z0-9_./+=-]{8,})["']"""
)
ENV_CREDENTIAL_ASSIGNMENT = re.compile(
    r"""(?im)^[ \t]*(?:export[ \t]+)?(?:api[_-]?key|access[_-]?token|auth[_-]?token|
    password|client[_-]?secret|secret[_-]?key|aws[_-]?secret[_-]?access[_-]?key)
    [ \t]*=[ \t]*([A-Za-z0-9_./+=-]{8,})[ \t]*(?:\#.*)?$""",
    re.VERBOSE,
)
PUBLICATION_SUFFIXES = {
    ".md", ".json", ".py", ".txt", ".csv", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".sh", ".ps1", ".js", ".ts", ".html", ".rst", ".tex", ".xml",
}


class ValidationError(ValueError):
    """A checkable structural or publication-content failure."""


def read_json(path: Path) -> object:
    """Read JSON strictly, including duplicate-key and non-finite rejection."""
    def pairs_to_dict(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValidationError("non-finite JSON number")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_to_dict,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("missing, unreadable, or malformed JSON") from exc


def text(value: object, field: str, *, nonempty: bool = False) -> str:
    """Require a string without exposing its contents in an error."""
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValidationError(f"{field}: expected {'nonempty ' if nonempty else ''}string")
    return value


def iso_date(value: object, field: str) -> date:
    """Require the canonical YYYY-MM-DD representation of a calendar day."""
    string = text(value, field)
    try:
        parsed = date.fromisoformat(string)
    except ValueError as exc:
        raise ValidationError(f"{field}: invalid ISO date") from exc
    if parsed.isoformat() != string:
        raise ValidationError(f"{field}: expected YYYY-MM-DD")
    return parsed


def iso_timestamp(value: object, field: str) -> None:
    """Accept an ISO calendar date or an ISO date/time."""
    string = text(value, field, nonempty=True)
    if len(string) == 10:
        iso_date(string, field)
        return
    try:
        datetime.fromisoformat(string)
    except ValueError as exc:
        raise ValidationError(f"{field}: invalid ISO date/time") from exc
    if not re.match(r"\d{4}-\d{2}-\d{2}T", string):
        raise ValidationError(f"{field}: expected ISO date/time")


def string_list(value: object, field: str) -> list[str]:
    """Require a list of distinct strings."""
    if not isinstance(value, list):
        raise ValidationError(f"{field}: expected list")
    result = [text(item, field, nonempty=True) for item in value]
    if len(set(result)) != len(result):
        raise ValidationError(f"{field}: duplicate entries")
    return result


def local_file(root: Path, value: object, field: str) -> Path:
    """Resolve a canonical repository-relative POSIX file, rejecting escapes."""
    string = text(value, field, nonempty=True)
    path = PurePosixPath(string)
    if (
        "\\" in string or ":" in string or "?" in string or "#" in string
        or path.is_absolute() or ".." in path.parts or path.as_posix() != string
    ):
        raise ValidationError(f"{field}: expected safe repo-relative POSIX file")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValidationError(f"{field}: file missing or outside repository")
    return resolved


def ledger(root: Path, filename: str, collection: str) -> dict[str, dict]:
    """Load a version-one ledger and index its uniquely named records."""
    value = read_json(root / "research" / filename)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValidationError(f"{filename}: invalid schema_version")
    if value["schema_version"] != 1 or not isinstance(value.get(collection), list):
        raise ValidationError(f"{filename}: expected version-one {collection} list")
    records: dict[str, dict] = {}
    for record in value[collection]:
        if not isinstance(record, dict):
            raise ValidationError(f"{filename}: expected record object")
        identifier = text(record.get("id"), f"{filename}.id")
        if not ID_PATTERN.fullmatch(identifier) or identifier in records:
            raise ValidationError(f"{filename}: invalid or duplicate stable ID")
        records[identifier] = record
    return records


def references(record: dict, field: str, known: dict) -> list[str]:
    """Require a distinct list of references to known records."""
    identifiers = string_list(record.get(field), field)
    if any(identifier not in known for identifier in identifiers):
        raise ValidationError(f"{field}: unknown reference")
    return identifiers


def check_queue_cycles(items: dict[str, dict]) -> None:
    """Check conjunctive dependencies without recursion-depth assumptions."""
    remaining = {identifier: set(item["depends_on"]) for identifier, item in items.items()}
    completed: set[str] = set()
    while remaining:
        ready = [identifier for identifier, deps in remaining.items() if deps <= completed]
        if not ready:
            raise ValidationError("queue: cyclic dependencies")
        for identifier in ready:
            completed.add(identifier)
            del remaining[identifier]


def validate_ledgers(root: Path) -> dict:
    """Validate all four ledgers and their cross-file consistency."""
    sources = ledger(root, "sources.json", "sources")
    claims = ledger(root, "evidence.json", "claims")
    items = ledger(root, "queue.json", "items")
    for source in sources.values():
        for field in ("title", "kind", "read_scope"):
            text(source.get(field), f"source.{field}", nonempty=True)
        text(source.get("limitations"), "source.limitations")
        url = text(source.get("url"), "source.url", nonempty=True)
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"https", "http"} and bool(parsed.hostname)
            valid = valid and not parsed.username and not parsed.password
            parsed.port
        except ValueError as exc:
            raise ValidationError("source.url: invalid URL") from exc
        host = parsed.hostname or ""
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            private_host = host == "localhost" or host.endswith((".localhost", ".local"))
        else:
            private_host = not address.is_global or address.is_multicast
        if not valid or private_host or any(character.isspace() for character in url):
            raise ValidationError("source.url: expected public HTTP(S) URL")
        if not isinstance(source.get("access_status"), str) or source["access_status"] not in {"read", "partial", "blocked"}:
            raise ValidationError("source.access_status: unsupported state")
        for field in ("version", "published_at"):
            if field not in source:
                raise ValidationError(f"source.{field}: missing nullable field")
            if source[field] is not None:
                text(source[field], f"source.{field}", nonempty=True)
        if source["published_at"] is not None:
            iso_timestamp(source["published_at"], "source.published_at")
        iso_timestamp(source.get("verified_at"), "source.verified_at")
    for claim in claims.values():
        text(claim.get("claim"), "claim.claim", nonempty=True)
        text(claim.get("limitations"), "claim.limitations")
        text(claim.get("verifier"), "claim.verifier", nonempty=True)
        iso_timestamp(claim.get("verified_at"), "claim.verified_at")
        state = claim.get("status")
        if not isinstance(state, str) or state not in EVIDENCE_STATES:
            raise ValidationError("claim.status: unsupported evidence state")
        source_ids = references(claim, "source_ids", sources)
        paths = [
            local_file(root, value, "claim.artifacts")
            for value in string_list(claim.get("artifacts"), "claim.artifacts")
        ]
        if state not in {"own-offline", "own-hardware"} and not source_ids:
            raise ValidationError("claim.source_ids: non-own evidence requires sources")
        if state == "own-offline":
            executable = any(
                path.suffix == ".py" and path.is_relative_to(root / "experiments")
                for path in paths
            )
            results = any(
                path.suffix in {".json", ".csv"} and path.is_relative_to(root / "experiments")
                for path in paths
            )
            if not executable or not results:
                raise ValidationError("own-offline: requires experiment Python and result artifacts")
        if state == "own-hardware":
            environment = any(path.stem == "environment" for path in paths)
            raw_results = any(
                path.stem.startswith("raw-results") and path.suffix in {".json", ".csv"}
                for path in paths
            )
            if not environment or not raw_results:
                raise ValidationError("own-hardware: requires environment and raw-results artifacts")
    for item in items.values():
        for field in ("title", "component", "next_step"):
            text(item.get(field), f"queue.{field}", nonempty=field != "next_step")
        if type(item.get("priority")) is not int:
            raise ValidationError("queue.priority: expected integer")
        deps = references(item, "depends_on", items)
        evidence = references(item, "evidence_ids", claims)
        artifacts = string_list(item.get("artifacts"), "queue.artifacts")
        for artifact in artifacts:
            local_file(root, artifact, "queue.artifacts")
        if not isinstance(item.get("status"), str) or item["status"] not in {"pending", "in_progress", "done", "blocked"}:
            raise ValidationError("queue.status: unsupported state")
        if item["status"] == "done" and (
            not artifacts or not evidence or any(items[dep]["status"] != "done" for dep in deps)
        ):
            raise ValidationError("queue.done: requires artifacts, evidence, and all prerequisites done")
    check_queue_cycles(items)
    progress = read_json(root / "research" / "progress.json")
    if not isinstance(progress, dict) or type(progress.get("schema_version")) is not int:
        raise ValidationError("progress: invalid schema_version")
    if progress["schema_version"] != 1:
        raise ValidationError("progress: unsupported schema_version")
    run_date = iso_date(progress.get("last_run_date"), "progress.last_run_date")
    completed = text(progress.get("last_completed_question"), "progress.last_completed_question")
    if completed:
        if completed not in items or items[completed]["status"] != "done":
            raise ValidationError("progress.last_completed_question: not a completed queue reference")
    elif any(item["status"] == "done" for item in items.values()):
        raise ValidationError("progress.last_completed_question: missing completed question")
    text(progress.get("active_branch"), "progress.active_branch", nonempty=True)
    if "active_pr" not in progress or (
        progress["active_pr"] is not None
        and (type(progress["active_pr"]) is not int or progress["active_pr"] <= 0)
    ):
        raise ValidationError("progress.active_pr: expected positive integer or null")
    if progress.get("learning_mastery") != "not-assessed":
        raise ValidationError("progress.learning_mastery: must remain not-assessed")
    string_list(progress.get("blocked"), "progress.blocked")
    for artifact in string_list(progress.get("research_artifacts"), "progress.research_artifacts"):
        local_file(root, artifact, "progress.research_artifacts")
    if "monitor_report" not in progress:
        raise ValidationError("progress.monitor_report: missing nullable field")
    if progress["monitor_report"] is not None:
        local_file(root, progress["monitor_report"], "progress.monitor_report")
    daily = root / "docs" / "daily"
    days: set[date] = set()
    for path in sorted(daily.rglob("*.md")) if daily.exists() else []:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", path.name):
            raise ValidationError("docs/daily: filenames must be YYYY-MM-DD.md")
        day = iso_date(path.stem, "docs/daily")
        if day in days:
            raise ValidationError("docs/daily: duplicate day")
        days.add(day)
    if days and max(days) != run_date:
        raise ValidationError("progress.last_run_date: must match latest daily note")
    return {"sources": len(sources), "claims": len(claims), "questions": len(items)}


def check_markdown_links(root: Path, path: Path, content: str) -> None:
    """Check inline/image and reference-definition targets, ignoring code and anchors."""
    content = re.sub(r"(?ms)^```.*?^```\s*$|^~~~.*?^~~~\s*$", "", content)
    content = re.sub(r"`[^`\n]+`", "", content)
    targets = re.findall(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s]+?)(?:\s+[\"'][^)]*)?\s*\)", content)
    targets += re.findall(r"(?m)^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", content)
    for target in targets:
        target = target.strip("<>")
        if target.startswith("#"):
            continue
        parsed = urlsplit(target)
        if parsed.scheme.lower() == "file":
            raise ValidationError("markdown link: file URL is not a repository-relative path")
        if parsed.scheme and parsed.scheme.lower() not in {"file"}:
            if len(parsed.scheme) == 1:
                raise ValidationError("markdown link: absolute local path")
            continue
        if parsed.netloc:
            continue
        local = unquote(parsed.path)
        if not local:
            continue
        if "\\" in local or ":" in local or "\x00" in local:
            raise ValidationError("markdown link: unsafe local path")
        base = root if local.startswith("/") else path.parent
        resolved = (base / local.lstrip("/")).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            raise ValidationError("markdown link: missing target or repository escape")


def check_publication(root: Path) -> None:
    """Find obvious hazards; this is not semantic review or privacy certification."""
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        environment_file = path.name == ".env" or path.name.startswith(".env.")
        if ".git" in relative.parts or (
            path.suffix.lower() not in PUBLICATION_SUFFIXES and not environment_file
        ):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValidationError("publication: file outside repository")
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValidationError(f"publication: unreadable text file {relative.as_posix()}") from exc
        if PRIVATE_PATH.search(content):
            raise ValidationError(f"publication: clear local private path in {relative.as_posix()}")
        matches = list(CREDENTIAL_ASSIGNMENT.finditer(content))
        if environment_file:
            matches += list(ENV_CREDENTIAL_ASSIGNMENT.finditer(content))
        for match in matches:
            value = match.group(1).lower()
            if not any(
                marker in value
                for marker in ("example", "placeholder", "redacted", "your-", "your_", "dummy", "test-only")
            ):
                raise ValidationError(f"publication: possible credential assignment in {relative.as_posix()}")
        if path.suffix.lower() == ".md":
            try:
                check_markdown_links(root, path, content)
            except (ValidationError, ValueError) as exc:
                raise ValidationError(f"publication: invalid markdown link in {relative.as_posix()}") from exc


def validate(root: Path) -> dict:
    """Run structural checks and obvious-hazard checks against a repository root."""
    root = root.resolve()
    counts = validate_ledgers(root)
    check_publication(root)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Run the CLI, returning nonzero and a redacted diagnostic on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        counts = validate(args.root)
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("Validation failed: filesystem_io_error", file=sys.stderr)
        return 1
    print(json.dumps({"status": "valid", **counts, "scope": "structural checks only; not privacy certification"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
