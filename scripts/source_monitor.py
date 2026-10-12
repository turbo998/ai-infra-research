"""Bounded explicit-source HTTPS monitor; body changes are not technical discoveries."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import ssl
import sys
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from uuid import uuid4

if __package__:
    from .validate_research import ID_PATTERN, ValidationError, iso_timestamp, read_json
else:
    from validate_research import ID_PATTERN, ValidationError, iso_timestamp, read_json


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HOSTS = frozenset({
    "docs.nvidia.com", "rocm.docs.amd.com", "ultraethernet.org", "www.jedec.org",
    "www.nvidia.com", "developer.nvidia.com", "docs.vllm.ai", "docs.pytorch.org",
    "pytorch.org", "github.com", "raw.githubusercontent.com", "arxiv.org",
    "docs.kernel.org", "kubernetes.io", "slurm.schedmd.com",
})
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_RETRIES = 2
MAX_REDIRECTS = 5
MAX_DELAY_SECONDS = 2.0
MEANING = "Body/hash changes are monitoring signals, not new products or technical results."


class FetchError(ValueError):
    """A stable redacted fetch error suitable for public reports."""

    def __init__(self, code: str, http_status: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class Response:
    """Bounded response metadata and an ephemeral body never persisted to disk."""

    status: int
    headers: dict[str, str]
    body: bytes = b""


def safe_url(url: str) -> str:
    """Require an exact approved HTTPS host without queries, userinfo, or custom ports."""
    if (
        not isinstance(url, str) or len(url) > 8192 or not url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        raise FetchError("unsafe_url")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise FetchError("unsafe_url") from exc
    if (
        parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username is not None or parsed.password is not None
        or parsed.query or "?" in url.split("#", 1)[0] or port not in {None, 443}
        or "\\" in url
    ):
        raise FetchError("unsafe_url")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


def public_addresses(host: str) -> list[str]:
    """Resolve and reject any non-public address, including mixed DNS answers."""
    try:
        answers = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError("dns_error") from exc
    addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses:
        raise FetchError("dns_error")
    for address in addresses:
        parsed = ipaddress.ip_address(address)
        if not parsed.is_global or parsed.is_multicast:
            raise FetchError("private_destination")
    return addresses


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Keep TLS hostname verification while connecting to a prevalidated IP."""

    def __init__(self, host: str, address: str, timeout: float) -> None:
        self._address = address
        self._tls_context = ssl.create_default_context()
        super().__init__(host, port=443, timeout=timeout, context=self._tls_context)

    def connect(self) -> None:
        """Connect directly, without proxy environment variables or DNS rebinding."""
        sock = socket.create_connection((self._address, 443), timeout=self.timeout)
        try:
            self.sock = self._tls_context.wrap_socket(sock, server_hostname=self.host)
        except (OSError, ValueError):
            sock.close()
            raise


def request_once(url: str, headers: dict[str, str], timeout: float) -> Response:
    """Perform one direct TLS request with a bounded body and prevalidated destination."""
    url = safe_url(url)
    parsed = urlsplit(url)
    host = parsed.hostname
    if host is None:
        raise FetchError("unsafe_url")
    addresses = public_addresses(host)
    connection = PinnedHTTPSConnection(host, addresses[0], timeout)
    try:
        connection.request("GET", parsed.path or "/", headers=headers)
        response = connection.getresponse()
        metadata = {key.lower(): value for key, value in response.getheaders()}
        body = b""
        if response.status == 200:
            length = metadata.get("content-length")
            if length is not None:
                try:
                    declared_size = int(length)
                except ValueError as exc:
                    raise FetchError("invalid_content_length", response.status) from exc
                if declared_size < 0:
                    raise FetchError("invalid_content_length", response.status)
                if declared_size > MAX_BODY_BYTES:
                    raise FetchError("body_too_large", response.status)
            body = response.read(MAX_BODY_BYTES + 1)
            if len(body) > MAX_BODY_BYTES:
                raise FetchError("body_too_large", response.status)
            if length is not None and len(body) != declared_size:
                raise FetchError("incomplete_body", response.status)
        return Response(response.status, metadata, body)
    finally:
        connection.close()


def fetch(url: str, headers: dict[str, str], timeout: float) -> Response:
    """Fetch with approved redirects and at most two bounded transient retries."""
    original = safe_url(url)
    for attempt in range(MAX_RETRIES + 1):
        current = original
        try:
            for hop in range(MAX_REDIRECTS + 1):
                response = request_once(current, headers, timeout)
                if response.status not in {301, 302, 303, 307, 308}:
                    break
                if hop == MAX_REDIRECTS:
                    raise FetchError("too_many_redirects", response.status)
                location = response.headers.get("location")
                if not location:
                    raise FetchError("redirect_without_location", response.status)
                try:
                    current = safe_url(urljoin(current, location))
                except FetchError as exc:
                    raise FetchError("unsafe_redirect", response.status) from exc
            if response.status == 429 or 500 <= response.status <= 599:
                if attempt == MAX_RETRIES:
                    raise FetchError(f"http_{response.status}", response.status)
                retry_after = response.headers.get("retry-after", "")
                delay = (
                    min(float(retry_after), MAX_DELAY_SECONDS)
                    if re.fullmatch(r"[0-9]{1,10}", retry_after)
                    else 0.25 * (attempt + 1)
                )
            elif response.status in {200, 304}:
                return response
            else:
                raise FetchError(f"http_{response.status}", response.status)
        except (TimeoutError, socket.timeout) as exc:
            if attempt == MAX_RETRIES:
                raise FetchError("timeout") from exc
            delay = 0.25 * (attempt + 1)
        except ssl.SSLError as exc:
            raise FetchError("tls_error") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise FetchError("network_error") from exc
        time.sleep(delay)
    raise FetchError("retry_limit")


def header_value(value: object) -> str | None:
    """Bound stored conditional validators and reject header control characters."""
    if value is None:
        return None
    if (
        not isinstance(value, str) or not value or len(value) > 1024
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValidationError("invalid conditional metadata")
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValidationError("invalid conditional metadata") from exc
    return value


def load_state(path: Path) -> dict[str, dict]:
    """Load validated successful baselines; absence alone means an initial run."""
    if not path.exists():
        return {}
    value = read_json(path)
    if (
        not isinstance(value, dict) or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1 or not isinstance(value.get("sources"), dict)
    ):
        raise ValidationError("invalid monitor state schema")
    for identifier, entry in value["sources"].items():
        if not ID_PATTERN.fullmatch(identifier) or not isinstance(entry, dict):
            raise ValidationError("invalid monitor state entry")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValidationError("invalid monitor state hash")
        header_value(entry.get("etag"))
        header_value(entry.get("last_modified"))
        iso_timestamp(entry.get("checked_at"), "state.checked_at")
    return value["sources"]


def monitor(
    sources: list[dict], state: dict[str, dict], *, timeout: float = 15.0,
) -> tuple[dict, dict[str, dict]]:
    """Collect independent results, retaining successful and previous baselines on errors."""
    checked_at = datetime.now(timezone.utc).isoformat()
    updated = {identifier: dict(entry) for identifier, entry in state.items()}
    results: list[dict] = []
    for source in sources:
        identifier = source["id"]
        previous = state.get(identifier)
        headers = {"User-Agent": "public-ai-infra-source-monitor/1", "Accept-Encoding": "identity"}
        if previous and previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if previous and previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
        record = {
            "source_id": identifier, "status": "error", "checked_at": checked_at,
            "sha256": previous["sha256"] if previous else None,
            "http_status": None, "error_code": None,
        }
        try:
            response = fetch(source["url"], headers, timeout)
            record["http_status"] = response.status
            if response.status == 304:
                if previous is None:
                    raise FetchError("not_modified_without_baseline", 304)
                digest = previous["sha256"]
                result_status = "unchanged"
            else:
                digest = hashlib.sha256(response.body).hexdigest()
                result_status = "initial" if previous is None else (
                    "unchanged" if digest == previous["sha256"] else "changed"
                )
            try:
                etag = header_value(response.headers.get("etag"))
                last_modified = header_value(response.headers.get("last-modified"))
            except ValidationError as exc:
                raise FetchError("invalid_conditional_metadata", response.status) from exc
            if response.status == 304 and previous:
                etag = etag or previous.get("etag")
                last_modified = last_modified or previous.get("last_modified")
            updated[identifier] = {
                "sha256": digest, "etag": etag, "last_modified": last_modified,
                "checked_at": checked_at,
            }
            record.update(status=result_status, sha256=digest)
        except FetchError as exc:
            record.update(error_code=exc.code, http_status=exc.http_status)
        results.append(record)
    summary = {status: sum(result["status"] == status for result in results)
               for status in ("initial", "unchanged", "changed", "error")}
    report = {
        "schema_version": 1, "checked_at": checked_at, "meaning": MEANING,
        "results": results, "summary": summary, "has_errors": bool(summary["error"]),
    }
    return report, updated


def write_json(path: Path, value: dict) -> None:
    """Atomically replace one JSON file without persisting fetched response bodies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.writing-{uuid4().hex}")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Read explicit sources, publish a bounded report, and fail on partial errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=REPO_ROOT / "research" / "sources.json")
    parser.add_argument("--state", type=Path, default=REPO_ROOT / "research" / "source-monitor.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "research" / "monitor-latest.json")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
            raise ValidationError("timeout must be finite and in (0, 30] seconds")
        if len({path.resolve() for path in (args.sources, args.state, args.output)}) != 3:
            raise ValidationError("sources, state, and output must be distinct files")
        value = read_json(args.sources)
        if (
            not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or not isinstance(value.get("sources"), list)
        ):
            raise ValidationError("invalid sources schema")
        sources: dict[str, dict] = {}
        for source in value["sources"]:
            if (
                not isinstance(source, dict) or not isinstance(source.get("id"), str)
                or not ID_PATTERN.fullmatch(source["id"]) or source["id"] in sources
                or not isinstance(source.get("url"), str)
            ):
                raise ValidationError("invalid or duplicate source entry")
            sources[source["id"]] = source
        if args.only and (len(set(args.only)) != len(args.only) or any(item not in sources for item in args.only)):
            raise ValidationError("unknown or duplicate source selection")
        selected = [source for identifier, source in sources.items() if not args.only or identifier in args.only]
        report, updated = monitor(selected, load_state(args.state), timeout=args.timeout)
        write_json(args.state, {"schema_version": 1, "sources": updated})
        write_json(args.output, report)
    except ValidationError as exc:
        print(f"Source monitor failed: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("Source monitor failed: file_io_error", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], sort_keys=True))
    if report["has_errors"]:
        print("Source monitor failed: one or more sources; see redacted error_code fields", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
