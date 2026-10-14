# Research maintenance

Python 3.12 standard library only. Run from the repository root.

## Structural validation

```powershell
python scripts\validate_research.py
python scripts\validate_research.py --root tests\fixture-repository
```

Checks the four `research` ledgers, stable-ID syntax/uniqueness, references,
conjunctive queue dependencies, completion consistency, existing file artifacts,
canonical repository-relative POSIX artifact paths, local Markdown links, daily
dates, and obvious publication hazards. IDs cannot be proved historically stable
by a single snapshot. Nullable unknown publication dates must remain null.
Empty `last_completed_question` is permitted only when nothing is done.
Daily filenames are exactly `YYYY-MM-DD.md`; the latest note, if present, must
match `progress.last_run_date`. Anchors are not validated.

`own-offline` claims require existing Python experiment and JSON/CSV result
artifacts beneath `experiments`. `own-hardware` requires artifacts named
`environment.*` and `raw-results*.json` or `raw-results*.csv`. This checks that
evidence files exist, not that their contents establish a valid benchmark.
Other evidence states require source references. Pending and blocked questions
are legitimate; this tool never requires the entire queue to be complete.

Content checks reject clear local private paths, obvious quoted credential
assignments, and bare credential assignments in `.env` files without printing
matched content. They are intentionally narrow,
not a secret scanner. Documentation discussing secrets is permitted. Passing
checks is **not semantic verification, security review, or privacy certification**.

## Explicit public-source monitoring

```powershell
python scripts\source_monitor.py
python scripts\source_monitor.py --only source-id another-source --timeout 15
```

This command makes real network requests when explicitly executed; tests do not.
It fetches only selected ledger URLs from a fixed HTTPS allowlist. No discovery,
crawling, authentication, query strings, userinfo, custom ports, proxy environment
variables, or private destination IPs are accepted. DNS answers are checked and
the TLS connection is pinned to a validated IP while retaining default certificate
and hostname verification. Redirects are validated again; at most five hops.
Each socket operation has a timeout (default 15 seconds, allowed up to 30);
this is not a single total wall-clock deadline. There are at most two transient
retries and delays at most two seconds. Bodies are bounded to 2 MiB.
DNS resolution also depends on the operating system resolver's own timeout.

No raw third-party body is stored. State stores SHA-256 and optional bounded
ETag/Last-Modified validators. A 304 without a previous successful baseline is an
error. Errors retain old baselines; partial success is persisted and exits nonzero.
Malformed input/state is an explicit failure, not an initial-run fallback.
**A changed hash is not a new product or technical research result.**

Default state: `research/source-monitor.json`:

```json
{
  "schema_version": 1,
  "sources": {
    "source-id": {
      "sha256": "64 lowercase hexadecimal characters",
      "etag": null,
      "last_modified": null,
      "checked_at": "ISO UTC date/time"
    }
  }
}
```

Default report: `research/monitor-latest.json`, with `schema_version`, `checked_at`,
`meaning`, `results`, `summary`, and `has_errors`. Each result has `source_id`,
`status` (`initial`, `unchanged`, `changed`, `error`), `checked_at`, `sha256`,
`http_status`, and `error_code`. Error hashes, when present, are the retained
previous successful hash. Summary counts all four statuses. Diagnostics use
fixed codes rather than URLs or raw exceptions. `--sources`, `--state`, and
`--output` can select distinct files; `--only` takes space-separated exact IDs.
Files are individually atomically replaced, not a transactional pair.

## Persistent shared-worktree lock

```powershell
python scripts\research_lock.py acquire --owner session-opaque-token
python scripts\research_lock.py status
python scripts\research_lock.py release --owner session-opaque-token
```

The lock is `ai-infra-research.lock` in `git rev-parse --git-common-dir`, shared
by worktrees and not tracked. `--repo` optionally selects another Git checkout.
Acquisition uses exclusive file creation. Ownership and UTC acquisition time
persist after the command exits. Existing, malformed, or unreadable locks fail
closed. Owner tokens accept only letters/digits/underscore/hyphen, up to 128
characters; no paths or email addresses. Status prints only safe ownership.
Release must match the stored token; repeated acquisition or release fails.

**Never steal on age or PID.** First verify that the owner is no longer active
using the application's live session status. Only then explicitly release with
that exact owner token. A malformed lock needs deliberate human repair, not
automatic recovery. This is an advisory coordination protocol for cooperating
agents, not access control against concurrent uncooperative filesystem writers.

## Offline tests

```powershell
python -m unittest discover -s tests
python -m experiments.capacity_roofline.run --check
python -m experiments.ring_cut.run --check
```

Tests use original synthetic fixtures under `tests` and mocked transports;
they do not call APIs, download code, run models, or need third-party packages.

The original Ring cut experiment accounts for one ideal Ring's directed payload
across synthetic node cuts and a shared send-side NIC ceiling. It does not
simulate NCCL graph search, measure network timings, or convert NIC payload
into GPU/HBM traffic. See its [inputs and limits](../experiments/ring_cut/README.md).
