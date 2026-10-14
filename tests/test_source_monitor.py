from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import socket
import ssl
from unittest.mock import Mock, patch

from scripts import source_monitor as monitor
from scripts.validate_research import ValidationError
from tests.fixture_support import FixtureCase


SOURCE = {"id": "src-one", "url": "https://docs.kernel.org/doc.html"}


class MonitorTests(FixtureCase):
    def setUp(self) -> None:
        super().setUp()
        self.transport = self.enterContext(patch.object(monitor, "request_once"))
        self.sleep = self.enterContext(patch.object(monitor.time, "sleep"))

    def initial(self, body: bytes = b"original fixture") -> tuple[dict, dict]:
        self.transport.return_value = monitor.Response(200, {"etag": '"original"'}, body)
        return monitor.monitor([SOURCE], {})

    def test_200_initial_same_and_changed(self) -> None:
        report, state = self.initial()
        self.assertEqual(report["results"][0]["status"], "initial")
        self.assertEqual(state["src-one"]["sha256"], hashlib.sha256(b"original fixture").hexdigest())
        report, state = monitor.monitor([SOURCE], state)
        self.assertEqual(report["results"][0]["status"], "unchanged")
        self.assertEqual(self.transport.call_args.args[1]["If-None-Match"], '"original"')
        self.transport.return_value = monitor.Response(200, {}, b"new original fixture")
        report, state = monitor.monitor([SOURCE], state)
        self.assertEqual(report["results"][0]["status"], "changed")
        self.assertIsNone(state["src-one"]["etag"])
        self.assertNotIn("new original fixture", json.dumps(report))
        self.assertNotIn("body", state["src-one"])
        self.assertIn("not new products", report["meaning"])

    def test_304_previous_baseline_and_no_baseline(self) -> None:
        _, state = self.initial()
        self.transport.return_value = monitor.Response(304, {})
        report, updated = monitor.monitor([SOURCE], state)
        self.assertEqual(report["results"][0]["status"], "unchanged")
        self.assertEqual(updated["src-one"]["sha256"], state["src-one"]["sha256"])
        self.assertEqual(updated["src-one"]["etag"], '"original"')
        report, updated = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["error_code"], "not_modified_without_baseline")
        self.assertEqual(updated, {})

    def test_404_preserves_baseline(self) -> None:
        _, state = self.initial()
        self.transport.reset_mock()
        self.transport.return_value = monitor.Response(404, {})
        report, updated = monitor.monitor([SOURCE], state)
        self.assertEqual(report["results"][0]["error_code"], "http_404")
        self.assertEqual(report["results"][0]["http_status"], 404)
        self.assertEqual(updated, state)
        self.transport.assert_called_once()
        self.sleep.assert_not_called()

    def test_429_then_success_with_bounded_retry_after(self) -> None:
        self.transport.side_effect = [
            monitor.Response(429, {"retry-after": "9999"}),
            monitor.Response(200, {}, b"original fixture"),
        ]
        report, state = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["status"], "initial")
        self.assertIn("src-one", state)
        self.assertEqual(self.transport.call_count, 2)
        self.sleep.assert_called_once_with(2.0)

    def test_retry_limit_for_429_and_5xx(self) -> None:
        for status in (429, 500, 503):
            with self.subTest(status=status):
                self.transport.reset_mock()
                self.sleep.reset_mock()
                self.transport.return_value = monitor.Response(status, {})
                report, state = monitor.monitor([SOURCE], {})
                self.assertEqual(report["results"][0]["error_code"], f"http_{status}")
                self.assertEqual(self.transport.call_count, 3)
                self.assertEqual(self.sleep.call_count, 2)
                self.assertEqual(state, {})

    def test_timeout_then_success_and_exhausted_timeout(self) -> None:
        self.transport.side_effect = [TimeoutError("private diagnostic"), monitor.Response(200, {}, b"fixture")]
        report, _ = monitor.monitor([SOURCE], {})
        self.assertFalse(report["has_errors"])
        self.transport.reset_mock()
        self.transport.side_effect = TimeoutError("private diagnostic")
        report, state = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["error_code"], "timeout")
        self.assertEqual(self.transport.call_count, 3)
        self.assertEqual(state, {})
        self.assertNotIn("private diagnostic", json.dumps(report))

    def test_tls_and_network_errors_are_redacted_without_retry(self) -> None:
        for exception, code in ((ssl.SSLError("private diagnostic"), "tls_error"), (OSError("private diagnostic"), "network_error")):
            self.transport.reset_mock()
            self.transport.side_effect = exception
            report, _ = monitor.monitor([SOURCE], {})
            self.assertEqual(report["results"][0]["error_code"], code)
            self.assertNotIn("private diagnostic", json.dumps(report))
            self.assertEqual(self.transport.call_count, 1)

    def test_partial_failure_retains_success_and_old_baselines(self) -> None:
        _, old = self.initial()
        second = {"id": "src-two", "url": "https://docs.nvidia.com/"}
        self.transport.side_effect = [monitor.Response(503, {}), monitor.Response(503, {}), monitor.Response(503, {}), monitor.Response(200, {}, b"new fixture")]
        report, updated = monitor.monitor([SOURCE, second], old)
        self.assertTrue(report["has_errors"])
        self.assertEqual(updated["src-one"], old["src-one"])
        self.assertIn("src-two", updated)
        self.assertEqual(report["summary"], {"initial": 1, "unchanged": 0, "changed": 0, "error": 1})

    def test_malicious_redirects_rejected_before_second_request(self) -> None:
        for target in (
            "https://evil.example.org/", "http://docs.kernel.org/", "https://127.0.0.1/",
            "https://docs.nvidia.com/path?private=opaque", "https://person@github.com/path",
            "https://github.com:8443/path",
        ):
            with self.subTest(target=target):
                self.transport.reset_mock()
                self.transport.return_value = monitor.Response(302, {"location": target})
                report, _ = monitor.monitor([SOURCE], {})
                self.assertEqual(report["results"][0]["error_code"], "unsafe_redirect")
                self.transport.assert_called_once()
                self.assertNotIn(target, json.dumps(report))

    def test_safe_relative_redirect_and_hop_limit(self) -> None:
        self.transport.side_effect = [
            monitor.Response(302, {"location": "/new.html#fragment"}),
            monitor.Response(200, {}, b"fixture"),
        ]
        report, _ = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["status"], "initial")
        self.assertEqual(self.transport.call_args.args[0], "https://docs.kernel.org/new.html")
        self.transport.reset_mock()
        self.transport.side_effect = None
        self.transport.return_value = monitor.Response(301, {"location": "/loop"})
        report, _ = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["error_code"], "too_many_redirects")
        self.assertEqual(self.transport.call_count, 6)

    def test_missing_location_and_invalid_metadata(self) -> None:
        self.transport.return_value = monitor.Response(302, {})
        report, _ = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["error_code"], "redirect_without_location")
        self.transport.return_value = monitor.Response(200, {"etag": "bad\r\nheader"}, b"fixture")
        report, state = monitor.monitor([SOURCE], {})
        self.assertEqual(report["results"][0]["error_code"], "invalid_conditional_metadata")
        self.assertEqual(state, {})

    def test_source_rejected_without_transport(self) -> None:
        report, state = monitor.monitor([{"id": "src-one", "url": "https://unapproved.example/"}], {})
        self.assertEqual(report["results"][0]["error_code"], "unsafe_url")
        self.transport.assert_not_called()
        self.assertEqual(state, {})

    def test_cli_persists_partial_success_and_returns_nonzero(self) -> None:
        sources = self.write_json("sources.json", {
            "schema_version": 1, "sources": [SOURCE, {"id": "src-two", "url": "https://docs.nvidia.com/"}],
        })
        state = self.root / "state.json"
        output = self.root / "report.json"
        self.transport.side_effect = [monitor.Response(200, {}, b"fixture"), monitor.Response(404, {})]
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            code = monitor.main(["--sources", str(sources), "--state", str(state), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertIn("src-one", json.loads(state.read_text())["sources"])
        self.assertTrue(json.loads(output.read_text())["has_errors"])
        self.assertEqual(json.loads(stdout.getvalue())["error"], 1)
        self.assertNotIn("https://", stderr.getvalue())
        self.assertEqual(list(self.root.glob("*.writing-*")), [])

    def test_cli_only_preserves_unselected_and_no_source_access_mutation(self) -> None:
        first = {**SOURCE, "access_status": "blocked"}
        second = {"id": "src-two", "url": "https://docs.nvidia.com/"}
        sources = self.write_json("sources.json", {"schema_version": 1, "sources": [first, second]})
        _, prior = self.initial()
        state_path = self.write_json("state.json", {"schema_version": 1, "sources": prior})
        original_sources = sources.read_bytes()
        self.transport.reset_mock()
        self.transport.return_value = monitor.Response(200, {}, b"another fixture")
        with redirect_stdout(io.StringIO()):
            code = monitor.main([
                "--sources", str(sources), "--state", str(state_path),
                "--output", str(self.root / "report.json"), "--only", "src-two",
            ])
        self.assertEqual(code, 0)
        self.transport.assert_called_once()
        self.assertEqual(sources.read_bytes(), original_sources)
        self.assertEqual(json.loads(state_path.read_text())["sources"]["src-one"], prior["src-one"])

    def test_cli_rejects_bad_inputs_without_network_or_baseline_reset(self) -> None:
        sources = self.write_json("sources.json", {"schema_version": 1, "sources": [SOURCE]})
        state = self.write_json("state.json", {"schema_version": 1, "sources": {"src-one": {"sha256": "bad"}}})
        original = state.read_bytes()
        base = ["--sources", str(sources), "--state", str(state), "--output", str(self.root / "report.json")]
        for tail in ([], ["--only", "missing"], ["--timeout", "nan"], ["--timeout", "0"], ["--timeout", "31"]):
            with redirect_stderr(io.StringIO()):
                self.assertEqual(monitor.main(base + tail), 1)
        self.transport.assert_not_called()
        self.assertEqual(state.read_bytes(), original)
        self.assertFalse((self.root / "report.json").exists())

    def test_last_modified_sent_and_validated_state(self) -> None:
        self.transport.return_value = monitor.Response(200, {"last-modified": "Mon, 12 Oct 2026 00:00:00 GMT"}, b"fixture")
        _, state = monitor.monitor([SOURCE], {})
        path = self.write_json("state.json", {"schema_version": 1, "sources": state})
        self.assertEqual(monitor.load_state(path), state)
        monitor.monitor([SOURCE], state)
        self.assertIn("If-Modified-Since", self.transport.call_args.args[1])
        self.write("state.json", "not JSON")
        with self.assertRaises(ValidationError):
            monitor.load_state(path)


class TransportTests(FixtureCase):
    def test_safe_url_contract(self) -> None:
        self.assertEqual(monitor.safe_url("https://github.com/org/repo#fragment"), "https://github.com/org/repo")
        for url in (
            "https://docs.kernel.org/?", "https://docs.kernel.org/?a=b", "https://docs.kernel.org.evil.org/",
            "https://user:opaque@docs.kernel.org/", "file:///fixture", "https://localhost/",
            "https://docs.kernel.org/\nheader", "https://docs.kernel.org:8443/", "https://docs.kernel.org/é",
        ):
            with self.subTest(url=url), self.assertRaises(monitor.FetchError):
                monitor.safe_url(url)

    def test_dns_private_and_mixed_answers(self) -> None:
        def answer(address: str) -> tuple:
            return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))

        for answers in ([answer("127.0.0.1")], [answer("10.1.2.3")], [answer("1.1.1.1"), answer("192.168.1.1")], [answer("224.0.0.1")]):
            with patch.object(monitor.socket, "getaddrinfo", return_value=answers):
                with self.assertRaisesRegex(monitor.FetchError, "private_destination"):
                    monitor.public_addresses("docs.kernel.org")
        with patch.object(monitor.socket, "getaddrinfo", return_value=[answer("1.1.1.1")]):
            self.assertEqual(monitor.public_addresses("docs.kernel.org"), ["1.1.1.1"])
        with patch.object(monitor.socket, "getaddrinfo", side_effect=socket.gaierror("private diagnostic")):
            with self.assertRaisesRegex(monitor.FetchError, "dns_error"):
                monitor.public_addresses("docs.kernel.org")

    def response(self, *, body: bytes, length: str | None = None) -> Mock:
        response = Mock(status=200)
        response.getheaders.return_value = [] if length is None else [("Content-Length", length)]
        response.read.return_value = body
        return response

    def test_body_size_bound_and_incomplete_body(self) -> None:
        for body, length, expected in (
            (b"", str(monitor.MAX_BODY_BYTES + 1), "body_too_large"),
            (b"x" * (monitor.MAX_BODY_BYTES + 1), None, "body_too_large"),
            (b"x", "2", "incomplete_body"), (b"x", "-1", "invalid_content_length"),
            (b"x", "garbage", "invalid_content_length"),
        ):
            response = self.response(body=body, length=length)
            connection = Mock()
            connection.getresponse.return_value = response
            with patch.object(monitor, "public_addresses", return_value=["1.1.1.1"]), patch.object(monitor, "PinnedHTTPSConnection", return_value=connection):
                with self.assertRaisesRegex(monitor.FetchError, expected):
                    monitor.request_once(SOURCE["url"], {}, 15)
            connection.close.assert_called_once()
            if length is None:
                response.read.assert_called_once_with(monitor.MAX_BODY_BYTES + 1)

    def test_exact_limit_body_is_accepted(self) -> None:
        body = b"x" * monitor.MAX_BODY_BYTES
        response = self.response(body=body, length=str(len(body)))
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(monitor, "public_addresses", return_value=["1.1.1.1"]), patch.object(monitor, "PinnedHTTPSConnection", return_value=connection):
            result = monitor.request_once(SOURCE["url"], {}, 15)
        self.assertEqual(result.body, body)
        connection.close.assert_called_once()

    def test_default_tls_and_hostname_preserved_for_pinned_ip(self) -> None:
        context = Mock()
        sock = Mock()
        with patch.object(monitor.ssl, "create_default_context", return_value=context), patch.object(monitor.socket, "create_connection", return_value=sock) as connect:
            connection = monitor.PinnedHTTPSConnection("docs.kernel.org", "1.1.1.1", 15)
            connection.connect()
        connect.assert_called_once_with(("1.1.1.1", 443), timeout=15)
        context.wrap_socket.assert_called_once_with(sock, server_hostname="docs.kernel.org")
