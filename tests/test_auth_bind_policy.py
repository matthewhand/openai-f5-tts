"""Synthetic startup bind-policy tests: no Flask, sockets, DNS or models."""
import ast
import os
from unittest import mock
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from auth_bind_policy import validate_auth_bind  # fails red until policy exists


class AuthBindPolicyTests(unittest.TestCase):
    def test_unauthenticated_loopback_ip_addresses_are_allowed(self):
        for host in ("127.0.0.1", "127.9.2.1", "::1"):
            with self.subTest(host=host):
                self.assertIsNone(validate_auth_bind(host, False))

    def test_unauthenticated_wildcard_and_lan_are_rejected(self):
        for host in ("0.0.0.0", "::", "192.168.1.20", "10.0.0.30",
                     "172.16.0.3", "8.8.8.8", "fe80::1", "2001:db8::10"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                validate_auth_bind(host, False)

    def test_unauthenticated_dns_names_and_ambiguous_hosts_fail_closed(self):
        for host in ("localhost", "LOCALHOST", "localhost.localdomain", "",
                     "127.0.0.1 ", "[::1]", "::ffff:127.0.0.1", "bad name", None):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                validate_auth_bind(host, False)

    def test_authenticated_mode_keeps_existing_host_behavior(self):
        for host in ("0.0.0.0", "::", "localhost", "192.168.1.20", "127.0.0.1"):
            with self.subTest(host=host):
                self.assertIsNone(validate_auth_bind(host, True))

    def test_auth_boolean_parser_rejects_typos_instead_of_disabling_auth(self):
        module = ast.parse((ROOT / "app" / "utils.py").read_text())
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "getenv_bool")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app/utils.py", "exec"), namespace)
        parse = namespace["getenv_bool"]
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(parse("REQUIRE_API_KEY", True))
            self.assertFalse(parse("REQUIRE_API_KEY", False))
        for value in ("true", "T", "yes", "Y", "1", " true "):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"REQUIRE_API_KEY": value}):
                self.assertTrue(parse("REQUIRE_API_KEY", True))
        for value in ("false", "F", "no", "N", "0", " false "):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"REQUIRE_API_KEY": value}):
                self.assertFalse(parse("REQUIRE_API_KEY", True))
        for value in ("tru", "enabled", "2", "", " ", "off", "yes please"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"REQUIRE_API_KEY": value}):
                with self.assertRaisesRegex(ValueError, "Invalid boolean configuration") as exc:
                    parse("REQUIRE_API_KEY", True)
                self.assertNotIn(value.strip(), str(exc.exception)) if value.strip() else self.assertNotIn("[secret]", str(exc.exception))

    def test_server_parses_auth_before_importing_heavy_tts_dependencies(self):
        module = ast.parse((ROOT / "app" / "server.py").read_text())
        indices = {node.module: index for index, node in enumerate(module.body)
                   if isinstance(node, ast.ImportFrom)}
        self.assertLess(indices["utils"], indices["tts_handler"])

    def test_server_checks_bind_before_model_handler_is_constructed(self):
        module = ast.parse((ROOT / "app" / "server.py").read_text(encoding="utf-8"))
        calls = []
        for index, node in enumerate(module.body):
            if isinstance(node, (ast.Assign, ast.Expr)):
                calls.append((index, ast.dump(node)))
        validation = [i for i, val in calls if "validate_auth_bind" in val]
        handler = [i for i, val in calls if "TTSHandler" in val]
        self.assertEqual(len(validation), 1)
        self.assertEqual(len(handler), 1)
        self.assertLess(validation[0], handler[0])


if __name__ == "__main__":
    unittest.main()
