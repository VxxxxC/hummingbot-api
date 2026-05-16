"""
Security regression tests for the four HIGH-severity vulnerability fixes:

1. Path traversal prevention in FileSystemUtil
2. Weak-credential warnings in SecuritySettings / config.py
3. debug_mode warning emitted at startup (main.py)
4. Plaintext password query params removed from WebSocket auth
"""

import os
import tempfile
import warnings

import pytest


# ---------------------------------------------------------------------------
# Fix 1 — Path traversal in FileSystemUtil
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """_safe_resolve() must block any path that escapes base_path."""

    def _make_util(self, tmp_dir: str):
        """Return a fresh FileSystemUtil singleton-cleared instance for testing."""
        from utils.file_system import FileSystemUtil
        # Reset singleton so each test gets its own instance with the right base_path
        FileSystemUtil._instance = None
        util = FileSystemUtil(base_path=tmp_dir)
        return util

    def test_normal_relative_path_allowed(self, tmp_path):
        """A plain relative path inside base_path must resolve without error."""
        util = self._make_util(str(tmp_path))
        sub = tmp_path / "subdir"
        sub.mkdir()
        result = util._safe_resolve("subdir")
        assert os.path.isdir(result)
        assert result.startswith(str(tmp_path))

    def test_dotdot_traversal_blocked(self, tmp_path):
        """../../ traversal must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util._safe_resolve("../../etc/passwd")

    def test_dotdot_in_middle_blocked(self, tmp_path):
        """Traversal embedded in path must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util._safe_resolve("subdir/../../etc/passwd")

    def test_absolute_path_outside_base_blocked(self, tmp_path):
        """An absolute path that lives outside base_path must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util._safe_resolve("/etc/passwd")

    def test_list_files_traversal_blocked(self, tmp_path):
        """list_files() with traversal must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.list_files("../../etc")

    def test_list_folders_traversal_blocked(self, tmp_path):
        """list_folders() with traversal must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.list_folders("../")

    def test_read_file_traversal_blocked(self, tmp_path):
        """read_file() with traversal must raise PermissionError."""
        # Write a file outside the base to make sure FS check fires before existence check
        target = tmp_path.parent / "secret.txt"
        target.write_text("secret")
        try:
            util = self._make_util(str(tmp_path))
            with pytest.raises(PermissionError):
                util.read_file("../secret.txt")
        finally:
            target.unlink(missing_ok=True)

    def test_read_yaml_file_traversal_blocked(self, tmp_path):
        """read_yaml_file() with a relative traversal must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.read_yaml_file("../../some.yml")

    def test_add_file_traversal_blocked(self, tmp_path):
        """add_file() with a traversal in directory must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.add_file("../../tmp", "evil.txt", "evil content")

    def test_delete_file_traversal_blocked(self, tmp_path):
        """delete_file() with traversal must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.delete_file("../../tmp", "file.txt")

    def test_delete_folder_traversal_blocked(self, tmp_path):
        """delete_folder() with traversal must raise PermissionError."""
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.delete_folder("../../", "tmp")

    def test_path_exists_traversal_returns_false(self, tmp_path):
        """path_exists() with a traversal must return False, not raise."""
        util = self._make_util(str(tmp_path))
        assert util.path_exists("../../etc") is False

    def test_normal_operations_still_work(self, tmp_path):
        """Ensure the fix does not break legitimate file operations."""
        util = self._make_util(str(tmp_path))

        # create a folder
        util.create_folder(".", "mydir")
        assert os.path.isdir(str(tmp_path / "mydir"))

        # add a file
        util.add_file("mydir", "test.txt", "hello")
        assert (tmp_path / "mydir" / "test.txt").read_text() == "hello"

        # read the file
        content = util.read_file("mydir/test.txt")
        assert content == "hello"

        # list files
        files = util.list_files("mydir")
        assert "test.txt" in files

        # delete the file
        util.delete_file("mydir", "test.txt")
        assert not (tmp_path / "mydir" / "test.txt").exists()


# ---------------------------------------------------------------------------
# Fix 2 — Weak credential warnings in SecuritySettings
# ---------------------------------------------------------------------------

class TestWeakCredentialWarnings:
    """SecuritySettings validators must emit warnings for weak defaults."""

    def test_default_password_warns(self):
        from config import SecuritySettings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SecuritySettings(username="admin", password="admin", config_password="a")
        messages = [str(w.message) for w in caught]
        assert any("password is weak" in m.lower() or "PASSWORD" in m for m in messages), \
            f"Expected weak-password warning; got: {messages}"

    def test_default_config_password_warns(self):
        from config import SecuritySettings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SecuritySettings(username="admin", password="admin", config_password="a")
        messages = [str(w.message) for w in caught]
        assert any("config_password" in m.lower() or "CONFIG_PASSWORD" in m for m in messages), \
            f"Expected config_password warning; got: {messages}"

    def test_strong_password_no_warning(self):
        from config import SecuritySettings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SecuritySettings(
                username="admin",
                password="Str0ng!Password#99",
                config_password="AnotherStr0ngPass!",
            )
        security_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and ("password" in str(w.message).lower() or "PASSWORD" in str(w.message))
        ]
        assert len(security_warnings) == 0, \
            f"Unexpected password warnings for strong creds: {[str(w.message) for w in security_warnings]}"

    def test_short_password_warns(self):
        from config import SecuritySettings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SecuritySettings(username="user", password="short", config_password="short")
        messages = [str(w.message) for w in caught]
        assert any("password" in m.lower() or "PASSWORD" in m for m in messages), \
            f"Expected warning for short password; got: {messages}"


# ---------------------------------------------------------------------------
# Fix 3 — debug_mode warning in main.py
# ---------------------------------------------------------------------------

class TestDebugModeWarning:
    """A WARNING must be logged when debug_mode is active."""

    def test_debug_mode_emits_log_warning(self, caplog):
        """Importing main with debug_mode=True must emit a WARNING log."""
        import importlib
        import sys

        # Temporarily force debug_mode to True via environment
        old = os.environ.get("DEBUG_MODE")
        os.environ["DEBUG_MODE"] = "true"
        try:
            # Remove cached modules to force re-evaluation of module-level code
            for mod in list(sys.modules.keys()):
                if mod in ("main", "config"):
                    del sys.modules[mod]

            import logging
            with caplog.at_level(logging.WARNING):
                try:
                    import main  # noqa: F401 — side-effects are what we test
                except Exception:
                    pass  # Startup may fail without full infra; the warning fires first
            assert any(
                "debug mode" in record.message.lower() or "DEBUG MODE" in record.message
                for record in caplog.records
            ), f"Expected debug-mode warning in logs; records={[r.message for r in caplog.records]}"
        finally:
            if old is None:
                os.environ.pop("DEBUG_MODE", None)
            else:
                os.environ["DEBUG_MODE"] = old
            # Restore modules
            for mod in list(sys.modules.keys()):
                if mod in ("main", "config"):
                    del sys.modules[mod]


# ---------------------------------------------------------------------------
# Fix 4 — Plaintext password query params removed from WebSocket auth
# ---------------------------------------------------------------------------

class TestWebSocketAuthNoBareQueryParams:
    """WebSocket _authenticate_websocket must reject ?username=&****** fallback."""

    def _make_mock_websocket(self, headers: dict, query_params: dict):
        """Create a minimal WebSocket stub."""
        from unittest.mock import MagicMock
        ws = MagicMock()
        ws.headers = headers
        ws.query_params = query_params
        return ws

    def test_bare_username_password_params_rejected(self):
        """
        When ONLY ?username=...&password=... is passed (no Authorization header,
        no token), authentication must FAIL (return False).
        """
        # Temporarily disable debug_mode so the real auth runs
        import config
        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", False)
        try:
            from routers.websocket import _authenticate_websocket
            ws = self._make_mock_websocket(
                headers={"authorization": ""},
                query_params={
                    "username": config.settings.security.username,
                    "password": config.settings.security.password,
                },
            )
            result = _authenticate_websocket(ws)
            assert result is False, (
                "Bare ?username=&password= query params must be rejected"
            )
        finally:
            object.__setattr__(config.settings.security, "debug_mode", original)

    def test_authorization_header_still_works(self):
        """Authorization: Basic header must still authenticate successfully."""
        import base64
        import config

        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", False)
        try:
            from routers.websocket import _authenticate_websocket
            creds = base64.b64encode(
                f"{config.settings.security.username}:{config.settings.security.password}".encode()
            ).decode()
            ws = self._make_mock_websocket(
                headers={"authorization": f"Basic {creds}"},
                query_params={},
            )
            result = _authenticate_websocket(ws)
            assert result is True
        finally:
            object.__setattr__(config.settings.security, "debug_mode", original)

    def test_token_query_param_still_works(self):
        """?token=base64(user:pass) must still authenticate successfully."""
        import base64
        import config

        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", False)
        try:
            from routers.websocket import _authenticate_websocket
            token = base64.b64encode(
                f"{config.settings.security.username}:{config.settings.security.password}".encode()
            ).decode()
            ws = self._make_mock_websocket(
                headers={"authorization": ""},
                query_params={"token": token},
            )
            result = _authenticate_websocket(ws)
            assert result is True
        finally:
            object.__setattr__(config.settings.security, "debug_mode", original)

    def test_no_credentials_rejected(self):
        """Empty request must be rejected."""
        import config

        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", False)
        try:
            from routers.websocket import _authenticate_websocket
            ws = self._make_mock_websocket(headers={"authorization": ""}, query_params={})
            assert _authenticate_websocket(ws) is False
        finally:
            object.__setattr__(config.settings.security, "debug_mode", original)

    def test_wrong_credentials_rejected(self):
        """Wrong password must be rejected."""
        import base64
        import config

        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", False)
        try:
            from routers.websocket import _authenticate_websocket
            creds = base64.b64encode(
                f"{config.settings.security.username}:wrongpassword".encode()
            ).decode()
            ws = self._make_mock_websocket(
                headers={"authorization": f"Basic {creds}"},
                query_params={},
            )
            assert _authenticate_websocket(ws) is False
        finally:
            object.__setattr__(config.settings.security, "debug_mode", original)

    def test_debug_mode_bypasses_auth(self):
        """debug_mode=True must return True regardless of credentials."""
        import config

        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", True)
        try:
            from routers.websocket import _authenticate_websocket
            ws = self._make_mock_websocket(headers={"authorization": ""}, query_params={})
            assert _authenticate_websocket(ws) is True
        finally:
            object.__setattr__(config.settings.security, "debug_mode", original)
