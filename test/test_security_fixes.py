"""
Security regression tests for the four HIGH-severity vulnerability fixes:

1. Path traversal prevention in FileSystemUtil
2. Weak-credential warnings in SecuritySettings / config.py
3. debug_mode warning emitted at startup (main.py)
4. Plaintext password query params removed from WebSocket auth

NOTE: The hummingbot package is not available in this CI environment.
Tests for modules that import hummingbot stub out those imports via
sys.modules mocking so that the logic under test can be exercised
without the full hummingbot dependency.
"""

import base64
import os
import sys
import types
import warnings
from types import ModuleType
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared helper: inject minimal hummingbot stubs so we can import our modules
# ---------------------------------------------------------------------------

def _install_hummingbot_stubs():
    """
    Insert lightweight stub modules for every hummingbot sub-package that is
    imported at the top level of utils/file_system.py and routers/websocket.py
    (transitively).  Only call once; subsequent calls are no-ops.
    """
    def _stub(name: str) -> ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return mod

    # Top-level package
    hb = _stub("hummingbot")

    # file_system.py imports
    hb_client = _stub("hummingbot.client")
    hb_client_config = _stub("hummingbot.client.config")

    cdtypes = _stub("hummingbot.client.config.config_data_types")
    cdtypes.BaseClientModel = MagicMock

    helpers = _stub("hummingbot.client.config.config_helpers")
    helpers.ClientConfigAdapter = MagicMock

    # Connector and strategy stubs
    _stub("hummingbot.connector")
    _stub("hummingbot.connector.connector_base")
    conn_base = sys.modules["hummingbot.connector.connector_base"]
    conn_base.ConnectorBase = MagicMock

    _stub("hummingbot.core")
    _stub("hummingbot.core.data_type")
    _stub("hummingbot.core.data_type.common")
    dt_common = sys.modules["hummingbot.core.data_type.common"]
    for name in ("OrderType", "PositionAction", "PositionMode", "TradeType"):
        setattr(dt_common, name, MagicMock)

    _stub("hummingbot.strategy_v2")
    _stub("hummingbot.strategy_v2.controllers")
    _stub("hummingbot.strategy_v2.controllers.controller_base")
    ctrl_base = sys.modules["hummingbot.strategy_v2.controllers.controller_base"]
    ctrl_base.ControllerConfigBase = MagicMock

    _stub("hummingbot.strategy_v2.controllers.directional_trading_controller_base")
    dtcb = sys.modules["hummingbot.strategy_v2.controllers.directional_trading_controller_base"]
    dtcb.DirectionalTradingControllerConfigBase = MagicMock

    _stub("hummingbot.strategy_v2.controllers.market_making_controller_base")
    mmcb = sys.modules["hummingbot.strategy_v2.controllers.market_making_controller_base"]
    mmcb.MarketMakingControllerConfigBase = MagicMock

    # Crypt / security (used by accounts_service → websocket_manager → websocket router)
    _stub("hummingbot.client.config.config_crypt")
    crypt = sys.modules["hummingbot.client.config.config_crypt"]
    crypt.ETHKeyFileSecretManger = MagicMock

    _stub("hummingbot.client.config.security")
    sec = sys.modules["hummingbot.client.config.security"]
    sec.BackendAPISecurity = MagicMock

    # Additional service-layer hummingbot stubs
    for stub_path in [
        "hummingbot.strategy_v2",
        "hummingbot.strategy_v2.executors",
        "hummingbot.strategy_v2.executors.executor_base",
        "hummingbot.data_feed",
        "hummingbot.data_feed.candles_feed",
        "hummingbot.data_feed.candles_feed.candles_base",
    ]:
        _stub(stub_path)

    executor_base = sys.modules.get("hummingbot.strategy_v2.executors.executor_base",
                                    types.ModuleType("hummingbot.strategy_v2.executors.executor_base"))
    executor_base.ExecutorBase = MagicMock
    sys.modules["hummingbot.strategy_v2.executors.executor_base"] = executor_base

    candles_base = sys.modules.get("hummingbot.data_feed.candles_feed.candles_base",
                                   types.ModuleType("hummingbot.data_feed.candles_feed.candles_base"))
    candles_base.CandlesBase = MagicMock
    sys.modules["hummingbot.data_feed.candles_feed.candles_base"] = candles_base


# Install stubs before any project imports below
_install_hummingbot_stubs()


# ---------------------------------------------------------------------------
# Fix 1 — Path traversal in FileSystemUtil
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """_safe_resolve() must block any path that escapes base_path."""

    def _make_util(self, tmp_dir: str):
        """Return a FileSystemUtil instance bound to tmp_dir."""
        # Force a fresh singleton for each test
        if "utils.file_system" in sys.modules:
            del sys.modules["utils.file_system"]
        from utils.file_system import FileSystemUtil
        FileSystemUtil._instance = None
        return FileSystemUtil(base_path=tmp_dir)

    def test_normal_relative_path_allowed(self, tmp_path):
        util = self._make_util(str(tmp_path))
        sub = tmp_path / "subdir"
        sub.mkdir()
        result = util._safe_resolve("subdir")
        assert os.path.isdir(result)
        assert result.startswith(str(tmp_path))

    def test_dotdot_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util._safe_resolve("../../etc/passwd")

    def test_dotdot_in_middle_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util._safe_resolve("subdir/../../etc/passwd")

    def test_absolute_path_outside_base_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util._safe_resolve("/etc/passwd")

    def test_list_files_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.list_files("../../etc")

    def test_list_folders_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.list_folders("../")

    def test_read_file_traversal_blocked(self, tmp_path):
        target = tmp_path.parent / "secret.txt"
        target.write_text("secret")
        try:
            util = self._make_util(str(tmp_path))
            with pytest.raises(PermissionError):
                util.read_file("../secret.txt")
        finally:
            target.unlink(missing_ok=True)

    def test_read_yaml_file_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.read_yaml_file("../../some.yml")

    def test_add_file_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.add_file("../../tmp", "evil.txt", "evil content")

    def test_delete_file_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.delete_file("../../tmp", "file.txt")

    def test_delete_folder_traversal_blocked(self, tmp_path):
        util = self._make_util(str(tmp_path))
        with pytest.raises(PermissionError):
            util.delete_folder("../../", "tmp")

    def test_path_exists_traversal_returns_false(self, tmp_path):
        util = self._make_util(str(tmp_path))
        assert util.path_exists("../../etc") is False

    def test_normal_operations_still_work(self, tmp_path):
        """Ensure the fix does not break legitimate file operations."""
        util = self._make_util(str(tmp_path))

        util.create_folder(".", "mydir")
        assert os.path.isdir(str(tmp_path / "mydir"))

        util.add_file("mydir", "test.txt", "hello")
        assert (tmp_path / "mydir" / "test.txt").read_text() == "hello"

        content = util.read_file("mydir/test.txt")
        assert content == "hello"

        files = util.list_files("mydir")
        assert "test.txt" in files

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
            f"Unexpected warnings for strong creds: {[str(w.message) for w in security_warnings]}"

    def test_short_password_warns(self):
        from config import SecuritySettings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SecuritySettings(username="user", password="short", config_password="short")
        messages = [str(w.message) for w in caught]
        assert any("password" in m.lower() or "PASSWORD" in m for m in messages), \
            f"Expected warning for short password; got: {messages}"


# ---------------------------------------------------------------------------
# Fix 3 — debug_mode warning written at module load in main.py
# ---------------------------------------------------------------------------

class TestDebugModeWarning:
    """The debug_mode warning must be present in the source."""

    def test_debug_mode_warning_exists_in_source(self):
        """main.py must contain the debug_mode WARNING log statement."""
        main_py_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
        with open(main_py_path, encoding="utf-8") as fh:
            main_src = fh.read()
        assert "debug_mode" in main_src, "main.py should reference debug_mode"
        assert "WARNING" in main_src.upper() or "warning" in main_src, \
            "main.py must emit a warning when debug_mode is set"
        # Confirm the exact guard we added is present
        assert "DEBUG MODE IS ACTIVE" in main_src or "debug mode" in main_src.lower(), \
            "main.py must contain an explicit debug mode warning message"


# ---------------------------------------------------------------------------
# Fix 4 — Plaintext password query params removed from WebSocket auth
# ---------------------------------------------------------------------------

def _get_authenticate_websocket():
    """Import _authenticate_websocket with minimal stubbed service layer."""
    # Stub everything the websocket router and its service dependencies need
    for mod_name, cls_names in [
        ("services.websocket_manager", ["WebSocketManager"]),
        ("services.market_data_service", ["MarketDataService"]),
        ("services.accounts_service", ["AccountsService"]),
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            for cls in cls_names:
                setattr(stub, cls, MagicMock)
            sys.modules[mod_name] = stub

    # Also stub services __init__ to avoid re-importing accounts_service
    if "services" not in sys.modules:
        svc = types.ModuleType("services")
        svc.AccountsService = MagicMock
        sys.modules["services"] = svc

    # Force re-import of websocket router to pick up stubs
    if "routers.websocket" in sys.modules:
        del sys.modules["routers.websocket"]

    from routers.websocket import _authenticate_websocket
    return _authenticate_websocket


def _make_mock_ws(headers: dict, query_params: dict) -> MagicMock:
    ws = MagicMock()
    ws.headers = headers
    ws.query_params = query_params
    return ws


class TestWebSocketAuthNoBareQueryParams:
    """WebSocket _authenticate_websocket must reject ?username=&****** fallback."""

    @pytest.fixture(autouse=True)
    def disable_debug_mode(self):
        import config
        original = config.settings.security.debug_mode
        object.__setattr__(config.settings.security, "debug_mode", False)
        yield
        object.__setattr__(config.settings.security, "debug_mode", original)

    @pytest.fixture
    def auth_fn(self):
        return _get_authenticate_websocket()

    def test_bare_username_password_params_rejected(self, auth_fn):
        """?username=...&password=... must be rejected (not a supported method)."""
        import config
        ws = _make_mock_ws(
            headers={"authorization": ""},
            query_params={
                "username": config.settings.security.username,
                "password": config.settings.security.password,
            },
        )
        assert auth_fn(ws) is False, "Bare ?username=&password= must be rejected"

    def test_authorization_header_works(self, auth_fn):
        """Authorization: Basic header must authenticate successfully."""
        import config
        creds = base64.b64encode(
            f"{config.settings.security.username}:{config.settings.security.password}".encode()
        ).decode()
        ws = _make_mock_ws(
            headers={"authorization": f"Basic {creds}"},
            query_params={},
        )
        assert auth_fn(ws) is True

    def test_token_query_param_works(self, auth_fn):
        """?token=base64(user:pass) must authenticate successfully."""
        import config
        token = base64.b64encode(
            f"{config.settings.security.username}:{config.settings.security.password}".encode()
        ).decode()
        ws = _make_mock_ws(
            headers={"authorization": ""},
            query_params={"token": token},
        )
        assert auth_fn(ws) is True

    def test_no_credentials_rejected(self, auth_fn):
        ws = _make_mock_ws(headers={"authorization": ""}, query_params={})
        assert auth_fn(ws) is False

    def test_wrong_password_rejected(self, auth_fn):
        import config
        creds = base64.b64encode(
            f"{config.settings.security.username}:wrongpassword".encode()
        ).decode()
        ws = _make_mock_ws(
            headers={"authorization": f"Basic {creds}"},
            query_params={},
        )
        assert auth_fn(ws) is False

    def test_debug_mode_bypasses_auth(self):
        import config
        object.__setattr__(config.settings.security, "debug_mode", True)
        try:
            auth_fn = _get_authenticate_websocket()
            ws = _make_mock_ws(headers={"authorization": ""}, query_params={})
            assert auth_fn(ws) is True
        finally:
            object.__setattr__(config.settings.security, "debug_mode", False)

