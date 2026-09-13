# -*- coding: utf-8 -*-
"""
main.py configures the root logger before building the app (#4774)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#3537 removed `logging.basicConfig` from main.py on the theory that
`uvicorn.run()` installs handlers on the root logger, and a second
`basicConfig` call duplicated every line. Checked against uvicorn's actual
`LOGGING_CONFIG`: it only ever configures the "uvicorn"/"uvicorn.access"
loggers (propagate=False) and "uvicorn.error" (stops at "uvicorn"'s
handler) -- never root. Every module logger in this codebase propagates
straight to root, which had no handler at any point in the process's
lifetime, so `logging.lastResort` (level WARNING) silently ate all 20
router-registration confirmations in config/routes.py, plus the sys.path
bootstrap's own INFO/DEBUG lines just below where this now sits.

These are source-level checks (matching this file's sibling
`test_main_py_sys_path_bootstrap_still_logs_path_at_debug` for the same
module), not an in-process import: main.py is already imported by many other
backend test modules in this suite, and `logging.basicConfig()` is a no-op on
a second call in the same process, so a naive "root has a handler after
`import main`" assertion would pass regardless of whether this fix is
present, depending on test order. A fresh `python main.py`/`import main` was
verified manually instead (see the commit): router-registration lines now
appear, and uvicorn's own "Uvicorn running on ..." line still appears exactly
once (no #3537 regression).
"""

import re
from pathlib import Path

_MAIN_PY = Path(__file__).parent.parent.parent / "auralis-web" / "backend" / "main.py"


def _source() -> str:
    return _MAIN_PY.read_text()


class TestLoggingConfiguredBeforeAppConstruction:
    def test_basic_config_is_called(self):
        source = _source()
        assert re.search(r'logging\.basicConfig\(', source), (
            "main.py no longer calls logging.basicConfig() -- router-registration "
            "INFO/DEBUG logs will be silently dropped by logging.lastResort again"
        )

    def test_basic_config_runs_before_the_sys_path_bootstrap_logs(self):
        """The bootstrap block's own logger.info/.debug calls (just below)
        must not fire before basicConfig -- they were part of what #4774
        found silently dropped."""
        source = _source()
        basic_config_pos = source.index('logging.basicConfig(')
        bootstrap_pos = source.index("getattr(sys, 'frozen'")
        assert basic_config_pos < bootstrap_pos, (
            "logging.basicConfig() must run before the sys.path bootstrap "
            "block's own logger.info/.debug calls"
        )

    def test_level_is_gated_on_dev_mode_not_hardcoded(self):
        """Router-registration confirmations are INFO; DEBUG detail (e.g.
        config/routes.py's 'Health router registered') should additionally
        surface in dev mode, mirroring is_dev_mode()'s own --dev/env-var check."""
        source = _source()
        basic_config_call = source[source.index('logging.basicConfig('):][:300]
        assert 'AURALIS_DEV_MODE' in source, (
            "the dev-mode env var is checked nowhere near the logging bootstrap"
        )
        assert 'logging.DEBUG' in basic_config_call and 'logging.INFO' in basic_config_call, (
            "logging level must switch between DEBUG (dev) and INFO (prod), not "
            "be hardcoded to one level"
        )

    def test_format_includes_logger_name_and_level(self):
        """A bare basicConfig() default format ('%(message)s') would make
        every propagated line indistinguishable from a bare print -- assert
        the format string carries enough to diagnose which module logged
        what at what severity."""
        source = _source()
        basic_config_call = source[source.index('logging.basicConfig('):][:300]
        assert '%(name)s' in basic_config_call
        assert '%(levelname)s' in basic_config_call

    def test_does_not_reintroduce_the_false_uvicorn_root_logger_claim(self):
        """The #3537 comment this replaced asserted uvicorn.run() configures
        the root logger. It doesn't (checked against uvicorn.config.LOGGING_CONFIG:
        only 'uvicorn'/'uvicorn.access'/'uvicorn.error', never ''/root) -- pin
        that the corrected rationale doesn't quietly regress back to the old
        claim on some future edit."""
        source = _source()
        assert 'installs its own logging configuration with handlers on the root logger' not in source
