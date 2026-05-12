"""Bootstrap CLI runner that does not rely on a working virtualenv launcher."""

from __future__ import annotations

import pathlib
import sys
import warnings


def _suppress_known_warnings() -> None:
    warnings.simplefilter("ignore", PendingDeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change in a future version\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Did not recognize type 'public\.geometry' of column '.*'",
    )


def _bootstrap_paths() -> None:
    root = pathlib.Path(__file__).resolve().parent
    site_packages = root / ".venv" / "Lib" / "site-packages"

    for path in (root, site_packages):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> int:
    _bootstrap_paths()
    _suppress_known_warnings()
    from src.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
