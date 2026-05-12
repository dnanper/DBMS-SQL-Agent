"""Run the local FastAPI web UI."""

from __future__ import annotations

import asyncio
import sys

from run_cli import _bootstrap_paths, _suppress_known_warnings


def main() -> int:
    _bootstrap_paths()
    _suppress_known_warnings()
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import uvicorn

    uvicorn.run("src.webapp:create_app", host="127.0.0.1", port=8000, reload=False, factory=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
