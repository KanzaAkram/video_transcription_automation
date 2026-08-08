"""Console entry point (`lecture-pipeline`).

A thin dispatcher over the scripts in scripts/, so there is one discoverable
command rather than a set of paths to remember.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

COMMANDS = {
    "dry-run": ("dry_run.py", "Run the full pipeline against a local fake Drive"),
    "verify-config": ("verify_config.py", "Validate configuration without any network calls"),
    "map-drive": ("bootstrap_drive_map.py", "Resolve syllabus nodes to Drive folders (read-only)"),
}


def _usage() -> str:
    lines = ["Usage: lecture-pipeline <command> [options]", "", "Commands:"]
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        lines.append(f"  {name:<{width}}  {description}")
    lines.append("")
    lines.append("See docs/SETUP.md for the order to run these in.")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(_usage())
        return 0

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    script = SCRIPTS / COMMANDS[command][0]
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return 1

    # Hand the remaining argv to the script as if it had been invoked directly.
    sys.argv = [str(script), *sys.argv[2:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
