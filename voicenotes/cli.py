from __future__ import annotations

import argparse
import json

from .config import config_as_dict, default_paths, load_config
from .state import status_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voicenotes")
    subparsers = parser.add_subparsers(dest="command")
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--json", action="store_true")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--json", action="store_true")

    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    if args.command == "config":
        print(json.dumps(config_as_dict(load_config()), indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        print(json.dumps(status_snapshot(default_paths()), indent=2, sort_keys=True))
        return 0

    parser.print_usage()
    return 2
