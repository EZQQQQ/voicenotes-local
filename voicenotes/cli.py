from __future__ import annotations

import argparse
import json

from .config import config_as_dict, default_paths, load_config
from .recorder import list_audio_devices, record_test, start_recording, stop_recording
from .state import status_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voicenotes")
    subparsers = parser.add_subparsers(dest="command")
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--json", action="store_true")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("devices")
    subparsers.add_parser("start")
    subparsers.add_parser("stop")
    subparsers.add_parser("toggle")
    record_test_parser = subparsers.add_parser("record-test")
    record_test_parser.add_argument("--duration", type=int, default=10)

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
    if args.command == "devices":
        print("\n".join(list_audio_devices()))
        return 0
    if args.command == "start":
        print(start_recording(load_config(), default_paths()))
        return 0
    if args.command == "stop":
        print(stop_recording(default_paths()))
        return 0
    if args.command == "toggle":
        paths = default_paths()
        if (paths.run / "current-recording.json").exists():
            print(stop_recording(paths))
        else:
            print(start_recording(load_config(), paths))
        return 0
    if args.command == "record-test":
        print(record_test(load_config(), default_paths(), args.duration))
        return 0

    parser.print_usage()
    return 2
