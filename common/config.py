"""YAML config loading and run metadata shared by all pipeline scripts.

Scripts call :func:`parse_args_with_config`, which reads ``--config`` first and
uses the matching config section as argparse defaults. Explicit command-line
flags still override the file, so existing shell commands keep working.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import subprocess
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_section(config: dict, dotted: str) -> dict:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return {}
        node = node[key]
    return node if isinstance(node, dict) else {}


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    section: str,
    default_config: str | None = None,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, dict]:
    """Parse args, using ``config[section]`` as defaults.

    Config keys use snake_case matching the argparse ``dest`` names
    (``--batch-size`` -> ``batch_size``). Unknown keys are ignored.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_config)
    known, _ = pre.parse_known_args(argv)

    parser.add_argument("--config", default=default_config, help="YAML config file")
    config = load_config(known.config)
    defaults = get_section(config, section)
    # Flags, plus config-only keys registered with parser.set_defaults().
    valid = {action.dest for action in parser._actions} | set(parser._defaults)
    parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid})
    # Arguments marked required are satisfied by config values.
    for action in parser._actions:
        if action.required and action.dest in defaults:
            action.required = False
    args = parser.parse_args(argv)
    return args, config


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{out}-dirty" if dirty else out
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(args: argparse.Namespace | dict | None = None) -> dict:
    """Information needed to reproduce a result, stored in every metrics JSON."""
    params = vars(args) if isinstance(args, argparse.Namespace) else (args or {})
    return {
        "git_commit": git_commit(),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "params": params,
    }
