"""Load contract and session config. Every tick size, margin figure, session
time, and spread value comes from these files, never from engine code."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_contracts(path=None):
    with open(path or CONFIG_DIR / "contracts.yaml") as f:
        data = yaml.safe_load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_sessions(path=None):
    with open(path or CONFIG_DIR / "sessions.yaml") as f:
        data = yaml.safe_load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_phases(path=None):
    with open(path or CONFIG_DIR / "phases.yaml") as f:
        return yaml.safe_load(f)
