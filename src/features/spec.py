"""Reads the 2A instruction document (feature_2a_spec.yaml)."""
from pathlib import Path

import yaml

SPEC_PATH = Path(__file__).with_name("feature_2a_spec.yaml")
SECTIONS = ("profile", "outcome", "filter", "flag")   # section name = role name


def load_spec(path: Path = SPEC_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def column_roles(spec: dict) -> dict[str, str]:
    """Every output column -> its role (key, context, profile, outcome, filter, flag)."""
    roles = dict(spec["passthrough"])
    for section in SECTIONS:
        roles.update({col: section for col in spec[section]})
    return roles


def blank_reasons(spec: dict) -> dict[str, list[str]]:
    """Column -> the only reasons it may be blank (empty list = must never be blank)."""
    return {col: body.get("blank_when", [])
            for section in SECTIONS for col, body in spec[section].items()}
