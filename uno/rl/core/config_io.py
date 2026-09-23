from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tomllib

from uno.rl.core.config import Collection, Config, Evaluation, Optim, RecurrentModel, RelationalModel
from uno.rl.core.runner import Execution, Tracking, TrainSpec


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def model_from_dict(data: dict):
    match data.get("kind", "relational"):
        case "relational":
            return RelationalModel(**data)
        case "recurrent":
            return RecurrentModel(**data)
        case kind:
            raise ValueError(f"unknown model kind {kind!r}")


def set_path(values: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    current = values

    for part in parts[:-1]:
        current = current.setdefault(part, {})

    current[parts[-1]] = value


def apply_overrides(values: dict, overrides: list[str]) -> None:
    for override in overrides:
        key, value = override.split("=", 1)
        set_path(values, key, parse_value(value))


def from_dict(values: dict) -> TrainSpec:
    values = deepcopy(values)

    optim_values = values.pop("optim", {})
    collection_values = values.pop("collection", {})
    evaluation_values = values.pop("evaluation", {})
    execution_values = values.pop("execution", {})
    tracking_values = values.pop("tracking", {})
    league = values.pop("league", "rulebot")

    # Backwards compatibility for older checkpoints where width/blocks are at root
    if "width" in values or "blocks" in values:
        model_values = {"kind": "relational"}
        if "width" in values:
            model_values["width"] = values.pop("width")
        if "blocks" in values:
            model_values["blocks"] = values.pop("blocks")
        # Let explicit model dict override if it somehow existed
        model_values.update(values.pop("model", {}))
    else:
        model_values = values.pop("model", {})

    config = replace(
        Config(),
        **values,
        model=model_from_dict(model_values),
        optim=replace(Optim(), **optim_values),
        collection=replace(Collection(), **collection_values),
        evaluation=replace(Evaluation(), **evaluation_values),
    )

    return TrainSpec(
        config=config,
        league=league,
        execution=Execution(**execution_values),
        tracking=Tracking(**tracking_values),
    )


def load(path: Path, overrides: list[str] = None) -> TrainSpec:
    with path.open("rb") as f:
        values = tomllib.load(f)
    
    if overrides:
        apply_overrides(values, overrides)
        
    return from_dict(values)


PRESETS = Path("configs/rl")

def preset_path(name: str) -> Path:
    candidates = (
        PRESETS / "local" / f"{name}.toml",
        PRESETS / f"{name}.toml",
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"unknown RL preset {name!r}"
    )


def load_preset(name: str, overrides: list[str] = None) -> TrainSpec:
    path = preset_path(name)
    return load(path, overrides)


def override_seed(spec: TrainSpec, seed: int) -> TrainSpec:
    # A small helper since seed is special-cased in the CLI
    return replace(spec, config=spec.config.with_(training_seed=seed))
