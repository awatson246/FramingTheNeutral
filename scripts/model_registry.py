"""
Load the model registry from config/models.yaml.

Exposes, keyed by model key:
  MODELS              api_id         (callable v2 models only)
  MODEL_PROVIDERS     provider       (callable v2 models only)
  MODEL_HEADROOM      reasoning_headroom (callable v2 models only)
  MODEL_LABELS        display label  (v2 + legacy, for plots of older data)
  MODEL_COLORS        plot color     (v2 + legacy)
"""

from pathlib import Path

import yaml

MODELS_FILE = Path(__file__).resolve().parent.parent / "config" / "models.yaml"

PROVIDERS = {"openai", "anthropic", "google", "deepseek", "mistral", "xai"}
MODEL_FIELDS  = {"key", "label", "provider", "api_id", "color", "reasoning_headroom"}
LEGACY_FIELDS = {"key", "label", "color"}


def _check(entries: list[dict], required: set[str], section: str) -> None:
    for e in entries:
        missing = required - set(e)
        if missing:
            raise ValueError(f"{MODELS_FILE}: {section} entry {e.get('key', e)} "
                             f"is missing {sorted(missing)}")
        if "provider" in required and e["provider"] not in PROVIDERS:
            raise ValueError(f"{MODELS_FILE}: {e['key']} has unknown provider "
                             f"{e['provider']!r} (expected one of {sorted(PROVIDERS)})")


def load_registry(path: Path = MODELS_FILE) -> tuple[list[dict], list[dict]]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("models") or []
    legacy = cfg.get("legacy_models") or []
    _check(models, MODEL_FIELDS, "models")
    _check(legacy, LEGACY_FIELDS, "legacy_models")
    keys = [e["key"] for e in models + legacy]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate model keys {dupes}")
    return models, legacy


_models, _legacy = load_registry()

MODELS          = {m["key"]: m["api_id"]             for m in _models}
MODEL_PROVIDERS = {m["key"]: m["provider"]           for m in _models}
MODEL_HEADROOM  = {m["key"]: m["reasoning_headroom"] for m in _models}
MODEL_LABELS    = {m["key"]: m["label"]              for m in _models + _legacy}
MODEL_COLORS    = {m["key"]: m["color"]              for m in _models + _legacy}
