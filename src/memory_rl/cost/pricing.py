"""Provider pricing profiles (plan.md §6.4, invariant §4.2.12).

Prices live only in versioned YAML files under ``configs/pricing/``; every
profile must record the page it was taken from and the date it was read. This
module validates them and never supplies a default price.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

CacheMode = Literal["automatic", "breakpoint"]
_MODES: tuple[str, ...] = ("automatic", "breakpoint")

#: Repository default location of the pricing profiles.
DEFAULT_PRICING_DIR = Path(__file__).resolve().parents[3] / "configs" / "pricing"


class PricingError(ValueError):
    """Raised for an invalid or incomplete pricing profile."""


@dataclass(frozen=True, slots=True)
class PricingProfile:
    """One provider/model's prompt-caching price schedule.

    ``mode`` selects the cache semantics simulated by ``PrefixCacheSimulator``:

    * ``automatic`` — every prompt becomes a cache entry; a later prompt reads
      its longest common prefix with any live entry (rounded down to
      ``block_size``). OpenAI-style.
    * ``breakpoint`` — entries are written only at explicit breakpoints and a
      read hits only when a *whole* entry is a prefix of the new prompt.
      Anthropic-style ``cache_control``.
    """

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float
    cache_write_multiplier: float
    min_cacheable_tokens: int
    block_size: int
    ttl_seconds: float
    mode: CacheMode
    source: str
    retrieved: str
    model: str = ""
    notes: str = ""
    illustrative: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PricingError("pricing profile needs a non-empty name")
        for field_name in (
            "input_per_mtok",
            "output_per_mtok",
            "cache_read_multiplier",
            "cache_write_multiplier",
            "ttl_seconds",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise PricingError(f"{self.name}: {field_name} must be a finite number >= 0")
        if self.ttl_seconds <= 0:
            raise PricingError(f"{self.name}: ttl_seconds must be > 0")
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise PricingError(f"{self.name}: block_size must be an integer >= 1")
        if not isinstance(self.min_cacheable_tokens, int) or self.min_cacheable_tokens < 0:
            raise PricingError(f"{self.name}: min_cacheable_tokens must be an integer >= 0")
        if self.mode not in _MODES:
            raise PricingError(f"{self.name}: mode must be one of {_MODES}, got {self.mode!r}")
        if not str(self.source).strip():
            raise PricingError(f"{self.name}: source (URL of the pricing page) is required")
        if not str(self.retrieved).strip():
            raise PricingError(f"{self.name}: retrieved (ISO date the price was read) is required")

    # --- $ per token --------------------------------------------------------

    @property
    def input_per_token(self) -> float:
        return self.input_per_mtok / 1_000_000

    @property
    def output_per_token(self) -> float:
        return self.output_per_mtok / 1_000_000

    @property
    def cache_read_per_token(self) -> float:
        return self.input_per_token * self.cache_read_multiplier

    @property
    def cache_write_per_token(self) -> float:
        return self.input_per_token * self.cache_write_multiplier

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED = (
    "name",
    "input_per_mtok",
    "output_per_mtok",
    "cache_read_multiplier",
    "cache_write_multiplier",
    "min_cacheable_tokens",
    "block_size",
    "ttl_seconds",
    "mode",
    "source",
    "retrieved",
)


def profile_from_mapping(data: Mapping[str, Any]) -> PricingProfile:
    """Build a profile from a parsed mapping; missing keys are an error."""
    missing = [key for key in _REQUIRED if key not in data or data[key] in (None, "")]
    if missing:
        raise PricingError(f"pricing profile is missing required keys: {missing}")
    known = set(_REQUIRED) | {"model", "notes", "illustrative"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise PricingError(f"pricing profile has unknown keys: {unknown}")
    return PricingProfile(
        name=str(data["name"]),
        input_per_mtok=float(data["input_per_mtok"]),
        output_per_mtok=float(data["output_per_mtok"]),
        cache_read_multiplier=float(data["cache_read_multiplier"]),
        cache_write_multiplier=float(data["cache_write_multiplier"]),
        min_cacheable_tokens=int(data["min_cacheable_tokens"]),
        block_size=int(data["block_size"]),
        ttl_seconds=float(data["ttl_seconds"]),
        mode=data["mode"],
        source=str(data["source"]),
        retrieved=str(data["retrieved"]),
        model=str(data.get("model", "")),
        notes=str(data.get("notes", "")),
        illustrative=bool(data.get("illustrative", False)),
    )


def load_profile(path: str | Path) -> PricingProfile:
    """Load and validate one YAML pricing profile."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PricingError(f"cannot read pricing profile {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise PricingError(f"pricing profile {path} must be a YAML mapping")
    return profile_from_mapping(data)


def available_profiles(directory: str | Path = DEFAULT_PRICING_DIR) -> dict[str, PricingProfile]:
    """All profiles in a directory, keyed by name (illustrative ones included)."""
    profiles: dict[str, PricingProfile] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        profile = load_profile(path)
        if profile.name in profiles:
            raise PricingError(f"duplicate pricing profile name {profile.name!r} in {directory}")
        profiles[profile.name] = profile
    return profiles


def get_profile(name: str, directory: str | Path = DEFAULT_PRICING_DIR) -> PricingProfile:
    profiles = available_profiles(directory)
    try:
        return profiles[name]
    except KeyError as exc:
        raise PricingError(
            f"unknown pricing profile {name!r}; available: {sorted(profiles)}"
        ) from exc
