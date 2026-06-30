"""Small W&B runtime preflight helpers."""

from __future__ import annotations

import netrc
import os
from collections.abc import Mapping
from pathlib import Path

_OFFLINE_WANDB_MODES = {"offline", "dryrun"}
_DISABLED_VALUES = {"1", "true", "yes", "on"}
_WANDB_NETRC_MACHINES = ("api.wandb.ai", "wandb.ai")


def wandb_config_error(
    *,
    no_wandb: bool,
    env: Mapping[str, str] | None = None,
    netrc_path: Path | None = None,
    allow_offline: bool = True,
) -> str | None:
    """Return a user-facing W&B setup error, or ``None`` when logging can run.

    The training scripts enable W&B by default. This preflight catches the
    common no-tty failure mode before a long dataset load or model load starts.
    Explicit offline/dryrun mode is accepted because it is an intentional W&B
    mode, while disabled mode is treated as a mismatch unless ``--no-wandb`` is
    used.
    """
    if no_wandb:
        return None

    resolved_env = os.environ if env is None else env
    mode = str(resolved_env.get("WANDB_MODE", "")).strip().lower()
    if mode in _OFFLINE_WANDB_MODES:
        if allow_offline:
            return None
        return (
            f"W&B online logging is required, but WANDB_MODE={mode} is set. "
            "Unset WANDB_MODE for a cloud run, pass --allow-wandb-offline for "
            "an intentional offline W&B run, or pass --no-wandb for an "
            "intentional local-only run."
        )
    if mode == "disabled" or _truthy(resolved_env.get("WANDB_DISABLED", "")):
        return (
            "W&B logging is enabled, but WANDB_MODE/WANDB_DISABLED disables it. "
            "Unset the disable flag for W&B logging, or pass --no-wandb for an "
            "intentional local-only run."
        )

    if str(resolved_env.get("WANDB_API_KEY", "")).strip():
        return None
    if _netrc_has_wandb_auth(netrc_path=netrc_path):
        return None

    return (
        "W&B logging is enabled, but no WANDB_API_KEY or saved wandb login was "
        "found. Run `wandb login` in this runtime, set WANDB_API_KEY, or pass "
        "--no-wandb for an intentional local-only run."
    )


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in _DISABLED_VALUES


def _netrc_has_wandb_auth(*, netrc_path: Path | None = None) -> bool:
    for candidate in _netrc_candidates(netrc_path):
        if not candidate.exists():
            continue
        try:
            credentials = netrc.netrc(str(candidate))
        except (OSError, netrc.NetrcParseError):
            continue
        for machine in _WANDB_NETRC_MACHINES:
            if credentials.authenticators(machine):
                return True
    return False


def _netrc_candidates(netrc_path: Path | None) -> tuple[Path, ...]:
    if netrc_path is not None:
        return (Path(netrc_path),)

    home = Path.home()
    candidates = [home / ".netrc"]
    if os.name == "nt":
        candidates.append(home / "_netrc")
    return tuple(candidates)


__all__ = ["wandb_config_error"]
