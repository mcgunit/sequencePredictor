"""Check that a llama-server endpoint is reachable and ready."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class EndpointUnavailable(RuntimeError):
    """The endpoint did not answer, or reported that it is not ready."""


def check(base_url: str, timeout_s: int = 5) -> None:
    """Raise EndpointUnavailable unless the server is up and has a model loaded.

    llama-server answers /health with 200 once a model is loaded, and with 503
    while it is still loading. A swap proxy may not implement /health at all,
    so a 404 is treated as "reachable, assume ready".
    """
    url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if resp.status == 200:
                return
            raise EndpointUnavailable(f"{url} returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            log.debug("%s has no /health, assuming ready", base_url)
            return
        if exc.code == 503:
            raise EndpointUnavailable(
                f"{base_url} is up but still loading a model"
            ) from exc
        raise EndpointUnavailable(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise EndpointUnavailable(f"{base_url} unreachable: {exc}") from exc


def loaded_models(base_url: str, timeout_s: int = 5) -> list[str]:
    """Return the model ids the endpoint advertises, or [] if it does not."""
    url = f"{base_url}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return [entry["id"] for entry in body.get("data", []) if "id" in entry]
    except (urllib.error.URLError, OSError, TimeoutError,
            json.JSONDecodeError, KeyError, TypeError):
        return []