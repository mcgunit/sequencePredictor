"""Query a llama-server through its OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class CompletionError(RuntimeError):
    """The server returned an error or an unusable response."""


class PermanentError(CompletionError):
    """The request will fail the same way if repeated. Do not retry."""


def _post(base_url: str, body: dict, timeout_s: int) -> dict:
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        # 4xx means the request itself is wrong; repeating it will not help.
        # 5xx and 429 are worth another attempt.
        if 400 <= exc.code < 500 and exc.code != 429:
            raise PermanentError(f"HTTP {exc.code}: {detail}") from exc
        raise CompletionError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise CompletionError(f"request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CompletionError(f"response was not JSON: {exc}") from exc


def _post_stream(base_url: str, body: dict, timeout_s: int, on_chunk) -> str:
    """The same request with `stream: true`, assembling the reply from the
    server-sent events and handing the text-so-far to `on_chunk` as it grows.

    The timeout is urllib's socket timeout, i.e. the longest the server may
    go WITHOUT sending anything - a reply that keeps producing tokens for
    longer than timeout_s is fine, a stall is not, which is the right shape
    for slow CPU inference.
    """
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(dict(body, stream=True)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    parts.append(delta)
                    try:
                        on_chunk("".join(parts))
                    except Exception:                    # noqa: BLE001
                        log.debug("chunk callback failed", exc_info=True)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if 400 <= exc.code < 500 and exc.code != 429:
            raise PermanentError(f"HTTP {exc.code}: {detail}") from exc
        raise CompletionError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise CompletionError(f"request failed: {exc}") from exc
    if not parts:
        raise CompletionError("streamed reply carried no content")
    return "".join(parts)


def build_body(question: str,
               system_prompt: str | None = None,
               model: str | None = None,
               temperature: float = 0.7,
               max_tokens: int = 2048,
               seed: int | None = None) -> dict:
    """Assemble the request body. Optional fields are omitted when unset."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if model:
        body["model"] = model
    if seed is not None:
        body["seed"] = seed
    return body


def ask(base_url: str, question: str, timeout_s: int,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: int | None = None,
        retries: int = 0,
        retry_delay_s: float = 5.0,
        on_chunk=None) -> str:
    """Send one question and return the assistant's reply text.

    `seed` is omitted from the request when None, leaving the server to choose
    one. A fixed seed reproduces an output only when the prompt, model, quant,
    server flags and backend are all unchanged as well.

    `retries` counts additional attempts after the first. Only transient
    failures are retried; a 4xx response is raised immediately.

    `on_chunk`, when given, switches the request to streaming and is called
    with the reply text so far as tokens arrive; the return value is the same
    complete text either way. A retried attempt starts its text from empty
    again, and the callback sees that.
    """
    body = build_body(question, system_prompt, model,
                      temperature, max_tokens, seed)

    last_error: CompletionError | None = None
    for attempt in range(retries + 1):
        if attempt:
            delay = retry_delay_s * attempt
            log.warning("retry %d/%d in %.0fs (%s)",
                        attempt, retries, delay, last_error)
            time.sleep(delay)
        try:
            if on_chunk is not None:
                return _post_stream(base_url, body, timeout_s, on_chunk)
            payload = _post(base_url, body, timeout_s)
        except PermanentError:
            raise
        except CompletionError as exc:
            last_error = exc
            continue

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentError(
                f"unexpected response shape: {payload}"
            ) from exc

    raise last_error if last_error else CompletionError("no attempt was made")