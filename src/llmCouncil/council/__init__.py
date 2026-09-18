"""LLM council: put one question to several independent models.

    from council import load_config, run_council, succeeded

    config = load_config("config.json")
    result = run_council("Which city is the capital of Belgium?", config)
    if succeeded(result):
        print(result["head"]["answer"])
"""

from council.answer import extract as extract_answer, is_undetermined
from council.orchestrator import (
    ConfigError,
    EndpointsUnavailable,
    PROMPT_VERSION,
    load_config,
    run_council,
    succeeded,
)

__all__ = [
    "ConfigError",
    "extract_answer",
    "is_undetermined",
    "EndpointsUnavailable",
    "PROMPT_VERSION",
    "load_config",
    "run_council",
    "succeeded",
]