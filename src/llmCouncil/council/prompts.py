"""Prompt assembly for members and the head.

Three things are configurable per endpoint:

`system_prompt`  replaces the default system prompt outright.
`role`           appended to the system prompt as a line of framing.
`context`        per-question data, passed to run_council(), shared by all.

Context is data, not framing: it is the same text for every member, so it does
not make their answers correlated in the way a per-member role would.
"""

from __future__ import annotations

# The default deliberately does NOT tell the member it is on a panel. A model
# told others will check its work tends to hedge and defer. Members cannot see
# each other, so what is wanted is each one's best independent answer. Override
# via `member_system_prompt` in config to test the alternative.
DEFAULT_MEMBER_SYSTEM_PROMPT = (
    "Answer the question directly and thoroughly. Show your reasoning. "
    "Where you are uncertain, say so explicitly rather than guessing. "
    "Do not pad the answer: additional detail you are not confident in is "
    "worse than a short answer."
)

DEFAULT_HEAD_SYSTEM_PROMPT = (
    "You chair a panel of independent experts. Each member answered the same "
    "question without seeing the others' answers.\n\n"
    "Weigh the answers on their merits and produce a single response for the "
    "reader. Apply these rules:\n"
    "- Be brief as possible in answers \n"
    "- Judge each claim on its own. A long, detailed answer is not more "
    "reliable than a short one; elaboration is often where errors appear.\n"
    "- Agreement between members is weak evidence, not proof. Members can "
    "share the same mistake.\n"
    "- Where members contradict each other on a point of fact, say so "
    "explicitly and give your own judgement with reasoning.\n"
    "- Drop claims you judge to be wrong, even when several members make "
    "them. Do not average the answers together.\n"
    "- State what remains genuinely uncertain rather than papering over it.\n\n"
    "Write the final answer directly, as the answer to the question. Do not "
    "describe the panel, the members, or the process of comparing them."
)

ANSWER_MARKER = "ANSWER:"

# For verifiable domains. Instead of weighing opinions, the head checks each
# result and discards what fails. Ends with a machine-readable line so the
# caller does not have to parse prose.
MATH_HEAD_SYSTEM_PROMPT = (
    "You chair a panel of independent experts. Each member answered the same "
    "question without seeing the others' answers. The question has a "
    "checkable answer, so do not weigh opinions - verify.\n\n"
    "Work in this order:\n"
    "1. Extract the result each member arrived at. A member who declined to "
    "answer, or who answered a different question than the one asked, "
    "contributes nothing; ignore it rather than treating it as dissent.\n"
    "2. Solve the problem yourself, from the question, before comparing. Do "
    "not start from any member's answer.\n"
    "3. Check each candidate result by substituting it back into the original "
    "problem and confirming it holds. State the substitution.\n"
    "4. Keep what survives the check. Discard what fails, however many "
    "members asserted it and however confidently.\n\n"
    "How many members agree is not evidence. One member with a result that "
    "verifies beats three members without one. Position in the list means "
    "nothing: the last answer you read is not more likely to be right.\n\n"
    "If a question looks ill-posed, answer the reasonable reading of it "
    "rather than objecting. Solving for x in '1 + 1 = x' means x = 2.\n\n"
    "Write a short explanation, then end your reply with a final line of "
    f"exactly this form and nothing after it:\n"
    f"{ANSWER_MARKER} <result>\n\n"
    "Put only the result after the marker - a number, an expression, or a "
    "short phrase. If you genuinely cannot determine it, write "
    f"'{ANSWER_MARKER} UNDETERMINED'."
)

HEAD_PRESETS = {
    "default": None,          # filled in below
    "math": MATH_HEAD_SYSTEM_PROMPT,
}

CONTEXT_HEADER = "Context for this question:"


HEAD_PRESETS["default"] = DEFAULT_HEAD_SYSTEM_PROMPT


def system_prompt(endpoint: dict, default: str) -> str:
    """Resolve an endpoint's system prompt, with its role appended if set."""
    base = endpoint.get("system_prompt") or default
    role = endpoint.get("role")
    return f"{base}\n\n{role}" if role else base


def member_system_prompt(member: dict, config: dict) -> str:
    """Member system prompt: per-member override, then config-wide, then default."""
    default = config.get("member_system_prompt") or DEFAULT_MEMBER_SYSTEM_PROMPT
    return system_prompt(member, default)


def head_system_prompt(head: dict, config: dict) -> str:
    """Head system prompt: per-head override, then config-wide text or preset.

    `head_preset` selects a built-in prompt by name ("default", "math").
    `head_system_prompt` supplies text directly and wins over the preset.
    """
    default = config.get("head_system_prompt")
    if not default:
        preset = config.get("head_preset", "default")
        if preset not in HEAD_PRESETS:
            raise ValueError(
                f"unknown head_preset {preset!r}; "
                f"choose from {sorted(HEAD_PRESETS)}"
            )
        default = HEAD_PRESETS[preset]
    return system_prompt(head, default)


def with_context(question: str, context: str | None) -> str:
    """Prepend per-question context to the question, if any was given."""
    if not context:
        return question
    return f"{CONTEXT_HEADER}\n{context.strip()}\n\n{question}"


def head_prompt(question: str, members: list[dict],
                context: str | None = None) -> str:
    """Assemble the head's user message from the successful member answers.

    Members are anonymised: the head sees "Member 1", never a model name, so it
    cannot defer to a brand rather than to an argument.
    """
    parts = []
    if context:
        parts.append(f"{CONTEXT_HEADER}\n{context.strip()}\n")
    parts.append(f"Question put to the panel:\n{question}\n")
    for index, member in enumerate(members, start=1):
        parts.append(f"--- Member {index} ---\n{member['answer'].strip()}\n")
    parts.append("Produce the final answer, following the rules you were given.")
    return "\n".join(parts)