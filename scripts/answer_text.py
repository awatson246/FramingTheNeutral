"""
Shared helper for reading stored I1 / I2 answers as plain text.

I2 cells (and legacy I1 cells) are plain strings. I1 cells are now
{"raw": "<model output>", "parsed": {"response": "...", "sources": [...]}}.
Anything that analyzes or re-prompts with an answer should go through
get_answer_text() so it never sees the dict or the sources list.
"""

import json


class AnswerTextError(ValueError):
    """Raised when a stored cell has no recoverable answer text."""


def _strip_fences(s: str) -> str:
    clean = s.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    return clean.strip()


def get_answer_text(cell) -> str:
    """
    Return the plain answer text for a stored I1 / I2 response cell.

    - str                               -> returned as-is
    - {"parsed": {"response": str}}     -> parsed["response"]
    - {"parsed": None, "raw": "<json>"} -> json.loads(raw)["response"]
    Sources are never included. Raises AnswerTextError otherwise.
    """
    if isinstance(cell, str):
        return cell

    if isinstance(cell, dict):
        parsed = cell.get("parsed")
        if isinstance(parsed, dict) and isinstance(parsed.get("response"), str):
            return parsed["response"]

        raw = cell.get("raw")
        if isinstance(raw, str) and raw.strip():
            try:
                # strict=False tolerates literal newlines/tabs inside strings,
                # the usual reason parse_i1_response failed at collection time
                obj = json.loads(_strip_fences(raw), strict=False)
            except json.JSONDecodeError as e:
                raise AnswerTextError(
                    f"Cell has no parsed.response and raw is not valid JSON ({e}). "
                    f"raw starts: {raw[:80]!r}"
                ) from e
            if isinstance(obj, dict) and isinstance(obj.get("response"), str):
                return obj["response"]
            raise AnswerTextError(
                f"Cell raw JSON has no string 'response' key. raw starts: {raw[:80]!r}"
            )

        raise AnswerTextError(
            f"Cell dict has no parsed.response and no raw text (keys: {sorted(cell)})"
        )

    raise AnswerTextError(f"Expected str or dict cell, got {type(cell).__name__}: {cell!r:.80}")
