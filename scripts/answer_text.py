"""
Shared helper for reading stored I1 / I2 answers as plain text.

Stored formats:
  v1 I2 (and legacy I1):  "<answer>"
  v1 I1:                  {"raw": "<model output>", "parsed": {"response": ..., "sources": [...]}}
  v2 I2:                  {"text": "<answer>", "model_version": ..., ...}        (no "parsed")
  v2 I1:                  {"text": "<model output>", ..., "parsed": {...} | None}
Anything that analyzes or re-prompts with an answer should go through
get_answer_text() so it never sees the dict, its metadata or the sources list.
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

    - str                                      -> returned as-is
    - {"parsed": {"response": str}}            -> parsed["response"]
    - {"text": str} with no "parsed" key       -> text (v2 free-text record)
    - {"parsed": None, "raw" | "text": "<json>"} -> json.loads(...)["response"]
    Sources and metadata are never included. Raises AnswerTextError otherwise.
    """
    if isinstance(cell, str):
        return cell

    if isinstance(cell, dict):
        parsed = cell.get("parsed")
        if isinstance(parsed, dict) and isinstance(parsed.get("response"), str):
            return parsed["response"]

        # v2 free-text record (I2): the model output is the answer itself
        if "parsed" not in cell and isinstance(cell.get("text"), str):
            return cell["text"]

        # JSON output that wasn't parsed at collection time (v1 "raw", v2 "text")
        raw = cell.get("raw", cell.get("text"))
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
