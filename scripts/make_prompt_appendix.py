"""
Appendix D: every prompt the study sent, verbatim, with variable slots shown as
{placeholders}, written to paper/appendix/prompts.tex and paper/appendix/prompts.md.

Sources:
  stored at run time (taken from the data file, then compared with the code):
    data/processed/i2_criteria_v1.json   meta.system_prompt, meta.user_template (v1)
    data/processed/i2_codes.json         meta.system_prompt, meta.user_template
    data/processed/i5_second_judge.json  meta.system_prompt, meta.user_template, tier_def, verifiable_def
  not stored (rendered by calling the code that sent them, with sentinel slot values):
    data/prompts/instruments.json        system prompts, I1 questions, I2 scenarios
    collect_llm_responses.py             build_i1_prompt, build_i3_prompt, build_i4_prompt,
                                         build_i5_prompt, call_i5_extractor system prompt
    plot_response_results.py             extract_structured, get_sankey_label_mapping
                                         (data/processed/sankey_extractions.json stores outputs only)
Checks (printed): each stored template against the code; each collection-script prompt
against the code at the spring run (git f07935c, 2026-04-14) via read-only `git show`.

No API calls; data/ is only read.

Usage:
    python scripts/make_prompt_appendix.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules["torchvision"] = None   # torch/torchvision mismatch; not needed here

import ast
import inspect
import json
import subprocess
import textwrap
from pathlib import Path

import collect_llm_responses as clr
import i2_code
import i5_second_judge
import plot_response_results as prp

OUT_DIR       = Path("paper/appendix")
SPRING_COMMIT = "f07935c"   # 2026-04-14: spring collection of I1-I5 for the five original models
INSTRUMENTS   = json.loads(clr.INSTRUMENTS_FILE.read_text(encoding="utf-8"))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


CRIT_META  = load("data/processed/i2_criteria_v1.json")["meta"]
CODES_META = load("data/processed/i2_codes.json")["meta"]
JUDGE_META = load("data/processed/i5_second_judge.json")["meta"]


def slot(name: str) -> str:
    """Sentinel for a variable slot; replaced by {name} after rendering."""
    return f"\x00{name}\x00"


def unslot(text: str) -> str:
    out, parts = [], text.split("\x00")
    for i, p in enumerate(parts):
        out.append(f"{{{p}}}" if i % 2 else p)
    return "".join(out)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

CHECKS: list[str] = []


def note(ok: bool, what: str) -> None:
    CHECKS.append(f"[{'ok' if ok else 'DIFF'}] {what}")


def git_show(commit: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def func_source(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return ""


def module_constant(src: str, name: str):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def run_checks() -> None:
    head = Path("scripts/collect_llm_responses.py").read_text(encoding="utf-8")
    spring = git_show(SPRING_COMMIT, "scripts/collect_llm_responses.py")
    for fn in ["build_i1_prompt", "build_i3_prompt", "build_i4_prompt", "build_i5_prompt"]:
        note(func_source(head, fn) == func_source(spring, fn),
             f"collect_llm_responses.{fn}: current code == spring run ({SPRING_COMMIT})")
    note(func_source(head, "call_i5_extractor").split("system=")[1].split("messages=")[0]
         == func_source(spring, "call_i5_extractor").split("system=")[1].split("messages=")[0],
         f"call_i5_extractor system prompt: current code == spring run ({SPRING_COMMIT})")
    for key in ["I1_QUESTION_TEXT", "I2_QUESTION_TEXT"]:
        note(module_constant(head, key) == module_constant(spring, key),
             f"collect_llm_responses.{key}: current code == spring run ({SPRING_COMMIT})")
    old_inst = json.loads(git_show(SPRING_COMMIT, "data/prompts/instruments.json"))
    for k in ["conditions"]:
        note(old_inst[k] == INSTRUMENTS[k], f"instruments.json {k}: current == spring run")
    for i in ["instrument_1", "instrument_2", "instrument_3", "instrument_4"]:
        a, b = old_inst["instruments"][i], INSTRUMENTS["instruments"][i]
        note(a == b, f"instruments.json {i}: current == spring run")
    # stored templates vs code
    note(CODES_META["system_prompt"] == i2_code.SYSTEM_PROMPT
         and CODES_META["user_template"] == i2_code.USER_TEMPLATE,
         "i2_codes.json meta prompts == scripts/i2_code.py SYSTEM_PROMPT/USER_TEMPLATE")
    note(JUDGE_META["system_prompt"] == i5_second_judge.SYSTEM_PROMPT
         and JUDGE_META["user_template"] == i5_second_judge.USER_TEMPLATE
         and JUDGE_META["tier_def"] == i5_second_judge.TIER_DEF
         and JUDGE_META["verifiable_def"] == i5_second_judge.VERIFIABLE_DEF,
         "i5_second_judge.json meta prompts == scripts/i5_second_judge.py")
    import i2_criteria
    note(CRIT_META["system_prompt"] == i2_criteria.SYSTEM_PROMPT,
         "i2_criteria_v1.json meta.system_prompt == scripts/i2_criteria.py SYSTEM_PROMPT")
    note(CRIT_META["user_template"] != i2_criteria.USER_TEMPLATE,
         "i2_criteria_v1.json meta.user_template is the v1 template (current code holds v2; "
         "v1 code was never committed, so the data file is the only record)")
    note(JUDGE_META["tier_def"] in clr.build_i5_prompt("x", "y"),
         "second-judge tier definition is word for word the extractor's")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def capture_sankey_prompts() -> dict:
    """Render the Sankey prompts through plot_response_results without calls or writes."""
    captured = []
    prp._sankey_store = {"meta": {}, "extractions": {}, "label_maps": {}}   # never the data file
    prp._save_store = lambda: None
    prp.sleep = lambda s: None

    def fake_call(prompt: str, max_tokens: int) -> str:
        captured.append((prompt, max_tokens))
        return "{}"
    prp._sankey_call = fake_call
    prp.SANKEY_DRY_RUN = False
    prp.extract_structured(slot("response"), "s2_accountability")
    prp.extract_structured(slot("response"), "s3_enforcement")
    prp.get_sankey_label_mapping([slot("labels")], slot("context"))
    (s2, t2), (s3, t3), (lm, t4) = captured
    # the label list is sent as json.dumps(labels, indent=2): show it as one slot
    lm = lm.replace(json.dumps([slot("labels")], indent=2), slot("labels"))
    return {"s2": (s2, t2), "s3": (s3, t3), "merge": (lm, t4)}


def criteria_v1_prompt() -> str:
    return CRIT_META["user_template"].format(label=slot("label"), question=slot("question"),
                                             definition=slot("definition"),
                                             max_n=CRIT_META["max_criteria"])


def i5_extractor_system() -> str:
    src = inspect.getsource(clr.call_i5_extractor)
    expr = src.split("system=")[1].split("messages=")[0].strip().rstrip(",")
    return ast.literal_eval(expr)


# ---------------------------------------------------------------------------
# Settings rows
# ---------------------------------------------------------------------------

SPRING = "five original models: queried before 2026-04-14 (commit f07935c; per-call dates not recorded)"


def subject_settings(max_tokens: int) -> str:
    return (f"GPT-4o: gpt-4o, temperature 1, max_tokens {max_tokens}; "
            f"Claude Sonnet: claude-sonnet-4-5, provider default temperature, max_tokens {max_tokens}; "
            f"DeepSeek-V3: deepseek-chat, default temperature, max_tokens {max_tokens}; "
            f"Mistral Large: mistral-large-latest, default temperature, max_tokens {max_tokens}; "
            f"Gemini: gemini-3.1-flash-lite-preview, default temperature, no max_tokens sent; "
            f"returned versions not recorded; {SPRING}. "
            f"Grok: grok-4.7 (returned grok-4.7), default temperature, "
            f"max_completion_tokens {max_tokens} + {clr.GROK_REASONING_HEADROOM} reasoning headroom; "
            f"queried 2026-10-22")


def build_entries() -> list[dict]:
    cond = INSTRUMENTS["conditions"]
    i1q = INSTRUMENTS["instruments"]["instrument_1"]["questions"]
    i2q = INSTRUMENTS["instruments"]["instrument_2"]["questions"]
    i3 = INSTRUMENTS["instruments"]["instrument_3"]
    i4 = INSTRUMENTS["instruments"]["instrument_4"]
    sank = capture_sankey_prompts()
    E = []

    def add(group, heading, text, source, slots=None, settings=None):
        E.append({"group": group, "heading": heading, "text": unslot(text), "source": source,
                  "slots": slots or [], "settings": settings})

    g = "System prompts (all subject-model instruments)"
    add(g, "Baseline system prompt", cond["baseline"]["system_prompt"],
        "data/prompts/instruments.json: conditions.baseline.system_prompt",
        settings="Sent as the system message for I1, I2, I3 (baseline) and all I4 calls")
    add(g, "CEO system prompt", cond["ceo"]["system_prompt"],
        "data/prompts/instruments.json: conditions.ceo.system_prompt",
        settings="Sent as the system message for I1, I2, I3 (CEO condition)")

    g = "Instrument 1: definitions"
    add(g, "I1 user prompt (JSON wrapper)", clr.build_i1_prompt({"text": slot("question")}),
        "scripts/collect_llm_responses.py: build_i1_prompt",
        slots=[("question", "one of the three I1 questions below")],
        settings=subject_settings(clr.MAX_TOKENS_JSON) + "; 3 runs per model and condition")
    for q in i1q:
        add(g, f"{q['id']} question", q["text"],
            f"data/prompts/instruments.json: instruments.instrument_1.questions[{q['id']}].text")

    g = "Instrument 2: scenarios"
    for q in i2q:
        add(g, f"{q['id']} scenario (sent as the user prompt)", q["text"],
            f"data/prompts/instruments.json: instruments.instrument_2.questions[{q['id']}].text",
            settings=None)
    E[-3]["settings"] = (subject_settings(clr.MAX_TOKENS_OPEN) + "; 3 runs per model and condition. "
                         "Mistral Large re-collection (primary): mistral-large-2512 (returned "
                         "mistral-large-2512), default temperature, max_tokens 4096, queried 2026-10-22")

    add(g, "Sankey extraction, housing scenario (I2_S2)", sank["s2"][0],
        "scripts/plot_response_results.py: extract_structured, extraction_type s2_accountability",
        slots=[("response", "run-1 answer to I2_S2 for one model and condition, plain text "
                            "(get_answer_text), first 2,000 characters")],
        settings=f"claude-sonnet-4-5-20250929 (returned version not recorded), provider default "
                 f"temperature, max_tokens {sank['s2'][1]}; queried 2026-10-27")
    add(g, "Sankey extraction, audit scenario (I2_S3)", sank["s3"][0],
        "scripts/plot_response_results.py: extract_structured, extraction_type s3_enforcement",
        slots=[("response", "run-1 answer to I2_S3 for one model and condition, plain text "
                            "(get_answer_text), first 2,000 characters")],
        settings=f"claude-sonnet-4-5-20250929 (returned version not recorded), provider default "
                 f"temperature, max_tokens {sank['s3'][1]}; queried 2026-10-27")
    add(g, "Sankey label merging", sank["merge"][0],
        "scripts/plot_response_results.py: get_sankey_label_mapping",
        slots=[("context", "accountability_responsible_parties, accountability_mechanisms, "
                           "enforcement_challenges or enforcement_solutions"),
               ("labels", "the distinct extracted labels for that context, as a JSON list "
                          "(json.dumps, indent 2)")],
        settings=f"claude-sonnet-4-5-20250929 (returned version not recorded), provider default "
                 f"temperature, max_tokens {sank['merge'][1]}; queried 2026-10-27")

    add(g, "Criterion extraction (v1): system prompt", CRIT_META["system_prompt"],
        "data/processed/i2_criteria_v1.json: meta.system_prompt")
    add(g, "Criterion extraction (v1): user prompt", criteria_v1_prompt(),
        "data/processed/i2_criteria_v1.json: meta.user_template ({max_n} = meta.max_criteria = "
        f"{CRIT_META['max_criteria']})",
        slots=[("label", "legal certainty, accountability or enforceability"),
               ("question", "the matching I1 question (I1_Q1, I1_Q2, I1_Q3), as in instruments.json"),
               ("definition", "one I1 answer, plain text (get_answer_text, no sources)")],
        settings=f"{CRIT_META['extractor_model']} (returned claude-sonnet-4-5-20250929), temperature "
                 f"{CRIT_META['temperature']}, max_tokens {CRIT_META['max_tokens']}; queried "
                 "2026-10-26 (23:24-23:34 UTC)")

    add(g, "Criterion coding: system prompt", CODES_META["system_prompt"],
        "data/processed/i2_codes.json: meta.system_prompt")
    add(g, "Criterion coding: user prompt",
        CODES_META["user_template"].format(label=slot("label"), scenario=slot("scenario"),
                                           answer=slot("answer"), criteria=slot("criteria")),
        "data/processed/i2_codes.json: meta.user_template",
        slots=[("label", "legal certainty (I2_S1), accountability (I2_S2) or enforceability (I2_S3)"),
               ("scenario", "the I2 scenario text, as in instruments.json"),
               ("answer", "one I2 answer, plain text (Mistral Large: re-collected answers)"),
               ("criteria", "the v1 criteria from the same model, condition and run's I1 answer, "
                            "one per line as \"n. criterion\"")],
        settings="Primary coder gpt-6.1-sol (returned gpt-6.1-sol), provider default temperature, "
                 "max_completion_tokens 20480 (4096 + 16384 reasoning headroom); second coder "
                 "claude-sonnet-4-5-20250929 (returned claude-sonnet-4-5-20250929), temperature 0, "
                 "max_tokens 4096; queried 2026-10-27 (01:55-02:09 UTC)")

    g = "Instrument 3: real-world decision scale"
    add(g, "I3 user prompt (all six scenarios in one message)", clr.build_i3_prompt(i3),
        "scripts/collect_llm_responses.py: build_i3_prompt (scenarios, anchors and instruction "
        "from instruments.json instrument_3)",
        settings=subject_settings(clr.MAX_TOKENS_JSON) + "; 3 runs per model and condition")

    g = "Instrument 4: peer evaluation"
    add(g, "I4 user prompt", clr.build_i4_prompt(slot("question_id"), slot("question"),
                                                 slot("response"), i4),
        "scripts/collect_llm_responses.py: build_i4_prompt (dimension prompts from "
        "instruments.json instrument_4); system message: baseline system prompt",
        slots=[("question", "the question text from collect_llm_responses.I1_QUESTION_TEXT / "
                            "I2_QUESTION_TEXT (I1 texts without the citation sentence)"),
               ("response", "the evaluatee's run-1 baseline answer to that question")],
        settings="P1-P5 (original evaluators, from instrument_4.json): unpinned IDs and parameters "
                 f"as the subject models, max_tokens {clr.MAX_TOKENS_JSON}, {SPRING}. "
                 "P3 Mistral I2 cells (re-collected answers): gpt-4o-2024-08-06 (returned "
                 "gpt-4o-2024-08-06), temperature 1, max_tokens 2048, queried 2026-10-22. "
                 "G1-G5 (Grok as evaluator): grok-4.7 (returned grok-4.7), default temperature, "
                 f"max_completion_tokens {clr.MAX_TOKENS_JSON} + {clr.GROK_REASONING_HEADROOM}, "
                 "queried 2026-10-22")
    add(g, "I4 question texts used in {question} (I1)",
        "\n\n".join(f"{k}: {v}" for k, v in clr.I1_QUESTION_TEXT.items()),
        "scripts/collect_llm_responses.py: I1_QUESTION_TEXT (I2 uses I2_QUESTION_TEXT, "
        "identical to the I2 scenarios above)")

    g = "Instrument 5: source extraction and assessment"
    add(g, "I5 extractor: system prompt", i5_extractor_system(),
        "scripts/collect_llm_responses.py: call_i5_extractor, system=")
    add(g, "I5 extractor: user prompt", clr.build_i5_prompt(slot("question_id"), slot("response")),
        "scripts/collect_llm_responses.py: build_i5_prompt",
        slots=[("question_id", "I1_Q1-I1_Q3 or I2_S1-I2_S3"),
               ("response", "the answer, plain text, first 3,000 characters")],
        settings="Five original models' answers: claude-sonnet-4-5 (alias; returned version not "
                 f"recorded), provider default temperature, max_tokens 2048, {SPRING}. Grok's answers: "
                 "claude-sonnet-4-5-20250929 (returned claude-sonnet-4-5-20250929), same settings, "
                 "queried 2026-10-22. Mistral re-collected I2 answers: same pinned model and settings, "
                 "queried 2026-10-27")
    add(g, "I5 second judge: system prompt", JUDGE_META["system_prompt"],
        "data/processed/i5_second_judge.json: meta.system_prompt")
    add(g, "I5 second judge: user prompt",
        JUDGE_META["user_template"].format(question_id=slot("question_id"), response=slot("response"),
                                           sources=slot("sources"), tier_def=JUDGE_META["tier_def"],
                                           verifiable_def=JUDGE_META["verifiable_def"]),
        "data/processed/i5_second_judge.json: meta.user_template, with meta.tier_def and "
        "meta.verifiable_def filled in",
        slots=[("question_id", "I1_Q1-I1_Q3 or I2_S1-I2_S3"),
               ("response", "the answer, plain text, first 3,000 characters"),
               ("sources", "the first judge's extracted sources, one per line as "
                           "\"k. name: ...; jurisdiction: ...; quote: \"...\"\" (quote JSON-encoded, "
                           "or null); tier, type and verifiable are not shown")],
        settings="gpt-6.1-sol (returned gpt-6.1-sol), provider default temperature, "
                 "max_completion_tokens 20480 (4096 + 16384 reasoning headroom); queried 2026-10-27 "
                 "(02:26-02:31 UTC; Mistral re-collected I2 cells 14:01-14:03 UTC)")
    return E


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

TEX_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%",
               "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
UNICODE_TEX = {"–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "→": r"$\to$",
               "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "é": r"\'e"}


def tex_escape(s: str) -> str:
    out = "".join(TEX_SPECIAL.get(ch, ch) for ch in s)
    for u, t in UNICODE_TEX.items():
        out = out.replace(u, t)
    return out


def lst_literate(entries: list[dict]) -> str:
    chars = sorted({ch for e in entries for ch in e["text"] if ord(ch) > 127})
    missing = [c for c in chars if c not in UNICODE_TEX]
    if missing:
        raise SystemExit(f"No LaTeX mapping for {missing}")
    return " ".join(f"{{{c}}}{{{{{UNICODE_TEX[c]}}}}}1" for c in chars)


def write_tex(entries: list[dict]) -> str:
    lines = [r"% Appendix D: prompts. Generated by scripts/make_prompt_appendix.py; do not edit by hand.",
             r"% Requires \usepackage{listings} in the preamble.",
             r"\lstset{basicstyle=\ttfamily\footnotesize, breaklines=true, breakatwhitespace=false,",
             r"  columns=fullflexible, keepspaces=true, frame=single, upquote=true,",
             r"  literate=" + lst_literate(entries) + "}", ""]
    group = None
    for e in entries:
        if e["group"] != group:
            group = e["group"]
            lines += [rf"\subsection*{{{tex_escape(group)}}}", ""]
        lines += [rf"\paragraph{{{tex_escape(e['heading'])}}}", "",
                  rf"\noindent\textit{{Source:}} {tex_escape(e['source'])}\\"]
        if e["settings"]:
            lines.append(rf"\textit{{Settings:}} {tex_escape(e['settings'])}\\")
        lines += [r"\begin{lstlisting}", e["text"], r"\end{lstlisting}"]
        if e["slots"]:
            lines.append(r"\begin{itemize}")
            lines += [rf"  \item \texttt{{\{{{tex_escape(n)}\}}}}: {tex_escape(d)}" for n, d in e["slots"]]
            lines.append(r"\end{itemize}")
        lines.append("")
    return "\n".join(lines)


def write_md(entries: list[dict]) -> str:
    lines = ["# Appendix D: Prompts", "",
             "<!-- Generated by scripts/make_prompt_appendix.py; do not edit by hand. -->", ""]
    group = None
    for e in entries:
        if e["group"] != group:
            group = e["group"]
            lines += [f"## {group}", ""]
        lines += [f"### {e['heading']}", "", f"*Source:* {e['source']}  "]
        if e["settings"]:
            lines.append(f"*Settings:* {e['settings']}")
        lines += ["", "```text", e["text"], "```", ""]
        if e["slots"]:
            lines += [f"- `{{{n}}}`: {d}" for n, d in e["slots"]] + [""]
    return "\n".join(lines)


def tex_structure_check(tex: str) -> list[str]:
    problems = []
    for env in ["lstlisting", "itemize"]:
        if tex.count(rf"\begin{{{env}}}") != tex.count(rf"\end{{{env}}}"):
            problems.append(f"unbalanced {env}")
    bodies = tex.split(r"\begin{lstlisting}")[1:]
    if any(r"\end{lstlisting}" in b.split("\n" + r"\end{lstlisting}")[0] for b in bodies):
        problems.append(r"\end{lstlisting} inside a listing")
    outside = "".join(b.split(r"\end{lstlisting}", 1)[1] for b in bodies)
    if any(ord(ch) > 127 for ch in outside):
        problems.append("non-ASCII text outside listings")
    return problems


if __name__ == "__main__":
    run_checks()
    entries = build_entries()
    for e in entries:
        if "\x00" in e["text"]:
            raise SystemExit(f"unreplaced slot in {e['heading']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex = write_tex(entries)
    (OUT_DIR / "prompts.tex").write_text(tex, encoding="utf-8")
    (OUT_DIR / "prompts.md").write_text(write_md(entries), encoding="utf-8")
    print("\n".join(CHECKS))
    print(f"{len(entries)} prompt blocks; tex structure: {tex_structure_check(tex) or 'ok'}")
    print(f"Saved {OUT_DIR / 'prompts.tex'} and {OUT_DIR / 'prompts.md'}")
