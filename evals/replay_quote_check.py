"""Replay the deterministic quote check over the RAG-audit drafts, offline.

Runs ``src.backend.quote_check.check_quotations`` over the production drafts and
the replayed drafts captured by the 2026-10-07 RAG audit, and prints what the
new check would verify, exempt and delete. It only reads local files: no LM
Studio, no Elasticsearch, no network. It imports nothing from
``lms_agentic_search`` (which needs langgraph and the service singletons), only
the pure ``quote_check`` module and the standard library.

Usage::

    PYTHONPATH=. <venv>/bin/python evals/replay_quote_check.py \\
        <path-to>/rag_audit_20261007/p2_raw

Inputs in the ``p2_raw`` directory:

* ``p18_verify_request.json`` and ``p07_verify_request.json``: the captured
  verifier requests. ``messages[0].content`` is the verifier prompt, from which
  :func:`parse_verifier_prompt` recovers the numbered sources and the
  production draft.
* ``E*.json``: replayed drafts. The draft is the file's ``content`` field (or,
  for a raw chat-completion response, ``choices[0].message.content``).

Questions (the exempt_question haystack): P18 ``Who is S.D. Goitein``; P07
``מה ידוע מן הגניזה על תשובות הגאונים?``.

Pairing rule for ``E*.json`` drafts. Every E draft is a replay of P18 or P07
and is checked against that prompt's sources and question. The first rule
that decides wins, and the printed report names the rule used:

1. ``--pair NAME=p18`` (or ``=p07``) on the command line, where ``NAME`` is the
   file name with or without ``.json``.
2. The file itself: its name, or a top-level string value other than
   ``content`` (e.g. ``"replay_of": "P18"``), names exactly one of the prompts
   as a whole token (``P18``, ``p07_...``) or contains exactly one of the two
   questions.
3. The draft's script: when more than half of its letters are Hebrew it pairs
   with P07 (the Hebrew question), otherwise with P18.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.backend.quote_check import QUOTE_CHECK_COUNT_KEYS, check_quotations, sentences_overlapping

PROMPT_FILES: Dict[str, str] = {
    "p18": "p18_verify_request.json",
    "p07": "p07_verify_request.json",
}
QUESTIONS: Dict[str, str] = {
    "p18": "Who is S.D. Goitein",
    "p07": "מה ידוע מן הגניזה על תשובות הגאונים?",
}
# The pairing rule's script fallback: P07 is the Hebrew question.
HEBREW_PROMPT = "p07"
LATIN_PROMPT = "p18"
GRAPH_CITATION_PREFIX = "Neo4j knowledge graph record"

_SOURCE_HEAD_REGEX = re.compile(r"^\[SOURCE (\d+)\] ?")
_QUOTEABLE_REGEX = re.compile(
    r"Original page text \(quoteable\): (.*?)(?:\nShelf marks cited in this source:|\Z)",
    re.S,
)
_PROMPT_TOKEN_REGEX = re.compile(r"(?<![a-z0-9])(p18|p07)(?![0-9])", re.I)
_HEBREW_LETTER_REGEX = re.compile(r"[א-ת]")


@dataclass
class Prompt:
    """One captured verifier request, parsed.

    :param key: ``p18`` or ``p07``.
    :param sources: Source dictionaries shaped like ``build_verification_sources``
        output (graph sources carry no ``quoteable_text`` key).
    :param draft: The production draft the verifier was asked to check.
    """

    key: str
    sources: List[Dict[str, Any]]
    draft: str


@dataclass
class DraftCase:
    """One draft to replay.

    :param name: Display name (``prod-P18``, or the E file's stem).
    :param prompt_key: The prompt whose sources and question it is checked
        against.
    :param rule: How the draft was paired with that prompt.
    :param draft: The draft text.
    """

    name: str
    prompt_key: str
    rule: str
    draft: str


def parse_verifier_prompt(prompt: str) -> Tuple[List[Dict[str, Any]], str]:
    """Split a captured verifier prompt into its sources and its draft.

    The prompt holds ``SOURCE CHUNKS:`` followed by ``[SOURCE N] <citation>``
    blocks separated by blank lines, then ``DRAFT ANSWER:`` and the draft,
    which ends where the instruction ``The type must be`` begins.

    :param prompt: The verifier prompt (``messages[0].content``).
    :returns: The parsed sources (``source_number``, ``citation``,
        ``quoteable_text``, ``evidence_text``) and the draft.
    :rtype: Tuple[List[Dict[str, Any]], str]
    :raises ValueError: If the prompt lacks the ``SOURCE CHUNKS`` or
        ``DRAFT ANSWER`` sections.
    """
    if "SOURCE CHUNKS:\n" not in prompt or "\n\nDRAFT ANSWER:\n" not in prompt:
        raise ValueError("not a verifier prompt: missing SOURCE CHUNKS or DRAFT ANSWER")
    sources_text, rest = prompt.split("SOURCE CHUNKS:\n", 1)[1].split("\n\nDRAFT ANSWER:\n", 1)
    draft = rest.split("\n\nThe type must be", 1)[0]
    sources: List[Dict[str, Any]] = []
    for block in re.split(r"\n\n(?=\[SOURCE \d+\])", sources_text.strip()):
        head, _, body = block.partition("\n")
        number = _SOURCE_HEAD_REGEX.match(head)
        quoteable = _QUOTEABLE_REGEX.search(body)
        sources.append({
            "source_number": int(number.group(1)) if number else len(sources) + 1,
            "citation": _SOURCE_HEAD_REGEX.sub("", head),
            "quoteable_text": quoteable.group(1) if quoteable else "",
            "evidence_text": body,
        })
    return sources, draft


def as_verification_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make parsed sources match ``build_verification_sources`` output.

    A graph source (citation starting ``Neo4j knowledge graph record``) has no
    ``quoteable_text`` key there, so it can never verify a quotation and its
    evidence text joins the title haystack. Bibliography sources keep the key,
    even when it is empty.

    :param sources: Sources from :func:`parse_verifier_prompt`.
    :returns: New source dictionaries with graph sources' ``quoteable_text``
        removed.
    :rtype: List[Dict[str, Any]]
    """
    converted: List[Dict[str, Any]] = []
    for source in sources:
        source = dict(source)
        if str(source.get("citation") or "").startswith(GRAPH_CITATION_PREFIX):
            source.pop("quoteable_text", None)
        converted.append(source)
    return converted


def load_json(path: Path) -> Any:
    """Read one JSON file.

    :param path: File to read.
    :returns: The decoded JSON value.
    :rtype: Any
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt(directory: Path, key: str) -> Prompt:
    """Load and parse one captured verifier request.

    :param directory: The ``p2_raw`` directory.
    :param key: ``p18`` or ``p07``.
    :returns: The parsed prompt.
    :rtype: Prompt
    """
    request = load_json(directory / PROMPT_FILES[key])
    sources, draft = parse_verifier_prompt(request["messages"][0]["content"])
    return Prompt(key=key, sources=as_verification_sources(sources), draft=draft)


def draft_content(record: Any) -> Optional[str]:
    """Extract the draft from an ``E*.json`` record.

    :param record: Decoded E file.
    :returns: The top-level ``content`` string, else a chat-completion
        response's ``choices[0].message.content``, else ``None``.
    :rtype: Optional[str]
    """
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("content"), str):
        return record["content"]
    choices = record.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content
    return None


def hebrew_letter_share(text: str) -> float:
    """Return the share of letters in ``text`` that are Hebrew.

    :param text: Draft text.
    :returns: Hebrew letters divided by all letters (0.0 when there are none).
    :rtype: float
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if _HEBREW_LETTER_REGEX.match(char)) / len(letters)


def prompt_named_in_file(name: str, record: Dict[str, Any]) -> Optional[str]:
    """Find the prompt an E file names in its file name or its own fields.

    :param name: The E file's stem.
    :param record: Decoded E file.
    :returns: ``p18`` or ``p07`` when the file name and the top-level string
        values other than ``content`` name exactly one prompt (as a whole
        token, or by containing its question), otherwise ``None``.
    :rtype: Optional[str]
    """
    texts = [name] + [
        value for key, value in record.items() if key != "content" and isinstance(value, str)
    ]
    named: Set[str] = set()
    for text in texts:
        named.update(match.lower() for match in _PROMPT_TOKEN_REGEX.findall(text))
        named.update(prompt for prompt, question in QUESTIONS.items() if question in text)
    return named.pop() if len(named) == 1 else None


def pair_draft(name: str, record: Dict[str, Any], draft: str, overrides: Dict[str, str]) -> Tuple[str, str]:
    """Pair an E draft with P18 or P07 by the module's documented rule.

    :param name: The E file's stem.
    :param record: Decoded E file.
    :param draft: The draft text.
    :param overrides: ``--pair`` values keyed by file stem.
    :returns: The prompt key and the rule that decided it.
    :rtype: Tuple[str, str]
    """
    if name in overrides:
        return overrides[name], "--pair"
    named = prompt_named_in_file(name, record)
    if named:
        return named, "file name or field"
    share = hebrew_letter_share(draft)
    prompt_key = HEBREW_PROMPT if share > 0.5 else LATIN_PROMPT
    return prompt_key, f"script ({share:.0%} Hebrew letters)"


def parse_pair(value: str) -> Tuple[str, str]:
    """Parse one ``--pair NAME=p18|p07`` value (an argparse ``type``).

    :param value: Raw ``--pair`` argument.
    :returns: The E file stem and the prompt key.
    :rtype: Tuple[str, str]
    :raises argparse.ArgumentTypeError: On a malformed value.
    """
    name, _, key = value.partition("=")
    key = key.strip().lower()
    if not name.strip() or key not in PROMPT_FILES:
        raise argparse.ArgumentTypeError(f"expected NAME=p18 or NAME=p07, got {value!r}")
    return Path(name.strip()).stem, key


def collect_cases(directory: Path, prompts: Dict[str, Prompt], overrides: Dict[str, str]) -> List[DraftCase]:
    """List the production drafts and every ``E*.json`` draft.

    :param directory: The ``p2_raw`` directory.
    :param prompts: Parsed prompts keyed ``p18``/``p07``.
    :param overrides: ``--pair`` values keyed by file stem.
    :returns: Draft cases, production drafts first, then E files by name.
    :rtype: List[DraftCase]
    """
    cases = [
        DraftCase(name=f"prod-{key.upper()}", prompt_key=key, rule="captured prompt", draft=prompt.draft)
        for key, prompt in prompts.items()
    ]
    for path in sorted(directory.glob("E*.json")):
        record = load_json(path)
        draft = draft_content(record)
        if draft is None:
            print(f"SKIP {path.name}: no content field", file=sys.stderr)
            continue
        prompt_key, rule = pair_draft(path.stem, record, draft, overrides)
        cases.append(DraftCase(name=path.stem, prompt_key=prompt_key, rule=rule, draft=draft))
    return cases


def replay_case(case: DraftCase, prompt: Prompt) -> Dict[str, int]:
    """Check one draft and print its counts and unverified items.

    :param case: The draft to check.
    :param prompt: The prompt it is paired with.
    :returns: The counts, plus ``sentences_deleted`` (sentences the check
        would delete).
    :rtype: Dict[str, int]
    """
    result = check_quotations(case.draft, prompt.sources, QUESTIONS[case.prompt_key])
    counts = dict(result.counts)
    counts["sentences_deleted"] = len(sentences_overlapping(case.draft, result.unverified_offsets()))
    print(f"== {case.name}  [{case.prompt_key.upper()}, paired by {case.rule}]")
    print("   " + "  ".join(f"{key}={value}" for key, value in counts.items()))
    for start, _end, text, kind in result.unverified_spans:
        print(f"   UNVERIFIED {kind} @{start}: {text}")
    return counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (``None`` reads ``sys.argv``).
    :returns: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("p2_raw", type=Path, help="Directory holding p18/p07_verify_request.json and E*.json")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        type=parse_pair,
        metavar="NAME=p18|p07",
        help="Pair an E file with a prompt explicitly (repeatable); overrides the automatic rule",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Replay the quote check over every audit draft and print a report.

    :param argv: Arguments (``None`` reads ``sys.argv``).
    :returns: Process exit code (0 on success, 2 when an input is missing).
    :rtype: int
    """
    args = parse_args(argv)
    directory: Path = args.p2_raw
    missing = [name for name in PROMPT_FILES.values() if not (directory / name).is_file()]
    if missing:
        print(f"Missing in {directory}: {', '.join(missing)}", file=sys.stderr)
        return 2
    overrides: Dict[str, str] = dict(args.pair)
    prompts = {key: load_prompt(directory, key) for key in PROMPT_FILES}
    cases = collect_cases(directory, prompts, overrides)

    totals: Dict[str, int] = dict.fromkeys((*QUOTE_CHECK_COUNT_KEYS, "sentences_deleted"), 0)
    for case in cases:
        for key, value in replay_case(case, prompts[case.prompt_key]).items():
            totals[key] += value
    print(f"TOTAL drafts={len(cases)}  " + "  ".join(f"{key}={value}" for key, value in totals.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
