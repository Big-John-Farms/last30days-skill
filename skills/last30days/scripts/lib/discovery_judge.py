"""Discovery-mode LLM passes: the stage-1 topic judge and stage-2 angle writer.

Stage 1 (``judge_discovery_topics``) runs BEFORE enrichment: one batched call
names each nominated topic cluster, flags junk shapes, and scores 0-100
content-worthiness. Stage 2 (``generate_discovery_angles``) runs AFTER the
confidence floor: one batched call turns every surviving topic into a podcast
hook and an X-article hook.

Both passes share the same LLM-with-heuristic-fallback contract: they never
raise. No provider, a failed call, or a malformed payload logs a warning and
returns ``None``, and the caller falls back - stage 1 to the deterministic
``topic_shape`` heuristics, stage 2 to shipping topics without angles. A key
missing from a structurally valid response means the model skipped that row
and the caller falls back per-row.

Ranking and floor logic (velocity scores, ``passes_discovery_floor``,
``judge_blended_score`` and its tunables) stays in ``rerank``.
"""

from __future__ import annotations

import json

from typing import Callable, NamedTuple, TypeVar

from . import http, log, providers
from .rerank import _fenced_untrusted_content


class DiscoveryJudgeVerdict(NamedTuple):
    """One cluster's stage-1 judge verdict (see judge_discovery_topics)."""

    short_name: str
    junk_shape: bool
    worthiness: float | None


_DiscoveryParsedT = TypeVar("_DiscoveryParsedT")


def _run_discovery_llm_pass(
    provider: providers.ReasoningClient | None,
    model: str | None,
    entries: list[dict[str, str]],
    prompt_builder: Callable[[list[dict[str, str]]], str],
    parser: Callable[[dict], dict[str, _DiscoveryParsedT]],
    failure_label: str,
) -> dict[str, _DiscoveryParsedT] | None:
    """Shared skeleton for the batched discovery LLM passes (stage-1 judge,
    stage-2 angles): guard on provider/model/entries, one generate_json call,
    parse. Any expected failure logs ``failure_label`` and returns None so the
    caller falls back. Never raises."""
    if not (provider and model and entries):
        return None
    try:
        payload = provider.generate_json(model, prompt_builder(entries))
        # providers.extract_json returns whatever json.loads yields, so a model
        # emitting valid non-object JSON (top-level array, null, bare string)
        # reaches here as a non-dict. Raising inside the try converts it into
        # the standard logged fallback instead of an AttributeError in the
        # parser - the "Never raises" contract must hold for that shape too.
        if not isinstance(payload, dict):
            raise ValueError(
                f"expected JSON object from provider, got {type(payload).__name__}"
            )
        return parser(payload)
    except (ValueError, KeyError, json.JSONDecodeError, OSError, http.HTTPError) as exc:
        log.source_log(
            "Discover",
            f"{failure_label}: {type(exc).__name__}: {exc}",
            tty_only=False,
        )
        return None


def judge_discovery_topics(
    *,
    domain: str,
    entries: list[dict[str, str]],
    provider: providers.ReasoningClient | None,
    model: str | None,
) -> dict[str, DiscoveryJudgeVerdict] | None:
    """Stage-1 discovery judge: one batched LLM call naming and scoring the
    nominated topic clusters BEFORE enrichment.

    ``entries`` carries one dict per cluster: ``topic_id`` plus the leader's
    ``title`` and ``snippet`` (fenced as untrusted in the prompt). Returns a
    mapping keyed by ``topic_id``; a key missing from a structurally valid
    response means the model skipped that cluster and the caller falls back
    per-cluster to the topic_shape heuristics. Returns ``None`` when no
    provider is configured or the call failed outright, signalling a
    whole-pool heuristic fallback. Never raises.
    """
    return _run_discovery_llm_pass(
        provider,
        model,
        entries,
        lambda batch: _build_discovery_judge_prompt(domain, batch),
        _parse_discovery_judge_payload,
        "stage-1 judge failed, using heuristic topic names",
    )


def _build_discovery_judge_prompt(domain: str, entries: list[dict[str, str]]) -> str:
    entry_block = "\n".join(
        "\n".join([
            f"- topic_id: {entry['topic_id']}",
            f"  title: {str(entry.get('title') or '')[:220]}",
            f"  snippet: {str(entry.get('snippet') or '')[:420]}",
        ])
        for entry in entries
    )
    domain_label = domain or "global trending (no domain filter)"
    return (
        "You are the stage-1 topic judge for a trend-research tool. Each entry "
        "below is one candidate trending TOPIC (the leading post of a cluster "
        "of community chatter). The names you produce become search queries "
        "and podcast/article research briefs.\n\n"
        f"Domain being swept: {domain_label}\n\n"
        "For EVERY entry return one verdict:\n"
        "- short_name: a short SEARCHABLE topic name, 2-6 words. Name the "
        "underlying entities, launches, or debates (products, models, "
        "companies, events, controversies). Strip question/anecdote "
        "scaffolding. No punctuation, no quote characters. Names must be "
        "unique within this batch: when two entries cover different stories "
        "about the same entity, add a distinguishing word to each.\n"
        "- junk_shape: true when the post shape is not content-worthy: "
        "help-me/beginner questions, personal musings, am-I-the-only-one "
        "asks. Launches and entity-bearing news statements are not junk.\n"
        "- worthiness: 0-100. Would this make a good podcast or article "
        "topic for a tech-savvy audience? Judge novelty, stakes, "
        "specificity, and discussion-worthiness. 90+ is a story people "
        "would subscribe for; below 20 is filler.\n\n"
        "Return JSON only:\n"
        '{"topics": [{"topic_id": "id", "short_name": "2-6 word name", '
        '"junk_shape": false, "worthiness": 0-100}]}\n\n'
        f"{_fenced_untrusted_content(entry_block)}"
    )


# Defensive cap on judge-supplied names: they become search queries and the
# /last30days handoff, so a runaway (or adversarial) response never yields an
# unbounded string. Mirrors the pre-judge 96-char title cap.
_JUDGE_NAME_MAX_CHARS = 96

# Unified trailing-punctuation charset for word-boundary truncation: judge
# names and angle sentences share it so the strip sets cannot drift.
_TRUNCATE_STRIP_CHARS = " \"'`.,;:!?-"


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Cap ``text`` at ``max_chars``, cutting back to a word boundary and
    stripping trailing punctuation. Text within the cap passes through
    untouched."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(_TRUNCATE_STRIP_CHARS)


def _parse_discovery_judge_payload(payload: dict) -> dict[str, DiscoveryJudgeVerdict]:
    verdicts: dict[str, DiscoveryJudgeVerdict] = {}
    for row in payload.get("topics") or []:
        if not isinstance(row, dict):
            continue
        topic_id = str(row.get("topic_id") or "").strip()
        name = " ".join(str(row.get("short_name") or "").split())
        name = name.strip(_TRUNCATE_STRIP_CHARS)
        name = _truncate_at_word(name, _JUDGE_NAME_MAX_CHARS)
        if not topic_id or not name:
            # Missing identity or name: treat the row as absent so the caller
            # falls back to the deterministic heuristics for that cluster.
            continue
        raw_worthiness = row.get("worthiness")
        worthiness: float | None
        try:
            worthiness = (
                None if isinstance(raw_worthiness, bool)
                else max(0.0, min(100.0, float(raw_worthiness)))
            )
        except (TypeError, ValueError):
            worthiness = None
        verdicts[topic_id] = DiscoveryJudgeVerdict(
            short_name=name,
            junk_shape=bool(row.get("junk_shape")),
            worthiness=worthiness,
        )
    return verdicts


class DiscoveryAngles(NamedTuple):
    """One surfaced topic's stage-2 content hooks (see
    generate_discovery_angles). Either field may be None when the model
    returned nothing usable for that medium."""

    podcast_angle: str | None
    x_article_angle: str | None


def generate_discovery_angles(
    *,
    domain: str,
    entries: list[dict[str, str]],
    provider: providers.ReasoningClient | None,
    model: str | None,
) -> dict[str, DiscoveryAngles] | None:
    """Stage-2 discovery angle pass: one batched LLM call AFTER the floor,
    turning every surfaced topic into a podcast hook and an X-article hook.

    ``entries`` carries one dict per floor survivor: ``topic_id`` plus the
    topic's ``name`` and its strongest evidence (``titles``, ``top_comment``,
    ``engagement``, fenced as untrusted in the prompt). Returns a mapping
    keyed by ``topic_id``; a key missing from a structurally valid response
    means the model skipped that topic and it ships with None angles.
    Returns ``None`` when no provider is configured or the call failed
    outright - every topic then ships without angles. Never raises.
    """
    return _run_discovery_llm_pass(
        provider,
        model,
        entries,
        lambda batch: _build_discovery_angle_prompt(domain, batch),
        _parse_discovery_angle_payload,
        "stage-2 angle pass failed, topics ship without content angles",
    )


def _build_discovery_angle_prompt(domain: str, entries: list[dict[str, str]]) -> str:
    entry_block = "\n".join(
        "\n".join([
            f"- topic_id: {entry['topic_id']}",
            f"  name: {str(entry.get('name') or '')[:96]}",
            f"  evidence_titles: {str(entry.get('titles') or '')[:420]}",
            f"  top_comment: {str(entry.get('top_comment') or '')[:340]}",
            f"  engagement: {str(entry.get('engagement') or '')[:200]}",
        ])
        for entry in entries
    )
    domain_label = domain or "global trending (no domain filter)"
    return (
        "You are the stage-2 content-angle writer for a trend-research tool. "
        "The reader is a podcaster who also writes X articles. Each entry "
        "below is one CONFIRMED trending topic with its strongest evidence.\n\n"
        f"Domain being swept: {domain_label}\n\n"
        "For EVERY entry return two hooks, one sentence each:\n"
        "- podcast_angle: a discussion hook for a podcast segment - the "
        "question or tension a host would talk through on air. Frame it as "
        "something to argue about or unpack, never a summary.\n"
        "- x_article_angle: a written-take hook for an X article - the "
        "claim, thesis, or listicle-able angle the piece would open with.\n\n"
        "Ground each hook in the evidence shown; never invent facts.\n\n"
        "Return JSON only:\n"
        '{"topics": [{"topic_id": "id", "podcast_angle": "one sentence", '
        '"x_article_angle": "one sentence"}]}\n\n'
        f"{_fenced_untrusted_content(entry_block)}"
    )


# Defensive cap on angle sentences: they render verbatim on trend cards, so a
# runaway (or adversarial) response never yields an unbounded string.
_ANGLE_MAX_CHARS = 200


def _sanitized_angle(raw: object) -> str | None:
    """One whitespace-collapsed, length-capped angle sentence, or None for
    anything unusable. Non-strings are rejected outright, never coerced."""
    if not isinstance(raw, str):
        return None
    text = _truncate_at_word(" ".join(raw.split()), _ANGLE_MAX_CHARS)
    return text or None


def _parse_discovery_angle_payload(payload: dict) -> dict[str, DiscoveryAngles]:
    angles: dict[str, DiscoveryAngles] = {}
    for row in payload.get("topics") or []:
        if not isinstance(row, dict):
            continue
        topic_id = str(row.get("topic_id") or "").strip()
        podcast = _sanitized_angle(row.get("podcast_angle"))
        article = _sanitized_angle(row.get("x_article_angle"))
        if not topic_id or (podcast is None and article is None):
            # Missing identity or no usable hook at all: treat the row as
            # absent so the topic ships without angles.
            continue
        angles[topic_id] = DiscoveryAngles(
            podcast_angle=podcast,
            x_article_angle=article,
        )
    return angles
