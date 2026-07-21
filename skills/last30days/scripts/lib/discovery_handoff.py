"""File contracts for the three-command host-judged discovery protocol.

Leg 1 (``--discover --nominate-only``) writes the nominations bundle: the
FULL judge pool, each nomination with its complete seed item set, serialized
losslessly so leg 2 can recompute floor/velocity/entity-token disambiguation
exactly as an in-memory run would. Leg 2 (``--discover --judgments <file>``)
reads host judgments (names/junk/worthiness) bound to the bundle by
bundle_id. Leg 3 (``--discover --finalize [--angles <file>]``) applies
host-written content angles.

This module owns the three contracts - bundle writer/reader, judgments
reader, angles reader - plus the host-facing digest and the post-judgment
name-collision resolver. Readers are strict at the top level (typed
``HandoffContractError``, mapped to exit 2 by the CLI layer) and lenient per
row: a malformed or omitted row falls back to the bundle's heuristics rather
than failing the run.
"""

from __future__ import annotations

import json
import secrets
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import env, log, pipeline, rerank, schema


# How long a nominations bundle stays valid. Deliberately a module constant
# and NOT the LAST30DAYS_REPORT_CACHE_TTL_SECONDS env knob: a user who
# lowered the report-cache TTL for drill freshness must not shrink the
# window a host has to author judgments.
DISCOVERY_HANDOFF_TTL_SECONDS = 3600.0

NOMINATIONS_BUNDLE_FILENAME = "discover-nominations.json"
PENDING_REPORT_FILENAME = "discover-pending.json"

_VALID_TIERS = ("deep", "shallow")

_RESWEEP_REMEDY = "Run a fresh `--discover --nominate-only` re-sweep."

# Defensive caps on host-supplied text, ported from the retired engine-judge
# pass: names become search queries and the /last30days handoff, angles
# render verbatim on trend cards, so a runaway (or adversarial) value never
# yields an unbounded string.
_NAME_MAX_CHARS = 96
_ANGLE_MAX_CHARS = 200

# Unified trailing-punctuation charset for word-boundary truncation: names
# and angle sentences share it so the strip sets cannot drift.
_TRUNCATE_STRIP_CHARS = " \"'`.,;:!?-"

# Digest evidence caps: the surface the engine judge used to see per
# nomination (leader title, leader snippet, strongest community comment).
_DIGEST_TITLE_MAX_CHARS = 220
_DIGEST_SNIPPET_MAX_CHARS = 420
_DIGEST_COMMENT_MAX_CHARS = 340


class HandoffContractError(Exception):
    """A handoff file failed its contract: unreadable, invalid JSON, wrong
    shape or schema version, stale, or not bound to the current bundle.
    The CLI layer maps this to exit code 2."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class PoolEntry:
    """One judge-pool nomination as handed to the bundle writer (leg 1).

    ``heuristic_name`` and ``heuristic_junk`` are the deterministic
    topic_shape fallbacks, kept alongside the nomination so leg 2 can fill
    any row the host omitted without re-deriving them.
    """

    nomination: pipeline.Nomination
    cluster_id: str
    heuristic_name: str
    heuristic_junk: bool


@dataclass(frozen=True)
class BundleNomination:
    """One nomination read back from a bundle, with its stable id."""

    nomination_id: str
    nomination: pipeline.Nomination
    cluster_id: str
    heuristic_name: str
    heuristic_junk: bool
    sources: list[str]
    engagement_by_source: dict[str, dict[str, float | int]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class NominationsBundle:
    """A parsed leg-1 nominations bundle (also returned by the writer)."""

    schema_version: str
    bundle_id: str
    generated_at: str
    from_date: str
    to_date: str
    domain: str
    tier: str
    enrichment_source_boundary: list[str] | None
    requested_sources: list[str] | None
    lookback_days: int
    nominations: list[BundleNomination]
    path: Path | None = None


@dataclass(frozen=True)
class HostJudgment:
    """One host verdict row. ``None`` on any field means the host left it
    absent for that row and the caller falls back to the bundle's heuristic
    value (name/junk) or to no worthiness signal."""

    name: str | None
    junk: bool | None
    worthiness: int | None


# The per-row-absent marker: what ``judgment_for`` returns for a nomination
# the host omitted entirely. Every field falls back to the bundle heuristics.
ROW_ABSENT = HostJudgment(name=None, junk=None, worthiness=None)


@dataclass(frozen=True)
class HostAngles:
    """One host-written angle row; either field may be absent."""

    podcast: str | None
    x_article: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _warn(message: str) -> None:
    log.source_log("Discover", message, tty_only=False)


def handoff_state_dir(
    save_dir: str | Path | None,
    config_dir: Path | None,
) -> Path | None:
    """Resolve the handoff state directory: ``save_dir`` when provided, else
    the config dir (mirrors the report-cache convention in last30days.py).
    Both are accepted as arguments so this module never imports the CLI
    layer above it. Returns None when neither location is available."""
    if save_dir:
        return Path(save_dir).expanduser().resolve()
    if config_dir is not None:
        return Path(config_dir)
    return None


def nominations_bundle_path(state_dir: str | Path) -> Path:
    """The nominations bundle file inside a handoff state directory."""
    return Path(state_dir) / NOMINATIONS_BUNDLE_FILENAME


def pending_report_path(state_dir: str | Path) -> Path:
    """The leg-2 pending-report file inside a handoff state directory."""
    return Path(state_dir) / PENDING_REPORT_FILENAME


def _bundle_search_paths(
    save_dir: str | Path | None,
    config_dir: Path | None,
) -> list[Path]:
    """Candidate bundle locations in lookup order: save-dir, then config dir."""
    paths: list[Path] = []
    if save_dir:
        paths.append(nominations_bundle_path(Path(save_dir).expanduser().resolve()))
    if config_dir is not None:
        candidate = nominations_bundle_path(Path(config_dir))
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _searched_lines(searched: list[Path]) -> str:
    if not searched:
        return "  (no --save-dir and no config directory available)"
    return "\n".join(f"  - {path}" for path in searched)


def write_nominations_bundle(
    entries: Sequence[PoolEntry],
    *,
    domain: str,
    tier: str,
    from_date: str,
    to_date: str,
    lookback_days: int,
    enrichment_source_boundary: list[str] | None,
    requested_sources: list[str] | None,
    save_dir: str | Path | None = None,
    config_dir: Path | None = None,
) -> NominationsBundle:
    """Write the leg-1 nominations bundle and return its parsed form.

    Nomination ids are assigned ``n1, n2, ...`` in pool order. The leg-1
    invocation context (enrichment source boundary, requested discovery
    sources, lookback days) rides along so leg 2 resumes with identical
    settings. ``None`` boundaries are preserved as null - "no boundary" and
    "empty boundary" are different contracts.
    """
    if tier not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {_VALID_TIERS}, got {tier!r}")
    state_dir = handoff_state_dir(save_dir, config_dir)
    if state_dir is None:
        raise HandoffContractError(
            "No handoff location available to write the nominations bundle: "
            "pass --save-dir or configure ~/.config/last30days/."
        )
    bundle_id = secrets.token_hex(8)
    generated_at = _utc_now()

    rows: list[dict[str, Any]] = []
    nominations: list[BundleNomination] = []
    for index, entry in enumerate(entries, start=1):
        nomination_id = f"n{index}"
        sources = sorted({item.source for item in entry.nomination.items})
        engagement = pipeline._discovery_engagement(entry.nomination.items)
        rows.append({
            "id": nomination_id,
            "cluster_id": entry.cluster_id,
            "heuristic_name": entry.heuristic_name,
            "heuristic_junk": bool(entry.heuristic_junk),
            "sources": sources,
            "engagement_by_source": engagement,
            "nomination": schema.nomination_to_dict(entry.nomination),
        })
        nominations.append(BundleNomination(
            nomination_id=nomination_id,
            nomination=entry.nomination,
            cluster_id=entry.cluster_id,
            heuristic_name=entry.heuristic_name,
            heuristic_junk=bool(entry.heuristic_junk),
            sources=sources,
            engagement_by_source=engagement,
        ))

    payload = {
        "schema_version": schema.DISCOVERY_NOMINATIONS_SCHEMA_VERSION,
        "kind": schema.DISCOVERY_NOMINATIONS_KIND,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "from_date": from_date,
        "to_date": to_date,
        "domain": domain,
        "tier": tier,
        "context": {
            "enrichment_source_boundary": (
                list(enrichment_source_boundary)
                if enrichment_source_boundary is not None
                else None
            ),
            "requested_sources": (
                list(requested_sources) if requested_sources is not None else None
            ),
            "lookback_days": int(lookback_days),
        },
        "nominations": rows,
    }
    path = nominations_bundle_path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return NominationsBundle(
        schema_version=schema.DISCOVERY_NOMINATIONS_SCHEMA_VERSION,
        bundle_id=bundle_id,
        generated_at=generated_at,
        from_date=from_date,
        to_date=to_date,
        domain=domain,
        tier=tier,
        enrichment_source_boundary=(
            list(enrichment_source_boundary)
            if enrichment_source_boundary is not None
            else None
        ),
        requested_sources=(
            list(requested_sources) if requested_sources is not None else None
        ),
        lookback_days=int(lookback_days),
        nominations=nominations,
        path=path,
    )


def read_nominations_bundle(
    *,
    save_dir: str | Path | None = None,
    config_dir: Path | None = None,
) -> NominationsBundle:
    """Locate and parse the nominations bundle for legs 2 and 3.

    Lookup order is save-dir then config dir. Raises HandoffContractError
    (naming every searched location and the re-sweep remedy) when no bundle
    exists, and for any top-level contract violation in the file found.
    """
    searched = _bundle_search_paths(save_dir, config_dir)
    path = next((candidate for candidate in searched if candidate.exists()), None)
    if path is None:
        raise HandoffContractError(
            "No discovery nominations bundle found. Searched:\n"
            f"{_searched_lines(searched)}\n{_RESWEEP_REMEDY}"
        )
    return _parse_bundle_file(path)


def _parse_bundle_file(path: Path) -> NominationsBundle:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HandoffContractError(
            f"Could not read nominations bundle {path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HandoffContractError(
            f"Nominations bundle {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HandoffContractError(
            f"Nominations bundle {path} must be a top-level JSON object, "
            f"got {type(payload).__name__}."
        )
    version = payload.get("schema_version")
    if version != schema.DISCOVERY_NOMINATIONS_SCHEMA_VERSION:
        raise HandoffContractError(
            f"Nominations bundle {path} has schema version {version!r}; this "
            f"build reads {schema.DISCOVERY_NOMINATIONS_SCHEMA_VERSION!r}. "
            f"{_RESWEEP_REMEDY}"
        )
    kind = payload.get("kind")
    if kind != schema.DISCOVERY_NOMINATIONS_KIND:
        raise HandoffContractError(
            f"Nominations bundle {path} has kind {kind!r}; expected "
            f"{schema.DISCOVERY_NOMINATIONS_KIND!r}. {_RESWEEP_REMEDY}"
        )
    bundle_id = str(payload.get("bundle_id") or "")
    if not bundle_id:
        raise HandoffContractError(
            f"Nominations bundle {path} is missing its bundle_id; judgments "
            f"cannot bind to it. {_RESWEEP_REMEDY}"
        )
    generated_at = payload.get("generated_at")
    if not env.is_timestamp_fresh(generated_at, DISCOVERY_HANDOFF_TTL_SECONDS):
        raise HandoffContractError(
            f"Nominations bundle {path} is stale (generated_at="
            f"{generated_at!r}, TTL {int(DISCOVERY_HANDOFF_TTL_SECONDS)}s): "
            f"the momentum window it captured has moved on. {_RESWEEP_REMEDY}"
        )

    context = payload.get("context") or {}
    boundary = context.get("enrichment_source_boundary")
    requested = context.get("requested_sources")
    try:
        lookback_days = int(context.get("lookback_days") or 30)
    except (TypeError, ValueError):
        lookback_days = 30

    nominations: list[BundleNomination] = []
    for position, row in enumerate(payload.get("nominations") or [], start=1):
        # Lenient per row: the bundle is engine-written, but one corrupted
        # row must not discard the rest of the pool.
        if not isinstance(row, dict):
            _warn(
                f"skipping malformed nomination row {position} in "
                f"{path.name} (not an object)"
            )
            continue
        try:
            nomination = pipeline.Nomination(
                **schema.nomination_kwargs_from_dict(row.get("nomination") or {})
            )
        except (KeyError, TypeError, ValueError) as exc:
            _warn(
                f"skipping unparseable nomination row {position} in "
                f"{path.name}: {type(exc).__name__}: {exc}"
            )
            continue
        engagement_raw = row.get("engagement_by_source")
        engagement = {
            str(source): dict(metrics)
            for source, metrics in (
                engagement_raw.items() if isinstance(engagement_raw, dict) else ()
            )
            if isinstance(metrics, dict)
        }
        nominations.append(BundleNomination(
            nomination_id=str(row.get("id") or f"n{position}"),
            nomination=nomination,
            cluster_id=str(row.get("cluster_id") or ""),
            heuristic_name=str(row.get("heuristic_name") or ""),
            heuristic_junk=bool(row.get("heuristic_junk")),
            sources=[str(source) for source in row.get("sources") or []],
            engagement_by_source=engagement,
        ))

    return NominationsBundle(
        schema_version=str(version),
        bundle_id=bundle_id,
        generated_at=str(generated_at or ""),
        from_date=str(payload.get("from_date") or ""),
        to_date=str(payload.get("to_date") or ""),
        domain=str(payload.get("domain") or ""),
        tier=str(payload.get("tier") or "deep"),
        enrichment_source_boundary=(
            [str(source) for source in boundary]
            if isinstance(boundary, list) else None
        ),
        requested_sources=(
            [str(source) for source in requested]
            if isinstance(requested, list) else None
        ),
        lookback_days=lookback_days,
        nominations=nominations,
        path=path,
    )


def _load_host_file(path: str | Path, label: str) -> dict[str, Any]:
    """Load a host-authored handoff file with strict top-level checks."""
    file_path = Path(path).expanduser()
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HandoffContractError(
            f"Could not read {label} file {file_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HandoffContractError(
            f"{label.capitalize()} file {file_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HandoffContractError(
            f"{label.capitalize()} file {file_path} must be a top-level JSON "
            f"object, got {type(payload).__name__}."
        )
    return payload


def _require_bundle_binding(
    payload: dict[str, Any],
    bundle: NominationsBundle,
    *,
    label: str,
    save_dir: str | Path | None,
    config_dir: Path | None,
) -> None:
    """Enforce bundle-id binding between a host file and the current bundle."""
    file_bundle_id = str(payload.get("bundle_id") or "")
    if file_bundle_id == bundle.bundle_id:
        return
    searched = _bundle_search_paths(save_dir, config_dir)
    if not searched and bundle.path is not None:
        searched = [bundle.path]
    raise HandoffContractError(
        f"The {label} file is bound to bundle_id {file_bundle_id!r} but the "
        f"current nominations bundle is {bundle.bundle_id!r}. Bundle "
        "locations searched:\n"
        f"{_searched_lines(searched)}\n{_RESWEEP_REMEDY}"
    )


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Cap ``text`` at ``max_chars``, cutting back to a word boundary and
    stripping trailing punctuation. Text within the cap passes through
    untouched."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(_TRUNCATE_STRIP_CHARS)


def _sanitized_name(raw: object) -> str | None:
    """One whitespace-collapsed, punctuation-stripped, length-capped topic
    name, or None for anything unusable (non-strings, and names that
    sanitize to empty - e.g. emoji-only - count as per-row-absent)."""
    if not isinstance(raw, str):
        return None
    name = " ".join(raw.split()).strip(_TRUNCATE_STRIP_CHARS)
    name = _truncate_at_word(name, _NAME_MAX_CHARS)
    if not any(char.isalnum() for char in name):
        return None
    return name


def _sanitized_angle(raw: object) -> str | None:
    """One whitespace-collapsed, length-capped angle sentence, or None for
    anything unusable. Non-strings are rejected outright, never coerced."""
    if not isinstance(raw, str):
        return None
    text = _truncate_at_word(" ".join(raw.split()), _ANGLE_MAX_CHARS)
    return text or None


def _clamped_worthiness(raw: object) -> int | None:
    """Worthiness clamped to 0-100 integers; anything non-numeric is absent."""
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0, min(100, round(value)))


def read_judgments(
    path: str | Path,
    bundle: NominationsBundle,
    *,
    save_dir: str | Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, HostJudgment]:
    """Read the host judgments file for leg 2, keyed by nomination id.

    Strict at the top level (readable, valid JSON object, ``judgments`` list,
    bundle_id bound to ``bundle``), lenient per row: an unknown id is warned
    and ignored, a missing/unusable name or junk field is per-row-absent, and
    worthiness is clamped to 0-100 integers. Nominations with no row at all
    are simply missing from the mapping - use ``judgment_for`` to get the
    ROW_ABSENT marker for them.
    """
    payload = _load_host_file(path, "judgments")
    _require_bundle_binding(
        payload, bundle, label="judgments", save_dir=save_dir, config_dir=config_dir,
    )
    rows = payload.get("judgments")
    if not isinstance(rows, list):
        raise HandoffContractError(
            f"Judgments file {path} must carry a top-level \"judgments\" list."
        )
    known = {entry.nomination_id for entry in bundle.nominations}
    judgments: dict[str, HostJudgment] = {}
    for row in rows:
        if not isinstance(row, dict):
            _warn("skipping malformed judgments row (not an object)")
            continue
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            _warn("skipping judgments row with no nomination id")
            continue
        if row_id not in known:
            _warn(f"ignoring judgment for unknown nomination id {row_id!r}")
            continue
        judgments[row_id] = HostJudgment(
            name=_sanitized_name(row.get("name")),
            junk=bool(row["junk"]) if "junk" in row else None,
            worthiness=_clamped_worthiness(row.get("worthiness")),
        )
    return judgments


def judgment_for(
    judgments: dict[str, HostJudgment],
    nomination_id: str,
) -> HostJudgment:
    """The host's verdict for one nomination, or ROW_ABSENT when the host
    omitted the row (caller falls back to the bundle's heuristic name/junk)."""
    return judgments.get(nomination_id, ROW_ABSENT)


def read_angles(
    path: str | Path | None,
    bundle: NominationsBundle,
    *,
    save_dir: str | Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, HostAngles]:
    """Read the host angles file for leg 3, keyed by nomination id.

    A missing angles file is legal: ``path=None`` returns an empty mapping
    and every topic ships without angles. When a path is given the same
    strict-top-level / lenient-per-row rules as judgments apply; angle
    sentences are word-boundary capped at 200 chars.
    """
    if path is None:
        return {}
    payload = _load_host_file(path, "angles")
    _require_bundle_binding(
        payload, bundle, label="angles", save_dir=save_dir, config_dir=config_dir,
    )
    rows = payload.get("angles")
    if not isinstance(rows, list):
        raise HandoffContractError(
            f"Angles file {path} must carry a top-level \"angles\" list."
        )
    known = {entry.nomination_id for entry in bundle.nominations}
    angles: dict[str, HostAngles] = {}
    for row in rows:
        if not isinstance(row, dict):
            _warn("skipping malformed angles row (not an object)")
            continue
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            _warn("skipping angles row with no nomination id")
            continue
        if row_id not in known:
            _warn(f"ignoring angles for unknown nomination id {row_id!r}")
            continue
        podcast = _sanitized_angle(row.get("podcast"))
        x_article = _sanitized_angle(row.get("x_article"))
        if podcast is None and x_article is None:
            # No usable hook at all: treat the row as absent.
            continue
        angles[row_id] = HostAngles(podcast=podcast, x_article=x_article)
    return angles


def resolve_name_collisions(
    pairs: Sequence[tuple[pipeline.Nomination, str]],
) -> list[str]:
    """Re-run the nominate-stage casefold/entity-token collision rules over
    host-applied names, returning one collision-free name per input pair in
    order.

    Short host-judged names collide far more often than raw titles; a
    colliding name gets the later nomination's strongest non-shared entity
    token appended (``pipeline._disambiguated_topic_name``, fed synthetic
    per-nomination clusters built from the seed items). Unlike the nominate
    stage, a collision can never DROP a nomination here - the pool already
    de-duplicated same-story clusters at leg 1 - so when no distinguishing
    entity token exists the name falls back to an ordinal suffix.
    """
    candidate_map: dict[str, schema.Candidate] = {}
    clusters: list[schema.Cluster] = []
    for index, (nomination, _applied) in enumerate(pairs):
        candidate_ids: list[str] = []
        for item_index, item in enumerate(nomination.items):
            candidate_id = f"handoff-{index}-{item_index}"
            candidate_map[candidate_id] = schema.Candidate(
                candidate_id=candidate_id,
                item_id=item.item_id,
                source=item.source,
                title=item.title,
                url=item.url,
                snippet=item.snippet,
                subquery_labels=[],
                native_ranks={},
                local_relevance=0.0,
                freshness=0,
                engagement=None,
                source_quality=0.0,
                rrf_score=0.0,
            )
            candidate_ids.append(candidate_id)
        clusters.append(schema.Cluster(
            cluster_id=f"handoff-n{index}",
            title=nomination.name,
            candidate_ids=candidate_ids,
            representative_ids=candidate_ids[:1],
            sources=sorted({item.source for item in nomination.items}),
            score=nomination.seed_score,
        ))

    resolved_names: list[str] = []
    taken: dict[str, schema.Cluster] = {}
    entity_counts_cache: dict[str, Counter] = {}
    for index, (_nomination, applied) in enumerate(pairs):
        cluster = clusters[index]
        name = applied
        key = name.casefold()
        if key in taken:
            resolved = pipeline._disambiguated_topic_name(
                name, cluster, taken[key], candidate_map, entity_counts_cache,
                taken,
            )
            if resolved is None:
                # Indistinguishable by content: keep the nomination anyway
                # (distinct stories at leg 1) under an ordinal suffix.
                suffix = 2
                while f"{name} {suffix}".casefold() in taken:
                    suffix += 1
                resolved = f"{name} {suffix}"
            name = resolved
            key = name.casefold()
        taken[key] = cluster
        resolved_names.append(name)
    return resolved_names


def _one_line(text: str) -> str:
    return " ".join(text.split())


def build_host_digest(bundle: NominationsBundle) -> str:
    """The host-facing judging digest for a nominations bundle: plain,
    promptable text with one line per nomination (id, leader title, seed
    source names, velocity/engagement signal) plus capped evidence lines
    (leader snippet, strongest community comment - the surface the engine
    judge used to see). Names the bundle file and instructs the host to read
    its full evidence before judging."""
    location = str(bundle.path) if bundle.path is not None else (
        NOMINATIONS_BUNDLE_FILENAME
    )
    domain_label = bundle.domain or "global trending (no domain filter)"
    lines = [
        f"Discovery nominations awaiting host judgment "
        f"({len(bundle.nominations)} topics).",
        f"Domain: {domain_label} | window {bundle.from_date} -> "
        f"{bundle.to_date} | tier {bundle.tier}",
        f"Bundle file: {location} (bundle_id {bundle.bundle_id})",
        "Read the bundle file's per-nomination evidence before judging; the "
        "lines below are only a digest.",
        "",
    ]
    for entry in bundle.nominations:
        items = entry.nomination.items
        leader = items[0] if items else None
        title = _one_line((leader.title if leader else "") or entry.nomination.name)
        sources = ", ".join(entry.sources) if entry.sources else "unknown"
        native_total = sum(
            rerank.discovery_engagement_total(item) for item in items
        )
        lines.append(
            f"{entry.nomination_id} | {title[:_DIGEST_TITLE_MAX_CHARS]} | "
            f"sources: {sources} | signal: seed velocity "
            f"{entry.nomination.seed_score:.1f}, "
            f"{native_total:,.0f} native interactions"
        )
        snippet_text = _one_line(
            (leader.snippet if leader else "") or entry.nomination.summary
        )
        if snippet_text:
            lines.append(f"  snippet: {snippet_text[:_DIGEST_SNIPPET_MAX_CHARS]}")
        top_comment = pipeline._best_community_comment(items)
        if top_comment:
            lines.append(
                f"  top comment: "
                f"{_one_line(top_comment)[:_DIGEST_COMMENT_MAX_CHARS]}"
            )
    return "\n".join(lines)
