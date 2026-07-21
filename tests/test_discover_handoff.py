"""U1 - discovery handoff file contracts for the three-leg host-judged protocol.

Leg 1 (``--discover --nominate-only``) writes a nominations bundle carrying
the FULL judge pool losslessly; leg 2 (``--discover --judgments <file>``)
binds host judgments to that bundle by bundle_id; leg 3 (``--discover
--finalize [--angles <file>]``) applies host-written angles. This file pins
the bundle writer/reader round-trip, strict-top-level / lenient-per-row
reader semantics, TTL and version gating, sanitation, collision handling,
and the host-facing digest.
"""

import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest

from lib import discovery_handoff as handoff
from lib import pipeline, rerank, schema


def _item(
    item_id: str,
    source: str,
    title: str,
    *,
    published_at: str = "2026-07-18",
    engagement: dict[str, int | float] | None = None,
    snippet: str = "",
    metadata: dict | None = None,
) -> schema.SourceItem:
    return schema.SourceItem(
        item_id=item_id,
        source=source,
        title=title,
        body=title,
        url=f"https://{source}.example/{item_id}",
        published_at=published_at,
        engagement=engagement or {},
        snippet=snippet or f"Evidence about {title}",
        metadata=metadata or {},
    )


def _nomination(
    name: str,
    items: list[schema.SourceItem],
    *,
    seed_score: float = 42.5,
    summary: str = "",
    junk_shape: bool = False,
    worthiness: float | None = None,
) -> pipeline.Nomination:
    return pipeline.Nomination(
        name=name,
        seed_score=seed_score,
        items=items,
        summary=summary or f"Summary of {name}",
        junk_shape=junk_shape,
        worthiness=worthiness,
    )


def _entry(
    nomination: pipeline.Nomination,
    *,
    cluster_id: str = "c1",
    heuristic_name: str | None = None,
    heuristic_junk: bool = False,
) -> "handoff.PoolEntry":
    return handoff.PoolEntry(
        nomination=nomination,
        cluster_id=cluster_id,
        heuristic_name=heuristic_name if heuristic_name is not None else nomination.name,
        heuristic_junk=heuristic_junk,
    )


def _pool() -> list["handoff.PoolEntry"]:
    agent = _nomination(
        "Agent SDK Wars",
        [
            _item(
                "hn1", "hackernews",
                "Agent SDK Wars heat up as Anthropic ships a Claude agent runtime",
                engagement={"points": 900, "comments": 400},
            ),
            _item(
                "rd1", "reddit",
                "Agent SDK wars: which runtime are you betting on?",
                engagement={"score": 300, "num_comments": 80},
                metadata={"top_comments": [{
                    "excerpt": "The SDK churn is unsustainable for small teams",
                    "score": 1635,
                    "author": "dev_a",
                }]},
            ),
        ],
        seed_score=61.2,
    )
    quantum = _nomination(
        "Quantum Error Correction",
        [
            _item(
                "hn2", "hackernews",
                "Quantum error correction milestone announced",
                engagement={"points": 250, "comments": 60},
            ),
        ],
        seed_score=18.4,
    )
    return [
        _entry(agent, cluster_id="c-agent", heuristic_junk=False),
        _entry(quantum, cluster_id="c-quantum", heuristic_junk=True),
    ]


def _write(config_dir, entries=None, **overrides) -> "handoff.NominationsBundle":
    kwargs = dict(
        domain="AI",
        tier="deep",
        from_date="2026-06-21",
        to_date="2026-07-21",
        lookback_days=30,
        enrichment_source_boundary=None,
        requested_sources=["hackernews", "reddit"],
        save_dir=None,
        config_dir=config_dir,
    )
    kwargs.update(overrides)
    return handoff.write_nominations_bundle(
        entries if entries is not None else _pool(), **kwargs
    )


def _judgments_file(tmp_path, payload) -> "Path":
    path = tmp_path / "judgments.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- Scenario 1: bundle round-trip ------------------------------------------


def test_bundle_round_trip_is_lossless(tmp_path):
    written = _write(tmp_path)
    read = handoff.read_nominations_bundle(save_dir=None, config_dir=tmp_path)
    assert read.bundle_id == written.bundle_id
    assert read.schema_version == schema.DISCOVERY_NOMINATIONS_SCHEMA_VERSION
    assert (read.from_date, read.to_date) == ("2026-06-21", "2026-07-21")
    assert read.domain == "AI"
    assert read.tier == "deep"
    assert read.lookback_days == 30
    assert read.enrichment_source_boundary is None
    assert read.requested_sources == ["hackernews", "reddit"]
    assert [row.nomination_id for row in read.nominations] == ["n1", "n2"]
    for row, entry in zip(read.nominations, _pool()):
        # Full dataclass equality: name, seed_score, every seed item field,
        # summary, junk_shape, worthiness.
        assert row.nomination == entry.nomination
        assert row.cluster_id == entry.cluster_id
        assert row.heuristic_name == entry.heuristic_name
        assert row.heuristic_junk == entry.heuristic_junk
        assert row.sources == sorted({i.source for i in entry.nomination.items})
    assert read.path == tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME


def test_save_dir_takes_precedence_over_config_dir(tmp_path):
    save_dir = tmp_path / "saves"
    config_dir = tmp_path / "config"
    written = _write(config_dir, save_dir=save_dir)
    assert written.path == save_dir / handoff.NOMINATIONS_BUNDLE_FILENAME
    read = handoff.read_nominations_bundle(save_dir=save_dir, config_dir=config_dir)
    assert read.bundle_id == written.bundle_id


def test_source_boundary_and_shallow_tier_survive_round_trip(tmp_path):
    _write(
        tmp_path,
        tier="shallow",
        enrichment_source_boundary=["reddit", "hackernews"],
        requested_sources=None,
        lookback_days=7,
    )
    read = handoff.read_nominations_bundle(config_dir=tmp_path)
    assert read.tier == "shallow"
    assert read.enrichment_source_boundary == ["reddit", "hackernews"]
    assert read.requested_sources is None
    assert read.lookback_days == 7


# --- Scenario 2: parity pin --------------------------------------------------


def test_parity_floor_and_velocity_inputs_survive_round_trip(tmp_path):
    entries = _pool()
    _write(tmp_path, entries)
    read = handoff.read_nominations_bundle(config_dir=tmp_path)
    for row, entry in zip(read.nominations, entries):
        before, after = entry.nomination.items, row.nomination.items
        assert len(after) == len(before)
        assert [i.engagement for i in after] == [i.engagement for i in before]
        assert [i.source for i in after] == [i.source for i in before]
        assert [i.published_at for i in after] == [i.published_at for i in before]
        # Entity-token disambiguation inputs (title + snippet) are lossless.
        assert [(i.title, i.snippet) for i in after] == [
            (i.title, i.snippet) for i in before
        ]
        # Velocity and floor inputs recompute identically to an in-memory run.
        assert rerank.discovery_velocity_score(
            after, as_of_date="2026-07-21"
        ) == rerank.discovery_velocity_score(before, as_of_date="2026-07-21")
        assert sum(rerank.discovery_engagement_total(i) for i in after) == sum(
            rerank.discovery_engagement_total(i) for i in before
        )
        assert {i.source for i in after} == {i.source for i in before}


# --- Scenario 3: judgments reader --------------------------------------------


def test_judgments_apply_by_id_with_per_row_leniency(tmp_path, capsys):
    bundle = _write(tmp_path)
    path = _judgments_file(tmp_path, {
        "bundle_id": bundle.bundle_id,
        "judgments": [
            {"id": "n1", "name": "Claude Agent Runtime Launch", "junk": False,
             "worthiness": 78},
            {"id": "n9", "name": "Ghost Topic", "worthiness": 50},
        ],
    })
    judgments = handoff.read_judgments(
        path, bundle, save_dir=None, config_dir=tmp_path
    )
    assert judgments["n1"] == handoff.HostJudgment(
        name="Claude Agent Runtime Launch", junk=False, worthiness=78,
    )
    # Unknown id: warned (always visible, tty_only=False) and ignored.
    assert "n9" not in judgments
    assert "n9" in capsys.readouterr().err
    # n2 omitted entirely -> per-row-absent marker; the caller falls back to
    # the bundle's heuristic name/junk.
    assert handoff.judgment_for(judgments, "n2") is handoff.ROW_ABSENT
    assert handoff.ROW_ABSENT.name is None
    assert handoff.ROW_ABSENT.junk is None
    assert handoff.ROW_ABSENT.worthiness is None


def test_judgments_worthiness_clamped_to_0_100_integers(tmp_path):
    bundle = _write(tmp_path)
    path = _judgments_file(tmp_path, {
        "bundle_id": bundle.bundle_id,
        "judgments": [
            {"id": "n1", "worthiness": 150},
            {"id": "n2", "worthiness": -3.7},
        ],
    })
    judgments = handoff.read_judgments(path, bundle)
    assert judgments["n1"].worthiness == 100
    assert judgments["n2"].worthiness == 0
    assert isinstance(judgments["n1"].worthiness, int)
    # No name on either row -> per-row-absent name and junk.
    assert judgments["n1"].name is None
    assert judgments["n1"].junk is None


def test_junk_accepted_even_without_usable_name(tmp_path):
    bundle = _write(tmp_path)
    path = _judgments_file(tmp_path, {
        "bundle_id": bundle.bundle_id,
        "judgments": [{"id": "n2", "name": "\U0001f525\U0001f525\U0001f525",
                       "junk": True}],
    })
    judgments = handoff.read_judgments(path, bundle)
    assert judgments["n2"].junk is True
    # Emoji-only sanitizes to empty = per-row-absent name.
    assert judgments["n2"].name is None


# --- Scenario 4: error matrix -------------------------------------------------


def test_error_unreadable_bundle_file(tmp_path):
    # A directory at the bundle path exists but cannot be read as a file.
    (tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME).mkdir()
    with pytest.raises(handoff.HandoffContractError) as excinfo:
        handoff.read_nominations_bundle(config_dir=tmp_path)
    assert excinfo.value.message


def test_error_invalid_json(tmp_path):
    (tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME).write_text(
        "{not json", encoding="utf-8"
    )
    with pytest.raises(handoff.HandoffContractError) as excinfo:
        handoff.read_nominations_bundle(config_dir=tmp_path)
    assert "JSON" in excinfo.value.message


def test_error_top_level_non_dict(tmp_path):
    (tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME).write_text(
        "[]", encoding="utf-8"
    )
    with pytest.raises(handoff.HandoffContractError):
        handoff.read_nominations_bundle(config_dir=tmp_path)


def test_error_wrong_schema_version(tmp_path):
    _write(tmp_path)
    path = tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "99.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(handoff.HandoffContractError) as excinfo:
        handoff.read_nominations_bundle(config_dir=tmp_path)
    assert "99.0" in excinfo.value.message


def test_error_stale_ttl(tmp_path):
    _write(tmp_path)
    path = tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=handoff.DISCOVERY_HANDOFF_TTL_SECONDS + 60
    )
    payload["generated_at"] = stale.isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(handoff.HandoffContractError) as excinfo:
        handoff.read_nominations_bundle(config_dir=tmp_path)
    assert "--discover --nominate-only" in excinfo.value.message


def test_ttl_is_not_the_report_cache_env_knob(tmp_path, monkeypatch):
    """A user who lowered LAST30DAYS_REPORT_CACHE_TTL_SECONDS for drill
    freshness must not shrink the judgment-authoring window."""
    monkeypatch.setenv("LAST30DAYS_REPORT_CACHE_TTL_SECONDS", "1")
    written = _write(tmp_path)
    path = tmp_path / handoff.NOMINATIONS_BUNDLE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    two_minutes_old = datetime.now(timezone.utc) - timedelta(seconds=120)
    payload["generated_at"] = two_minutes_old.isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")
    read = handoff.read_nominations_bundle(config_dir=tmp_path)
    assert read.bundle_id == written.bundle_id
    assert handoff.DISCOVERY_HANDOFF_TTL_SECONDS == 3600.0


def test_error_bundle_not_found_names_both_locations_and_remedy(tmp_path):
    save_dir = tmp_path / "saves"
    config_dir = tmp_path / "config"
    with pytest.raises(handoff.HandoffContractError) as excinfo:
        handoff.read_nominations_bundle(save_dir=save_dir, config_dir=config_dir)
    message = excinfo.value.message
    assert str(save_dir / handoff.NOMINATIONS_BUNDLE_FILENAME) in message
    assert str(config_dir / handoff.NOMINATIONS_BUNDLE_FILENAME) in message
    assert "--discover --nominate-only" in message
    assert message.rstrip().endswith("re-sweep.")


def test_error_bundle_id_mismatch_names_locations_and_remedy(tmp_path):
    save_dir = tmp_path / "saves"
    config_dir = tmp_path / "config"
    bundle = _write(config_dir)
    path = _judgments_file(tmp_path, {
        "bundle_id": "deadbeefdeadbeef",
        "judgments": [],
    })
    with pytest.raises(handoff.HandoffContractError) as excinfo:
        handoff.read_judgments(path, bundle, save_dir=save_dir, config_dir=config_dir)
    message = excinfo.value.message
    assert str(save_dir / handoff.NOMINATIONS_BUNDLE_FILENAME) in message
    assert str(config_dir / handoff.NOMINATIONS_BUNDLE_FILENAME) in message
    assert "--discover --nominate-only" in message
    assert message.rstrip().endswith("re-sweep.")


def test_error_unreadable_judgments_path(tmp_path):
    bundle = _write(tmp_path)
    with pytest.raises(handoff.HandoffContractError):
        handoff.read_judgments(tmp_path / "missing.json", bundle)


def test_error_judgments_top_level_strict(tmp_path):
    bundle = _write(tmp_path)
    # Missing the "judgments" list entirely: strict at top level.
    path = _judgments_file(tmp_path, {"bundle_id": bundle.bundle_id})
    with pytest.raises(handoff.HandoffContractError):
        handoff.read_judgments(path, bundle)
    # Top-level non-dict.
    non_dict = tmp_path / "non-dict.json"
    non_dict.write_text('["not", "a", "dict"]', encoding="utf-8")
    with pytest.raises(handoff.HandoffContractError):
        handoff.read_judgments(non_dict, bundle)


# --- Scenario 5: sanitation and collisions ------------------------------------


def test_long_host_name_truncates_at_word_boundary(tmp_path):
    bundle = _write(tmp_path)
    long_name = " ".join(["momentum"] * 40)  # well over 96 chars
    path = _judgments_file(tmp_path, {
        "bundle_id": bundle.bundle_id,
        "judgments": [{"id": "n1", "name": long_name}],
    })
    judgments = handoff.read_judgments(path, bundle)
    name = judgments["n1"].name
    assert name is not None
    assert len(name) <= 96
    # Cut at a word boundary: no partial trailing token.
    assert set(name.split()) == {"momentum"}


def test_case_only_name_collisions_disambiguate_not_collapse():
    first = _nomination(
        "Agent Wars",
        [_item("hn1", "hackernews",
               "Agent Wars heat up as Anthropic ships Claude runtime",
               engagement={"points": 900, "comments": 100})],
    )
    second = _nomination(
        "Agent Runtime Rivalry",
        [_item("rd1", "reddit",
               "Agent wars escalate as OpenAI counters with Codex swarm",
               engagement={"score": 250, "num_comments": 30})],
    )
    resolved = handoff.resolve_name_collisions([
        (first, "Agent Wars"),
        (second, "agent wars"),
    ])
    assert len(resolved) == 2  # never collapses distinct nominations
    assert resolved[0] == "Agent Wars"
    assert resolved[1].casefold() != "agent wars"
    assert resolved[1].casefold().startswith("agent wars")
    assert len({name.casefold() for name in resolved}) == 2


def test_indistinguishable_collision_still_never_drops():
    shared = [_item("hn1", "hackernews", "Agent Wars heat up",
                    engagement={"points": 100})]
    first = _nomination("Agent Wars", shared)
    second = _nomination("Agent Wars redux", shared)
    resolved = handoff.resolve_name_collisions([
        (first, "Agent Wars"),
        (second, "agent wars"),
    ])
    assert len(resolved) == 2
    assert len({name.casefold() for name in resolved}) == 2


# --- Scenario 6: angles reader -------------------------------------------------


def test_angles_apply_truncate_and_none_path_returns_empty(tmp_path):
    bundle = _write(tmp_path)
    # Missing angles file is legal.
    assert handoff.read_angles(None, bundle) == {}
    long_angle = " ".join(["angle"] * 60)  # well over 200 chars
    path = tmp_path / "angles.json"
    path.write_text(json.dumps({
        "bundle_id": bundle.bundle_id,
        "angles": [
            {"id": "n1",
             "podcast": "Why the agent SDK churn is a tax on small teams",
             "x_article": long_angle},
            {"id": "n9", "podcast": "Ghost angle"},
        ],
    }), encoding="utf-8")
    angles = handoff.read_angles(path, bundle)
    assert angles["n1"].podcast == (
        "Why the agent SDK churn is a tax on small teams"
    )
    x_article = angles["n1"].x_article
    assert x_article is not None
    assert len(x_article) <= 200
    assert set(x_article.split()) == {"angle"}  # word-boundary truncation
    assert "n9" not in angles  # unknown ids ignored


# --- Scenario 7: digest ----------------------------------------------------------


LONG_TITLE = (
    "Anthropic ships a Claude agent runtime and the fallout reshapes agents " * 4
).strip()  # > 220 chars


def test_digest_names_bundle_path_instruction_and_capped_evidence(tmp_path):
    long_snippet = (
        "The community reaction spans pricing, lock-in, and migration pain. " * 10
    ).strip()  # > 420 chars
    nomination = _nomination(
        "Agent Runtime Fallout",
        [_item(
            "hn1", "hackernews", LONG_TITLE,
            engagement={"points": 1200, "comments": 300},
            snippet=long_snippet,
            metadata={"top_comments": [{
                "excerpt": "This will consolidate the whole agent ecosystem "
                           "within a year",
                "score": 1635,
                "author": "dev_a",
            }]},
        )],
        seed_score=77.7,
    )
    entries = [_entry(nomination, cluster_id="c-fallout"), _pool()[1]]
    bundle = _write(tmp_path, entries)
    digest = handoff.build_host_digest(bundle)
    # (b) names the bundle file path and instructs reading it before judging.
    assert str(bundle.path) in digest
    assert "before judging" in digest
    # (a) one line per nomination, keyed by nomination id.
    lines = digest.splitlines()
    n1_lines = [line for line in lines if line.startswith("n1 | ")]
    n2_lines = [line for line in lines if line.startswith("n2 | ")]
    assert len(n1_lines) == 1
    assert len(n2_lines) == 1
    # Evidence caps: the old judge surface (title ~220, snippet ~420).
    assert LONG_TITLE[:220] in n1_lines[0]
    assert LONG_TITLE[:230] not in digest
    assert "hackernews" in n1_lines[0]  # seed source names
    assert "1,500 native interactions" in n1_lines[0]  # engagement signal
    assert long_snippet[:420] in digest
    assert long_snippet[:430] not in digest
    assert "consolidate the whole agent ecosystem" in digest  # top comment
    # Plain text: no markdown tables.
    assert not any(line.lstrip().startswith("|") for line in lines)


# --- Hygiene ---------------------------------------------------------------------


def test_handoff_module_does_not_reference_discovery_judge():
    """The legacy engine-judge module is scheduled for deletion; the handoff
    module must port its sanitizers, never import or reference the module."""
    source = inspect.getsource(handoff)
    assert "discovery_judge" not in source
