"""U2 - nomination ranking: cluster nominated items into named, seed-ranked
candidate topics.

nominate_topics() is the contract between the nominate stage and the
enrichment fan-out: short distilled names ordered by worthiness-blended seed
velocity, casefold-collision-safe, never padded past what the evidence
supports. The heuristic (no-provider) naming path is pinned here; the LLM
stage-1 judge path lives in test_discover_judge.py.
"""

from unittest import mock

from lib import dates, pipeline, render, rerank, schema, topic_shape


def _item(
    item_id: str,
    source: str,
    title: str,
    *,
    published_at: str = "2026-07-09",
    engagement: dict[str, int | float] | None = None,
) -> schema.SourceItem:
    return schema.SourceItem(
        item_id=item_id,
        source=source,
        title=title,
        body=title,
        url=f"https://{source}.example/{item_id}",
        published_at=published_at,
        engagement=engagement or {},
        snippet=f"Evidence about {title}",
    )


def _bundle(items: list[schema.SourceItem]) -> schema.RetrievalBundle:
    bundle = schema.RetrievalBundle()
    by_source: dict[str, list[schema.SourceItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    for source, source_items in by_source.items():
        bundle.add_items("discovery-listings", source, source_items)
    return bundle


def _query_plan(domain: str, sources: list[str]) -> schema.QueryPlan:
    return schema.QueryPlan(
        intent="breaking_news",
        freshness_mode="breaking",
        cluster_mode="story",
        raw_topic=domain,
        subqueries=[schema.SubQuery(
            label="discovery-listings",
            search_query=domain,
            ranking_query=f"What is accelerating in {domain}?",
            sources=sources,
        )],
        source_weights={source: 1.0 for source in sources},
        notes=["discover-mode", "listing-sweep"],
    )


def _plan(domain: str, sources: list[str]) -> schema.DiscoveryPlan:
    return schema.DiscoveryPlan(
        domain=domain, category=None, subreddits=["all"], sources=sources,
    )


def test_nominations_ranked_by_seed_velocity():
    """A high-engagement recent story outranks a low-engagement one."""
    items = [
        _item("hot1", "hackernews", "GPT-6 rumors flood the valley",
              engagement={"points": 900, "num_comments": 400}),
        _item("cold1", "hackernews", "Minor framework patch notes released",
              engagement={"points": 3, "num_comments": 1}),
    ]
    nominations = pipeline.nominate_topics(
        _bundle(items), _query_plan("AI", ["hackernews"]), _plan("AI", ["hackernews"]),
        to_date="2026-07-10", limit=10,
    )
    assert nominations, "expected at least one nomination"
    assert "GPT-6" in nominations[0].name
    assert nominations[0].seed_score >= (nominations[-1].seed_score)


def test_nominations_dedupe_names_casefold():
    """Two clusters resolving to the same casefolded name yield one nomination."""
    items = [
        _item("a1", "hackernews", "OpenAI Agent SDK",
              engagement={"points": 500, "num_comments": 100}),
        _item("a2", "reddit", "openai agent sdk",
              engagement={"score": 300, "num_comments": 80}),
    ]
    nominations = pipeline.nominate_topics(
        _bundle(items), _query_plan("AI agents", ["hackernews", "reddit"]),
        _plan("AI agents", ["hackernews", "reddit"]),
        to_date="2026-07-10", limit=10,
    )
    names = [nomination.name.casefold() for nomination in nominations]
    assert len(names) == len(set(names))


def test_fewer_clusters_than_limit_returns_all_without_padding():
    items = [
        _item("only1", "hackernews", "Quantum breakthrough announced",
              engagement={"points": 250, "num_comments": 60}),
    ]
    nominations = pipeline.nominate_topics(
        _bundle(items), _query_plan("quantum", ["hackernews"]),
        _plan("quantum", ["hackernews"]),
        to_date="2026-07-10", limit=8,
    )
    assert 1 <= len(nominations) < 8


def test_zero_velocity_clusters_are_dropped():
    """Items with no engagement produce no nomination at all."""
    items = [
        _item("dead1", "hackernews", "Silent post nobody engaged with",
              engagement={"points": 0, "num_comments": 0}),
    ]
    nominations = pipeline.nominate_topics(
        _bundle(items), _query_plan("AI", ["hackernews"]), _plan("AI", ["hackernews"]),
        to_date="2026-07-10", limit=8,
    )
    assert nominations == []


# Real-run shapes from the motivating 2026-07 discovery sweep (see topic_shape).
ANECDOTE_TITLE = (
    "My coworker let an AI agent handle Slack replies while he was "
    '"unavailable." It did not go well.'
)
HELP_TITLE = "I need help starting to learn about AI agents"


def test_names_are_short_distilled_topics_not_raw_titles():
    """The nomination's name IS the enrichment search query and the
    /last30days handoff - anecdote/question scaffolding must not leak into it."""
    items = [
        _item("story1", "hackernews", ANECDOTE_TITLE,
              engagement={"points": 400, "comments": 100}),
    ]
    nominations = pipeline.nominate_topics(
        _bundle(items), _query_plan("AI agents", ["hackernews"]),
        _plan("AI agents", ["hackernews"]),
        to_date="2026-07-10", limit=10,
    )
    assert nominations
    name = nominations[0].name
    assert len(name.split()) <= 6
    assert not name.lower().startswith("my coworker")


def test_no_provider_names_are_distilled_and_deterministic():
    """provider=None (keyless/mock) is the pure-heuristic path: names come
    from topic_shape.distill_topic_name, junk flags from is_junk_shape, and
    two identical runs produce identical output - no LLM, no randomness."""
    items = [
        _item("story1", "hackernews", ANECDOTE_TITLE,
              engagement={"points": 400, "comments": 100}),
        _item("junk1", "hackernews", HELP_TITLE,
              engagement={"points": 200, "comments": 50}),
    ]
    bundle = _bundle(items)

    def run() -> list[pipeline.Nomination]:
        return pipeline.nominate_topics(
            bundle, _query_plan("AI agents", ["hackernews"]),
            _plan("AI agents", ["hackernews"]),
            to_date="2026-07-10", limit=10, provider=None, model=None,
        )

    first, second = run(), run()
    assert [nomination.name for nomination in first] == [
        nomination.name for nomination in second
    ]
    assert [nomination.junk_shape for nomination in first] == [
        nomination.junk_shape for nomination in second
    ]

    by_leader = {nomination.items[0].item_id: nomination for nomination in first}
    story = by_leader["story1"]
    assert story.name == topic_shape.distill_topic_name(ANECDOTE_TITLE)
    assert story.junk_shape is False
    assert by_leader["junk1"].junk_shape is True
    # No provider -> no worthiness signal; ranking stays velocity-only.
    assert all(nomination.worthiness is None for nomination in first)


def test_nomination_carries_leader_summary_and_items():
    items = [
        _item("s1", "hackernews", "Rust rewrite of the Linux scheduler",
              engagement={"points": 700, "num_comments": 250}),
    ]
    nominations = pipeline.nominate_topics(
        _bundle(items), _query_plan("Linux", ["hackernews"]), _plan("Linux", ["hackernews"]),
        to_date="2026-07-10", limit=8,
    )
    assert nominations
    top = nominations[0]
    assert top.items and top.items[0].item_id == "s1"
    assert top.summary


# --- U3 leg 1: nominate-only judge pool ---------------------------------------


def test_nominate_topic_pool_pairs_nominations_with_cluster_ids():
    """The pool variant returns the SAME nominations as nominate_topics, each
    paired with its non-empty, unique source cluster id."""
    items = [
        _item("hot1", "hackernews", "GPT-6 rumors flood the valley",
              engagement={"points": 900, "num_comments": 400}),
        _item("warm1", "hackernews", "Quantum error correction milestone announced",
              engagement={"points": 250, "num_comments": 60}),
    ]
    bundle = _bundle(items)
    query_plan = _query_plan("AI", ["hackernews"])
    plan = _plan("AI", ["hackernews"])
    pool = pipeline.nominate_topic_pool(
        bundle, query_plan, plan, to_date="2026-07-10", limit=10,
    )
    nominations = pipeline.nominate_topics(
        bundle, query_plan, plan, to_date="2026-07-10", limit=10,
    )
    assert [nomination for nomination, _cluster_id in pool] == nominations
    cluster_ids = [cluster_id for _nomination, cluster_id in pool]
    assert all(cluster_ids)
    assert len(set(cluster_ids)) == len(cluster_ids)


# Ten clearly distinct stories: enough clusters to prove the judge pool
# reaches past the ENRICH_LIMIT cut that the one-shot path applies.
POOL_TITLES = [
    "Kestrel avionics merger approved by regulators",
    "Sourdough robot bakery raises series B",
    "Quantum error correction milestone announced",
    "Rust rewrite of the Linux scheduler lands",
    "Solar balcony panels top German sales charts",
    "Deep sea mining moratorium gains momentum",
    "Vertical farming startup exits stealth with kale gigafactory",
    "Formula E battery swap trial starts in Rome",
    "Open source weather models beat commercial forecasts",
    "Cheese aging caves converted to data centers",
]


def _hn_raw(item_id: str, title: str, points: int, comments: int, *, date: str = "2026-07-09") -> dict:
    return {
        "id": item_id,
        "title": title,
        "url": f"https://example.com/{item_id}",
        "hn_url": f"https://news.ycombinator.com/item?id={item_id}",
        # Distinct authors: weighted_rrf caps the pool per author, and this
        # fixture exists to overflow the ENRICH_LIMIT cut, not that cap.
        "author": f"author-{item_id}",
        "date": date,
        "engagement": {"points": points, "comments": comments},
        "relevance": 0.9,
    }


def _nominate_only(items_by_source: dict[str, list[dict]], **kwargs) -> "pipeline.DiscoverNominateResult":
    def fake_fetch(source, plan, *, from_date, to_date, depth, mock, config, keyword_gate=True):
        return items_by_source.get(source, []), None

    with mock.patch.object(
        pipeline, "available_sources", return_value=list(items_by_source),
    ), mock.patch.object(
        pipeline, "_fetch_discovery_source", side_effect=fake_fetch,
    ):
        return pipeline.run_discover_nominate(
            domain=kwargs.pop("domain", ""),
            config={},
            as_of_date="2026-07-10",
            **kwargs,
        )


def _full_pool_items() -> dict[str, list[dict]]:
    return {"hackernews": [
        _hn_raw(f"hn{index}", title, 900 - index * 40, 120 - index * 5)
        for index, title in enumerate(POOL_TITLES)
    ]}


def test_nominate_only_emits_full_judge_pool_beyond_enrich_cut():
    """Leg 1 hands the host the FULL judge pool (up to JUDGE_POOL_LIMIT), not
    the one-shot path's post-cut enrichment list."""
    result = _nominate_only(_full_pool_items())
    assert len(result.pool) > pipeline.ENRICH_LIMIT
    assert len(result.pool) <= rerank.JUDGE_POOL_LIMIT
    names = [nomination.name.casefold() for nomination, _cluster_id in result.pool]
    assert len(names) == len(set(names))


def test_nominate_only_is_heuristic_deterministic_and_provider_free():
    """Leg 1 never resolves a reasoning provider: names/junk flags are the
    deterministic topic_shape heuristics and two runs agree exactly."""
    items = _full_pool_items()
    items["hackernews"].append(
        _hn_raw("junk1", HELP_TITLE, 400, 90)
    )
    with mock.patch.object(pipeline.providers, "resolve_runtime") as resolve:
        first = _nominate_only(items)
        second = _nominate_only(items)
    resolve.assert_not_called()
    assert [
        (nomination.name, nomination.junk_shape, cluster_id)
        for nomination, cluster_id in first.pool
    ] == [
        (nomination.name, nomination.junk_shape, cluster_id)
        for nomination, cluster_id in second.pool
    ]
    assert all(nomination.worthiness is None for nomination, _ in first.pool)
    junk_flags = {
        nomination.items[0].item_id: nomination.junk_shape
        for nomination, _ in first.pool
    }
    assert junk_flags.get("junk1") is True


def test_nominate_only_window_matches_sweep_dates():
    result = _nominate_only(_full_pool_items(), lookback_days=7)
    assert (result.from_date, result.to_date) == dates.get_date_range(
        7, as_of_date="2026-07-10"
    )


def test_nominate_only_never_enriches_or_researches():
    """No enrichment, no full research sub-runs on leg 1 - the host judges
    the seed evidence first."""
    with mock.patch.object(pipeline, "enrich_nominations") as enrich, \
         mock.patch.object(pipeline, "run") as full_run:
        result = _nominate_only(_full_pool_items())
    enrich.assert_not_called()
    full_run.assert_not_called()
    assert result.pool


def test_nominate_only_zero_pool_renders_nothing_solid_brief():
    """An empty sweep short-circuits to the existing nothing-solid brief."""
    result = _nominate_only({"hackernews": []})
    assert result.pool == []
    report = pipeline.nominate_nothing_solid_report(result)
    assert report.outcome == "nothing-solid"
    assert report.topics == []
    assert (report.range_from, report.range_to) == (result.from_date, result.to_date)
    rendered = render.render_discovery(report)
    assert "Nothing solid this window." in rendered
