import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import last30days as cli
from lib import pipeline, planner, reddit_listing, render, rerank, schema


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _candidate(item: schema.SourceItem) -> schema.Candidate:
    return schema.Candidate(
        candidate_id=f"candidate-{item.item_id}",
        item_id=item.item_id,
        source=item.source,
        title=item.title,
        url=item.url,
        snippet=item.snippet,
        subquery_labels=["discovery-listings"],
        native_ranks={f"discovery-listings:{item.source}": 1},
        local_relevance=0.9,
        freshness=95,
        engagement=100,
        source_quality=0.8,
        rrf_score=0.1,
        sources=[item.source],
        source_items=[item],
        final_score=80,
    )


def test_discovery_plan_reuses_category_peer_mapping():
    plan = planner.build_discovery_plan(
        "AI agents",
        available_sources=["reddit", "hackernews"],
    )

    assert plan.category == "ai_agent_framework"
    assert plan.subreddits == ["LangChain", "LocalLLaMA", "AI_Agents", "MachineLearning"]
    assert plan.sources == ["reddit", "hackernews"]


def test_discovery_plan_keeps_keyless_reddit_for_unknown_domains():
    plan = planner.build_discovery_plan(
        "urban gardening",
        available_sources=["reddit", "hackernews"],
    )

    assert plan.category is None
    assert plan.subreddits == ["all"]
    assert plan.sources == ["reddit", "hackernews"]


def test_discovery_plan_empty_domain_is_global_trending():
    """Bare --discover: sweep every river feed's hot list; X sits out of the
    nominate stage because its search lane needs a keyword."""
    plan = planner.build_discovery_plan(
        "",
        available_sources=["reddit", "hackernews", "digg", "x"],
    )

    assert plan.domain == ""
    assert plan.category is None
    assert plan.subreddits == ["all"]
    assert plan.sources == ["reddit", "hackernews", "digg"]
    assert "x" not in plan.sources


def test_global_discovery_disables_keyword_gate():
    """Global trending fetches with keyword_gate=False; domain runs keep it on."""
    seen: dict[str, bool] = {}

    def fake_fetch(source, plan, *, from_date, to_date, depth, mock, config, keyword_gate=True):
        seen[plan.domain or "global"] = keyword_gate
        return [], None

    with mock.patch.object(pipeline, "available_sources", return_value=["hackernews"]), \
         mock.patch.object(pipeline, "_fetch_discovery_source", side_effect=fake_fetch):
        pipeline.run_discover(domain="", config={}, as_of_date="2026-07-10")
        pipeline.run_discover(domain="AI agents", config={}, as_of_date="2026-07-10")

    assert seen["global"] is False
    assert seen["AI agents"] is True


def test_uncategorized_discovery_uses_parseable_r_all_listing_paths():
    card = (
        '<shreddit-post permalink="/r/gardening/comments/abc123/urban_garden/" '
        'post-title="Urban gardening is taking off" score="42" comment-count="7" '
        'author="gardener" subreddit-name="gardening" '
        'created-timestamp="2026-07-09T12:00:00+00:00">'
    )
    requested_urls: list[str] = []

    def fake_get(url, **_kwargs):
        requested_urls.append(url)
        return card

    with mock.patch.object(reddit_listing.http, "reddit_keyless_get_text", side_effect=fake_get):
        result = reddit_listing.fetch_discovery_listings(
            ["all"], query="urban gardening",
        )

    assert len(result["items"]) == 1
    assert any("/r/all/rising/" in url for url in requested_urls)
    assert any("/r/all/top/?t=week" in url for url in requested_urls)
    assert all("name=all" not in url for url in requested_urls)


def test_velocity_scoring_favors_a_recent_spike_over_static_bigness():
    recent = _item(
        "recent",
        "reddit",
        "Recent spike",
        published_at="2026-07-09",
        engagement={"score": 100, "num_comments": 10},
    )
    old = _item(
        "old",
        "reddit",
        "Older large thread",
        published_at="2026-06-20",
        engagement={"score": 300, "num_comments": 10},
    )

    assert rerank.engagement_velocity_score(recent, as_of_date="2026-07-10") > (
        rerank.engagement_velocity_score(old, as_of_date="2026-07-10")
    )


def test_domain_filter_ignores_generic_ai_only_matches():
    assert pipeline._matches_discovery_domain(
        "AI agents", "An AI agent bankrupted its operator"
    )
    assert not pipeline._matches_discovery_domain(
        "AI agents", "Global dialogue on AI governance"
    )


@pytest.mark.parametrize(
    ("domain", "listing_title"),
    [
        ("城市园艺", "城市园艺技巧与社区花园"),
        ("גינון עירוני", "מדריך חדש לגינון עירוני"),
    ],
)
def test_domain_filter_tokenizes_non_latin_domains(domain, listing_title):
    assert pipeline._matches_discovery_domain(domain, listing_title)


def test_x_velocity_excludes_views_and_bookmarks():
    xquik_item = _item(
        "xquik",
        "x",
        "X backend reach",
        engagement={
            "likes": 10,
            "reposts": 3,
            "replies": 2,
            "quotes": 1,
            "views": 100_000,
            "bookmarks": 5_000,
        },
    )
    standard_item = _item(
        "standard",
        "x",
        "X backend interactions",
        engagement={"likes": 10, "reposts": 3, "replies": 2, "quotes": 1},
    )

    assert rerank.discovery_engagement_total(xquik_item) == 16
    assert rerank.engagement_velocity_score(
        xquik_item, as_of_date="2026-07-10"
    ) == rerank.engagement_velocity_score(standard_item, as_of_date="2026-07-10")


def test_discovery_renderer_snapshot():
    report = schema.DiscoveryReport(
        domain="AI agents",
        range_from="2026-06-10",
        range_to="2026-07-10",
        generated_at="2026-07-10T00:00:00+00:00",
        plan=schema.DiscoveryPlan(
            domain="AI agents",
            category="ai_agent_framework",
            subreddits=["AI_Agents"],
            sources=["reddit", "hackernews"],
        ),
        topics=[schema.DiscoveryTopic(
            rank=1,
            name="Agent memory protocols",
            why_spiking="Two independent listing items accelerated this week.",
            momentum="new-this-week",
            velocity_score=123.45,
            sources=["hackernews", "reddit"],
            engagement_by_source={
                "reddit": {"score": 120, "num_comments": 30},
                "hackernews": {"points": 80},
            },
            command='/last30days "Agent memory protocols"',
        )],
    )

    with mock.patch.object(render, "_render_badge", return_value=["BADGE", ""]):
        rendered = render.render_discovery(report)

    assert rendered == (
        "BADGE\n\n"
        "# Trending discovery: AI agents\n\n"
        "Window: 2026-06-10 to 2026-07-10\n"
        "Feeds: reddit, hackernews\n"
        "Communities: r/AI_Agents\n\n"
        "## 1. Agent memory protocols\n\n"
        "**Momentum:** New this week · velocity 123.45\n\n"
        "Two independent listing items accelerated this week.\n\n"
        "**Evidence:** Reddit: score 120, num comments 30 · Hacker News: points 80\n\n"
        "**Research next:** `/last30days \"Agent memory protocols\"`\n"
    )


def test_keyless_discovery_degrades_without_digg():
    def fake_fetch(source, plan, *, from_date, to_date, depth, mock, config, keyword_gate=True):
        return pipeline._mock_discovery_items(source, plan.domain, to_date), None

    with mock.patch.object(pipeline, "available_sources", return_value=["reddit", "hackernews"]), \
         mock.patch.object(pipeline, "_fetch_discovery_source", side_effect=fake_fetch):
        report = pipeline.run_discover(
            domain="AI agents",
            config={},
            as_of_date="2026-07-10",
        )

    assert 5 <= len(report.topics) <= 10
    assert report.source_status["reddit"].state == "ok"
    assert report.source_status["hackernews"].state == "ok"
    assert report.source_status["digg"].state == "skipped-unconfigured"
    assert report.source_status["x"].state == "skipped-unconfigured"
    assert all(topic.command.startswith('/last30days "') for topic in report.topics)


def test_discovery_drops_zero_velocity_clusters():
    raw_item = {
        "id": "zero-engagement",
        "text": "AI agent launch with no interactions",
        "url": "https://x.com/example/status/1",
        "author_handle": "example",
        "date": "2026-07-09",
        "engagement": {"likes": 0, "reposts": 0, "replies": 0, "quotes": 0},
        "relevance": 0.9,
    }

    with mock.patch.object(pipeline, "available_sources", return_value=["x"]), \
         mock.patch.object(pipeline, "_fetch_discovery_source", return_value=([raw_item], None)):
        report = pipeline.run_discover(
            domain="AI agents",
            config={},
            as_of_date="2026-07-10",
        )

    assert report.topics == []
    assert report.outcome == "nothing-solid"
    assert any("confidence floor" in warning for warning in report.warnings)


def test_explicit_unavailable_discovery_source_does_not_widen_to_other_sources():
    with mock.patch.object(pipeline, "available_sources", return_value=[]), \
         mock.patch.object(pipeline, "_fetch_discovery_source") as fetch:
        with pytest.raises(ValueError, match="No listing sources are available"):
            pipeline.run_discover(
                domain="AI agents",
                config={},
                requested_sources=["digg"],
                as_of_date="2026-07-10",
            )

    fetch.assert_not_called()


def test_discovery_reads_browser_credentials_and_does_not_schedule_pending_x():
    parser = cli.build_parser()
    args, extra = parser.parse_known_args(["--discover", "AI agents"])
    assert cli._config_policy_for_args(args, "", extra).browser_cookies == "read"

    no_cookies_args, extra = parser.parse_known_args(
        ["--no-browser-cookies", "--discover", "AI agents"]
    )
    assert cli._config_policy_for_args(no_cookies_args, "", extra).browser_cookies == "off"

    fetched_sources: list[str] = []

    def fake_available_sources(config, requested_sources, *, x_pending=None, local_only=False):
        assert x_pending is False
        return ["reddit", "hackernews"] + (["x"] if x_pending is not False else [])

    def fake_fetch(source, plan, *, from_date, to_date, depth, mock, config, keyword_gate=True):
        fetched_sources.append(source)
        return pipeline._mock_discovery_items(source, plan.domain, to_date), None

    with mock.patch.object(pipeline, "available_sources", side_effect=fake_available_sources), \
         mock.patch.object(pipeline, "_fetch_discovery_source", side_effect=fake_fetch):
        report = pipeline.run_discover(
            domain="AI agents",
            config={"FROM_BROWSER": "firefox", "_BROWSER_COOKIE_MODE": "plan_only"},
            as_of_date="2026-07-10",
        )

    assert "x" not in fetched_sources
    assert report.source_status["x"].state == "skipped-unconfigured"


def test_authenticated_x_discovery_uses_available_backend():
    plan = planner.build_discovery_plan(
        "AI agents",
        available_sources=["x"],
    )
    raw = pipeline._mock_discovery_items("x", plan.domain, "2026-07-10")
    with mock.patch.object(pipeline.env, "x_backend_chain", return_value=["bird"]), \
         mock.patch.object(pipeline, "_fetch_x_backend", return_value=(raw, "")) as fetch:
        items, error = pipeline._fetch_discovery_source(
            "x",
            plan,
            from_date="2026-06-10",
            to_date="2026-07-10",
            depth="default",
            mock=False,
            config={"AUTH_TOKEN": "dummy", "CT0": "dummy"},
        )

    assert error is None
    assert len(items) == 6
    fetch.assert_called_once()


def test_listing_failure_is_not_reported_as_clean_no_results():
    def fake_fetch(source, plan, *, from_date, to_date, depth, mock, config, keyword_gate=True):
        if source == "reddit":
            return [], "connection timed out"
        return pipeline._mock_discovery_items(source, plan.domain, to_date), None

    with mock.patch.object(pipeline, "available_sources", return_value=["reddit", "hackernews"]), \
         mock.patch.object(pipeline, "_fetch_discovery_source", side_effect=fake_fetch):
        report = pipeline.run_discover(
            domain="AI agents",
            config={},
            as_of_date="2026-07-10",
        )

    assert report.source_status["reddit"].state == "timeout"
    assert report.source_status["reddit"].detail == "connection timed out"


def test_reddit_discovery_adapter_preserves_partial_feed_errors():
    item = {
        "url": "https://reddit.com/r/example/comments/1",
        "title": "AI agent launch",
    }
    with mock.patch.object(
        reddit_listing,
        "_fetch_one_with_status",
        side_effect=[([], "rising timed out"), ([item], None)],
    ):
        result = reddit_listing.fetch_discovery_listings(
            ["AI_Agents"], query="AI agents",
        )

    assert result["items"] == [item]
    assert result["errors"] == ["r/AI_Agents rising: rising timed out"]


def test_discovery_cli_json_contract_and_mutual_exclusion():
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "AI agents",
            "--mock",
            "--emit=json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.1"
    assert payload["kind"] == "discovery"
    assert 5 <= len(payload["results"]) <= 10
    assert payload["results"][0]["command"].startswith('/last30days "')
    # 1.1 fields ship in every result, with defaults when nothing set them.
    for topic in payload["results"]:
        assert topic["podcast_angle"] is None
        assert topic["x_article_angle"] is None
        assert topic["previously_surfaced_count"] == 0
        assert topic["last_surfaced"] is None
        assert topic["covered"] is False

    invalid = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "topic",
            "--discover",
            "AI agents",
            "--mock",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "cannot be combined with a positional topic" in invalid.stderr

    drill_conflict = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "AI agents",
            "--drill",
            "1",
            "--mock",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert drill_conflict.returncode == 2
    assert "mutually exclusive" in drill_conflict.stderr


def _discovery_report(topic: schema.DiscoveryTopic) -> schema.DiscoveryReport:
    return schema.DiscoveryReport(
        domain="AI agents",
        range_from="2026-06-10",
        range_to="2026-07-10",
        generated_at="2026-07-10T00:00:00+00:00",
        plan=schema.DiscoveryPlan(
            domain="AI agents",
            category="ai_agent_framework",
            subreddits=["AI_Agents"],
            sources=["reddit", "hackernews"],
        ),
        topics=[topic],
    )


def test_discovery_export_round_trips_angles_and_queue_annotations():
    """The 1.1 fields must carry real values through to_discovery_export."""
    payload = schema.to_discovery_export(_discovery_report(schema.DiscoveryTopic(
        rank=1,
        name="Agent memory protocols",
        why_spiking="Two independent listing items accelerated this week.",
        momentum="new-this-week",
        velocity_score=123.45,
        sources=["hackernews", "reddit"],
        engagement_by_source={"reddit": {"score": 120, "num_comments": 30}},
        command='/last30days "Agent memory protocols"',
        podcast_angle="Why agent memory is the next context-window fight",
        x_article_angle="Agent memory protocols, explained through this week's launches",
        previously_surfaced_count=2,
        last_surfaced="2026-07-03",
        covered=True,
    )))

    assert payload["schema_version"] == "1.1"
    result = payload["results"][0]
    assert result["podcast_angle"] == "Why agent memory is the next context-window fight"
    assert result["x_article_angle"] == (
        "Agent memory protocols, explained through this week's launches"
    )
    assert result["previously_surfaced_count"] == 2
    assert result["last_surfaced"] == "2026-07-03"
    assert result["covered"] is True


def test_discovery_topic_constructs_with_only_pre_existing_fields():
    """Pre-1.1 constructor calls must keep working; new fields default."""
    topic = schema.DiscoveryTopic(
        rank=1,
        name="Agent memory protocols",
        why_spiking="Two independent listing items accelerated this week.",
        momentum="building",
        velocity_score=10.0,
        sources=["reddit"],
        engagement_by_source={"reddit": {"score": 120}},
        command='/last30days "Agent memory protocols"',
    )

    assert topic.podcast_angle is None
    assert topic.x_article_angle is None
    assert topic.previously_surfaced_count == 0
    assert topic.last_surfaced is None
    assert topic.covered is False

    result = schema.to_discovery_export(_discovery_report(topic))["results"][0]
    assert result["podcast_angle"] is None
    assert result["x_article_angle"] is None
    assert result["previously_surfaced_count"] == 0
    assert result["last_surfaced"] is None
    assert result["covered"] is False


def test_discovery_cli_mock_render_has_no_angle_or_pipeline_lines():
    """--mock runs never resolve a reasoning provider, so rendered cards must
    omit the U5 angle and Pipeline lines entirely - and stay deterministic
    across runs (same-day mock fixtures)."""
    def _run_once() -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "skills/last30days/scripts/last30days.py",
                "--discover",
                "AI agents",
                "--mock",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    first = _run_once()
    second = _run_once()
    assert first.returncode == 0, first.stderr
    assert "**Podcast angle:**" not in first.stdout
    assert "**X article angle:**" not in first.stdout
    assert "**Pipeline:**" not in first.stdout
    assert first.stdout == second.stdout


def test_discovery_cli_bare_discover_is_global_trending():
    """Bare --discover (no domain) must run global trending, not error."""
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "--mock",
            "--emit=json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["kind"] == "discovery"
    assert payload["domain"] == ""
    assert payload["outcome"] in {"ok", "nothing-solid"}


def test_discovery_cli_shallow_skips_enrichment():
    """--discover-shallow ranks on listing evidence only (still floored)."""
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover", "AI agents",
            "--discover-shallow",
            "--mock",
            "--emit=json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["results"], "shallow mock sweep should still rank mock topics"
    assert all(
        "listing item" in topic["why_spiking"] for topic in payload["results"]
    ), "shallow mode must be judged on listing evidence, not enriched corpora"


def test_discovery_cli_rejects_shallow_without_discover():
    """--discover-shallow on a normal topic run must error, not silently no-op
    into a full research pass (P2 from PR #816 review)."""
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "AI agents",
            "--discover-shallow",
            "--mock",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--discover-shallow only applies to --discover runs" in result.stderr


def test_discovery_cli_rejects_historical_as_of():
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "AI agents",
            "--as-of",
            "2026-06-01",
            "--mock",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--as-of cannot be used with --discover" in result.stderr
    assert "current live listings" in result.stderr


def test_discovery_filters_incompatible_default_sources_but_rejects_explicit_only():
    default_result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "AI agents",
            "--mock",
            "--emit=json",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "LAST30DAYS_DEFAULT_SEARCH": "reddit,x,youtube,hn"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert default_result.returncode == 0, default_result.stderr

    explicit_result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "AI agents",
            "--search=youtube",
            "--mock",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert explicit_result.returncode == 2
    assert "unsupported: youtube" in explicit_result.stderr


def test_detect_category_rejects_suffix_false_positives():
    from lib import categories

    assert categories.detect_category("Dubai agents") is None
    assert categories.detect_category("Thai agents real estate") is None
    assert categories.detect_category("AI agents") == "ai_agent_framework"
    assert categories.detect_category("what's new in ai agent frameworks") == "ai_agent_framework"


def test_discovery_engagement_excludes_rank_metadata():
    from lib import pipeline, schema

    items = [
        schema.SourceItem(
            item_id=f"digg-{i}", source="digg", title="t", body="b",
            url=f"https://di.gg/{i}", published_at="2026-07-05", snippet="s",
            engagement={"postCount": 5, "rank": 100 * (i + 1), "rank_score": 0.5},
        )
        for i in range(3)
    ]
    totals = pipeline._discovery_engagement(items)
    assert totals["digg"]["postCount"] == 15
    assert "rank" not in totals["digg"]
    assert "rank_score" not in totals["digg"]


def test_domain_matching_preserves_non_plural_anchors():
    from lib import pipeline

    assert pipeline._matches_discovery_domain("AI bias", "Addressing AI bias in models")
    assert pipeline._matches_discovery_domain("supply chain crisis", "The crisis deepens for chip supply")
    # Plural matching still works both directions.
    assert pipeline._matches_discovery_domain("AI agents", "The best AI agent stacks")


def test_x_fallback_success_is_clean(monkeypatch):
    from lib import pipeline, env

    calls = []

    def fake_fetch(backend, subquery, from_date, to_date, depth, config):
        calls.append(backend)
        if backend == "bird":
            return [], "cookie expired"
        return [object()], None

    monkeypatch.setattr(pipeline, "_fetch_x_backend", fake_fetch)
    monkeypatch.setattr(env, "x_backend_chain", lambda config: ["bird", "xquik"])
    plan = pipeline.schema.DiscoveryPlan(
        domain="ai agents", category=None, subreddits=[], sources=["x"],
    )
    items, error = pipeline._fetch_discovery_source(
        "x", plan,
        from_date="2026-06-11", to_date="2026-07-11", depth="quick",
        mock=False, config={},
    )
    assert error is None
    assert len(items) == 1
    assert calls == ["bird", "xquik"]


def _digg_envelope(*clusters: dict) -> dict:
    return {"results": list(clusters)}


def _digg_cluster(cluster_id: str, title: str, tldr: str = "") -> dict:
    return {
        "clusterUrlId": cluster_id,
        "title": title,
        "tldr": tldr,
        "rank": 5,
        "postCount": 12,
        "uniqueAuthors": 8,
    }


def test_digg_discovery_drops_off_domain_clusters(monkeypatch):
    """Regression: a crypto sweep surfaced AI stories because the Digg
    branch (an AI-only leaderboard feed) applied no domain filter."""
    envelope = _digg_envelope(
        _digg_cluster("c1", "Bitcoin crypto rally accelerates"),
        _digg_cluster("c2", "OpenAI ships a new frontier model"),
    )
    monkeypatch.setattr(pipeline.digg, "search_digg", lambda *a, **k: envelope)
    plan = schema.DiscoveryPlan(
        domain="crypto", category=None, subreddits=[], sources=["digg"],
    )
    items, error = pipeline._fetch_discovery_source(
        "digg", plan,
        from_date="2026-06-11", to_date="2026-07-11", depth="quick",
        mock=False, config={},
    )
    assert error is None
    titles = [item["title"] for item in items]
    assert titles == ["Bitcoin crypto rally accelerates"]


def test_digg_discovery_keeps_domain_matching_clusters(monkeypatch):
    envelope = _digg_envelope(
        _digg_cluster("c1", "AI agents reshape support desks"),
        _digg_cluster("c2", "The best AI agent stacks compared"),
    )
    monkeypatch.setattr(pipeline.digg, "search_digg", lambda *a, **k: envelope)
    plan = schema.DiscoveryPlan(
        domain="AI agents", category=None, subreddits=[], sources=["digg"],
    )
    items, error = pipeline._fetch_discovery_source(
        "digg", plan,
        from_date="2026-06-11", to_date="2026-07-11", depth="quick",
        mock=False, config={},
    )
    assert error is None
    assert len(items) == 2


def test_digg_discovery_all_filtered_is_clean_no_results(monkeypatch):
    envelope = _digg_envelope(
        _digg_cluster("c1", "OpenAI ships a new frontier model"),
        _digg_cluster("c2", "Anthropic updates its agent SDK"),
    )
    monkeypatch.setattr(pipeline.digg, "search_digg", lambda *a, **k: envelope)
    plan = schema.DiscoveryPlan(
        domain="crypto", category=None, subreddits=[], sources=["digg"],
    )
    items, error = pipeline._fetch_discovery_source(
        "digg", plan,
        from_date="2026-06-11", to_date="2026-07-11", depth="quick",
        mock=False, config={},
    )
    assert error is None
    assert items == []


def test_x_discovery_preserves_producing_backends_own_error(monkeypatch):
    """A backend that returns items plus its own error is a partial outcome;
    only earlier failed-over backends' errors are observability-only."""
    monkeypatch.setattr(
        pipeline, "_fetch_x_backend",
        lambda *a, **k: ([{"id": "x-1", "title": "t"}], "rate limited after page 1"),
    )
    monkeypatch.setattr(pipeline.env, "x_backend_chain", lambda config: ["bird"])
    plan = schema.DiscoveryPlan(
        domain="ai agents", category=None, subreddits=[], sources=["x"],
    )
    items, error = pipeline._fetch_discovery_source(
        "x", plan,
        from_date="2026-06-11", to_date="2026-07-11", depth="quick",
        mock=False, config={},
    )
    assert len(items) == 1
    assert error == "rate limited after page 1"


# --- U6 persistent topic queue: discovery persistence hook + queue CLI -------


def _queue_topic(rank: int, name: str) -> schema.DiscoveryTopic:
    return schema.DiscoveryTopic(
        rank=rank,
        name=name,
        why_spiking=f"Listing evidence about {name}.",
        momentum="building",
        velocity_score=42.5,
        sources=["reddit"],
        engagement_by_source={"reddit": {"score": 120}},
        command=f'/last30days "{name}"',
    )


def _queue_report(names: list[str]) -> schema.DiscoveryReport:
    return schema.DiscoveryReport(
        domain="AI agents",
        range_from="2026-06-20",
        range_to="2026-07-20",
        generated_at="2026-07-20T00:00:00+00:00",
        plan=schema.DiscoveryPlan(
            domain="AI agents", category=None, subreddits=["all"],
            sources=["reddit"],
        ),
        topics=[_queue_topic(rank, name) for rank, name in enumerate(names, start=1)],
    )


def _run_scoped_discover(save_dir, config=None, names=("Gemma 4 chat templates",)):
    parser = cli.build_parser()
    args, _extra = parser.parse_known_args(
        ["--discover", "AI agents", "--save-dir", str(save_dir), "--save-suffix", os.urandom(4).hex()]
    )
    with mock.patch.object(pipeline, "run_discover", return_value=_queue_report(list(names))):
        return cli._run_discover(args, dict(config or {}))


def test_discovery_run_records_surfacings_in_scoped_db_only(tmp_path, monkeypatch, capsys):
    import store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "global" / "research.db")
    save_dir = tmp_path / "client"
    save_dir.mkdir()

    assert _run_scoped_discover(save_dir) == 0
    first = capsys.readouterr().out
    assert "**Pipeline:**" not in first  # nothing prior to annotate from

    scoped_db = save_dir / "research.db"
    assert scoped_db.is_file()
    assert not (tmp_path / "global" / "research.db").exists()

    import sqlite3
    conn = sqlite3.connect(scoped_db)
    rows = conn.execute(
        "SELECT name, surface_count, status FROM discovery_topics"
    ).fetchall()
    conn.close()
    assert rows == [("Gemma 4 chat templates", 1, "surfaced")]


def test_second_discovery_run_annotates_from_prior_state_then_records(tmp_path, capsys):
    save_dir = tmp_path / "client"
    save_dir.mkdir()

    assert _run_scoped_discover(save_dir) == 0
    capsys.readouterr()
    assert _run_scoped_discover(save_dir) == 0
    second = capsys.readouterr().out

    # Annotation reflects the state BEFORE this run's own surfacing was
    # recorded: one prior surfacing means this appearance is the 2nd.
    assert "**Pipeline:** surfaced 2nd time" in second

    import sqlite3
    conn = sqlite3.connect(save_dir / "research.db")
    count = conn.execute(
        "SELECT surface_count FROM discovery_topics WHERE name = ?",
        ("Gemma 4 chat templates",),
    ).fetchone()[0]
    conn.close()
    assert count == 2


def test_covered_topic_resurfacing_renders_marked_covered(tmp_path, capsys):
    import store

    save_dir = tmp_path / "client"
    save_dir.mkdir()

    assert _run_scoped_discover(save_dir) == 0
    with store.scoped_db(save_dir / "research.db"):
        assert store.mark_discovery_covered(
            "Gemma 4 chat templates", as_of="2026-07-20"
        ) is not None
    capsys.readouterr()

    assert _run_scoped_discover(save_dir) == 0
    rendered = capsys.readouterr().out
    assert "marked covered" in rendered
    assert "**Pipeline:** surfaced 2nd time, marked covered" in rendered


def test_queue_opt_out_via_process_env_seam(tmp_path, monkeypatch, capsys):
    from lib import env

    monkeypatch.setenv("LAST30DAYS_DISCOVERY_QUEUE", "off")
    monkeypatch.setattr(env, "CONFIG_FILE", tmp_path / "does-not-exist.env")
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(env, "_load_keychain", return_value={}), \
         mock.patch.object(env, "_load_pass", return_value={}):
        config = env.get_config()
    assert config["LAST30DAYS_DISCOVERY_QUEUE"] == "off"

    save_dir = tmp_path / "client"
    save_dir.mkdir()
    assert _run_scoped_discover(save_dir, config=config) == 0
    assert "**Pipeline:**" not in capsys.readouterr().out
    assert not (save_dir / "research.db").exists()


def test_queue_opt_out_via_env_file_seam(tmp_path, monkeypatch, capsys):
    from lib import env

    monkeypatch.delenv("LAST30DAYS_DISCOVERY_QUEUE", raising=False)
    env_file = tmp_path / "config.env"
    env_file.write_text("LAST30DAYS_DISCOVERY_QUEUE=off\n", encoding="utf-8")
    monkeypatch.setattr(env, "CONFIG_FILE", env_file)
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(env, "_load_keychain", return_value={}), \
         mock.patch.object(env, "_load_pass", return_value={}):
        config = env.get_config()
    assert config["LAST30DAYS_DISCOVERY_QUEUE"] == "off"

    save_dir = tmp_path / "client"
    save_dir.mkdir()
    assert _run_scoped_discover(save_dir, config=config) == 0
    assert not (save_dir / "research.db").exists()


def test_discovery_queue_failure_never_crashes_a_finished_run(tmp_path, monkeypatch, capsys):
    """P0: a broken research.db (locked, read-only, corrupt) must not destroy
    a finished multi-minute pipeline run - the brief still renders (exit 0)
    with a stderr warning, and queue fields keep their defaults."""
    import sqlite3

    import store

    def _locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "record_discovery_surfacing", _locked)
    save_dir = tmp_path / "client"
    save_dir.mkdir()

    assert _run_scoped_discover(save_dir) == 0
    captured = capsys.readouterr()
    assert "## 1. Gemma 4 chat templates" in captured.out
    assert "**Pipeline:**" not in captured.out
    assert "[last30days] Warning:" in captured.err
    assert "database is locked" in captured.err


def test_sibling_topics_in_same_run_do_not_cross_annotate(tmp_path, capsys):
    """Annotations describe the queue state BEFORE this run: two same-anchor
    siblings surfaced by ONE run must not fuzzy-match each other's rows and
    render a false 'surfaced 2nd time' on first-ever topics."""
    import sqlite3

    save_dir = tmp_path / "client"
    save_dir.mkdir()

    assert _run_scoped_discover(
        save_dir, names=("Gemma 4 chat templates", "Gemma 4 enterprise")
    ) == 0
    rendered = capsys.readouterr().out
    assert "## 1. Gemma 4 chat templates" in rendered
    assert "## 2. Gemma 4 enterprise" in rendered
    assert "**Pipeline:**" not in rendered

    conn = sqlite3.connect(save_dir / "research.db")
    rows = conn.execute(
        "SELECT name, surface_count FROM discovery_topics ORDER BY name"
    ).fetchall()
    conn.close()
    assert rows == [("Gemma 4 chat templates", 1), ("Gemma 4 enterprise", 1)]


def test_discovery_mock_run_writes_no_research_db(tmp_path):
    """--mock stays 100% side-effect-free: no queue writes, no research.db."""
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "--discover",
            "AI agents",
            "--mock",
            "--save-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "research.db").exists()


def test_queue_list_shows_uncovered_only_by_default(tmp_path, monkeypatch, capsys):
    import store

    save_dir = tmp_path / "client"
    save_dir.mkdir()
    with store.scoped_db(save_dir / "research.db"):
        store.record_discovery_surfacing(
            "Gemma 4 chat templates", domain="AI agents", run_ref="r1", as_of="2026-07-19",
        )
        store.record_discovery_surfacing(
            "OpenAI Agent SDK", domain="AI agents", run_ref="r1", as_of="2026-07-20",
        )
        store.mark_discovery_covered("OpenAI Agent SDK", as_of="2026-07-20")

    monkeypatch.setattr(cli.env, "get_config", lambda **_kwargs: {})
    monkeypatch.setattr(
        sys, "argv", ["last30days.py", "queue", "list", "--save-dir", str(save_dir)]
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "Gemma 4 chat templates" in out
    assert "OpenAI Agent SDK" not in out
    for column in ("name", "domain", "surface_count", "last_surfaced", "status"):
        assert column in out


def test_queue_list_empty_db_reports_no_recorded_runs(tmp_path, monkeypatch, capsys):
    """A db that exists but has zero discovery rows (e.g. created via --store
    by a topic run) must not claim every topic is covered."""
    import store

    save_dir = tmp_path / "client"
    save_dir.mkdir()
    store.init_db(save_dir / "research.db")

    monkeypatch.setattr(cli.env, "get_config", lambda **_kwargs: {})
    monkeypatch.setattr(
        sys, "argv", ["last30days.py", "queue", "list", "--save-dir", str(save_dir)]
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "no discovery run has recorded topics yet" in out
    assert "marked covered" not in out


def test_queue_cover_marks_topic_covered(tmp_path, monkeypatch, capsys):
    import sqlite3

    import store

    save_dir = tmp_path / "client"
    save_dir.mkdir()
    with store.scoped_db(save_dir / "research.db"):
        store.record_discovery_surfacing(
            "Gemma 4 chat templates", domain="AI agents", run_ref="r1", as_of="2026-07-19",
        )

    monkeypatch.setattr(cli.env, "get_config", lambda **_kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["last30days.py", "queue", "cover", "Gemma 4 chat templates", "--save-dir", str(save_dir)],
    )
    assert cli.main() == 0

    conn = sqlite3.connect(save_dir / "research.db")
    status, covered_at = conn.execute(
        "SELECT status, covered_at FROM discovery_topics WHERE name = ?",
        ("Gemma 4 chat templates",),
    ).fetchone()
    conn.close()
    assert status == "covered"
    assert covered_at


def test_queue_cover_unknown_name_exits_2_with_stderr(tmp_path, monkeypatch, capsys):
    import store

    save_dir = tmp_path / "client"
    save_dir.mkdir()
    with store.scoped_db(save_dir / "research.db"):
        store.record_discovery_surfacing(
            "Gemma 4 chat templates", domain="AI agents", run_ref="r1", as_of="2026-07-19",
        )

    monkeypatch.setattr(cli.env, "get_config", lambda **_kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["last30days.py", "queue", "cover", "No Such Topic", "--save-dir", str(save_dir)],
    )
    assert cli.main() == 2
    err = capsys.readouterr().err
    assert "No Such Topic" in err


def test_queue_cover_cli_unknown_name_subprocess_exit_code(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "skills/last30days/scripts/last30days.py",
            "queue",
            "cover",
            "No Such Topic",
            "--save-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "No Such Topic" in result.stderr


def test_discovery_exits_when_configured_sources_have_no_discovery_feed(monkeypatch, capsys):
    """A configured source boundary must hold: never silently widen a sweep
    to feeds the user filtered out."""
    monkeypatch.setattr(
        cli.env, "get_config", lambda **_kwargs: {"LAST30DAYS_DEFAULT_SEARCH": "youtube"}
    )
    monkeypatch.setattr(sys, "argv", ["last30days.py", "--discover", "AI agents", "--mock"])
    with mock.patch.object(pipeline, "run_discover") as run:
        assert cli.main() == 2

    run.assert_not_called()
    err = capsys.readouterr().err
    assert "no discovery-capable sources" in err
    assert "reddit" in err
