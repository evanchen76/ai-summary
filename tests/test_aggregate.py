from datetime import UTC, datetime, timedelta

from ai_review_summary.aggregate import aggregate_aspects, select_snippets
from ai_review_summary.models import Sentiment, StoredAspect

NOW = datetime(2026, 11, 20, tzinfo=UTC)


def record(
    review_id: str,
    aspect: str,
    sentiment: Sentiment,
    snippet: str,
    days_ago: int,
) -> StoredAspect:
    return StoredAspect(
        review_id=review_id,
        aspect=aspect,
        sentiment=sentiment,
        snippet=snippet,
        entity_id="hotel_001",
        created_at=NOW - timedelta(days=days_ago),
    )


def test_recent_records_weigh_more_than_old_ones() -> None:
    records = [
        record("r1", "地點交通", "positive", "近車站", 10),
        record("r2", "地點交通", "negative", "要轉車", 400),
        record("r3", "地點交通", "positive", "機場巴士直達", 500),
    ]
    [stat] = aggregate_aspects(records, ["地點交通"], NOW)
    assert stat.mention_review_count == 3
    assert stat.weighted_pos == 1.5 + 1.0
    assert stat.weighted_neg == 1.0
    assert stat.positive_ratio == 0.71  # 2.5 / 3.5


def test_neutral_only_aspect_has_no_positive_ratio() -> None:
    records = [record(f"s{i}", "客人建議", "neutral", f"建議{i}", i) for i in range(3)]
    [stat] = aggregate_aspects(records, ["客人建議"], NOW)
    assert stat.positive_ratio is None


def test_llm_view_hides_counts_and_weighted_scores() -> None:
    records = [record(f"r{i}", "地點交通", "positive", f"正評{i}", i) for i in range(3)]
    [stat] = aggregate_aspects(records, ["地點交通"], NOW)
    assert set(stat.llm_view()) == {"aspect", "positive_ratio", "sample_snippets"}


def test_aspect_with_fewer_than_three_reviews_is_dropped() -> None:
    records = [
        record("r1", "周邊環境", "positive", "樓下超商", 5),
        record("r1", "周邊環境", "negative", "晚上有車聲", 5),
        record("r2", "周邊環境", "positive", "宵夜多", 6),
    ]
    assert aggregate_aspects(records, ["周邊環境"], NOW) == []


def test_snippets_follow_sentiment_ratio_and_keep_minority() -> None:
    records = [
        record(f"p{i}", "地點交通", "positive", f"正評{i}", i) for i in range(9)
    ] + [record("n0", "地點交通", "negative", "尖峰塞車", 50)]
    picked = select_snippets(records, weighted_pos=9, weighted_neg=1, limit=8)
    assert len(picked) == 8
    assert "尖峰塞車" in picked
    # 正評取最新的 7 句
    assert picked[:7] == [f"正評{i}" for i in range(7)]


def test_neutral_only_aspect_takes_latest_snippets() -> None:
    records = [record(f"s{i}", "客人建議", "neutral", f"建議{i}", i) for i in range(10)]
    picked = select_snippets(records, weighted_pos=0, weighted_neg=0, limit=8)
    assert picked == [f"建議{i}" for i in range(8)]


def test_duplicate_snippets_are_collapsed() -> None:
    records = [record(f"r{i}", "地點交通", "positive", "離車站近", i) for i in range(5)]
    assert select_snippets(records, 5, 0, 8) == ["離車站近"]
