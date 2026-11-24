from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_review_summary.aspects import aspect_names
from ai_review_summary.graph import PipelineConfig, build_graph, run_pipeline
from ai_review_summary.models import (
    AspectExtraction,
    AspectStat,
    AspectSummary,
    Entity,
    EntityType,
    QCResult,
    Review,
)
from ai_review_summary.repository import InMemoryRepository

NOW = datetime(2026, 11, 20, tzinfo=UTC)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class FakeReviewLLM:
    """每則評論萃取成第一個維度；QC 依序回傳 `qc_results`。"""

    model_version = "fake"

    def __init__(self, qc_results: list[bool] | None = None) -> None:
        self.qc_results = list(qc_results or [True])
        self.extracted_ids: list[str] = []
        self.feedbacks: list[str] = []

    def extract(
        self, entity_type: EntityType, reviews: list[Review]
    ) -> list[AspectExtraction]:
        self.extracted_ids += [review.id for review in reviews]
        aspect = aspect_names(entity_type)[0]
        return [
            AspectExtraction(
                review_id=review.id,
                aspect=aspect,
                sentiment="positive" if review.rating >= 4 else "negative",
                snippet=review.text[:12],
            )
            for review in reviews
        ] + [
            # 不在這批的 review_id 與自創維度都應該被丟掉
            AspectExtraction(
                review_id="ghost", aspect=aspect, sentiment="positive", snippet="x"
            ),
            AspectExtraction(
                review_id=reviews[0].id,
                aspect="自創維度",
                sentiment="positive",
                snippet="x",
            ),
        ]

    def summarize(
        self,
        entity: Entity,
        stats: list[AspectStat],
        feedback: str,
    ) -> list[AspectSummary]:
        self.feedbacks.append(feedback)
        return [AspectSummary(aspect=s.aspect, summary="摘要") for s in stats]

    def review(
        self, stats: list[AspectStat], summaries: list[AspectSummary]
    ) -> QCResult:
        passed = self.qc_results.pop(0) if self.qc_results else True
        return QCResult(
            passed=passed,
            safety="pass",
            groundedness="pass" if passed else "fail",
            reason="" if passed else "提到輸入沒有的溫泉",
        )


def make_review(review_id: str, days_ago: int, text: str = "交通方便") -> Review:
    return Review(
        id=review_id,
        entity_id="hotel_001",
        rating=5,
        created_at=NOW - timedelta(days=days_ago),
        text=text,
    )


def test_first_run_publishes_summary_with_counts_from_code() -> None:
    repo = InMemoryRepository.from_json(DATA_DIR)
    llm = FakeReviewLLM()
    state = run_pipeline(build_graph(llm, repo), "hotel_001", NOW)

    assert state["status"] == "published"
    text_reviews = repo.list_text_reviews_after("hotel_001", None)
    assert sorted(llm.extracted_ids) == sorted(r.id for r in text_reviews)

    summary = repo.get_summary("hotel_001")
    assert summary is not None
    [item] = summary.aspect_summaries
    assert item.aspect == "地點交通"
    assert item.mention_review_count == len(text_reviews)
    assert "（根據" in summary.render()
    # ghost id 與自創維度被過濾
    assert {a.review_id for a in repo.list_aspects("hotel_001")} <= {
        r.id for r in text_reviews
    }
    assert {a.aspect for a in repo.list_aspects("hotel_001")} == {"地點交通"}


def test_batches_are_split_by_batch_size() -> None:
    repo = InMemoryRepository.from_json(DATA_DIR)
    llm = FakeReviewLLM()
    graph = build_graph(llm, repo, PipelineConfig(batch_size=5))
    state = run_pipeline(graph, "hotel_001", NOW)
    batches = [line for line in state["trace"] if line.startswith("Step 2 萃取")]
    expected = -(-len(repo.list_text_reviews_after("hotel_001", None)) // 5)
    assert len(batches) == expected


def test_qc_rejection_feeds_reason_back_to_summarize() -> None:
    repo = InMemoryRepository.from_json(DATA_DIR)
    llm = FakeReviewLLM(qc_results=[False, True])
    state = run_pipeline(build_graph(llm, repo), "hotel_001", NOW)

    assert state["status"] == "published"
    assert state["summary_attempts"] == 2
    assert llm.feedbacks == ["", "提到輸入沒有的溫泉"]


def test_qc_failures_keep_previous_summary() -> None:
    repo = InMemoryRepository.from_json(DATA_DIR)
    llm = FakeReviewLLM(qc_results=[False, False, False])
    state = run_pipeline(build_graph(llm, repo), "hotel_001", NOW)

    assert state["status"] == "qc_failed"
    assert state["summary_attempts"] == 3
    assert repo.get_summary("hotel_001") is None


def test_second_run_only_extracts_new_reviews_and_respects_threshold() -> None:
    repo = InMemoryRepository.from_json(DATA_DIR)
    llm = FakeReviewLLM()
    graph = build_graph(llm, repo)
    run_pipeline(graph, "hotel_001", NOW)
    first_summary = repo.get_summary("hotel_001")

    # 沒有新評論：不萃取、不重算
    llm.extracted_ids.clear()
    state = run_pipeline(graph, "hotel_001", NOW + timedelta(days=1))
    assert state["status"] == "skipped"
    assert llm.extracted_ids == []

    # 少量新評論：只增量萃取，未達 15 則門檻不重寫摘要
    later = NOW + timedelta(days=2)
    repo.add_reviews(
        [
            make_review("new_1", days_ago=-2),
            make_review("new_2", days_ago=-2, text=""),
        ]
    )
    state = run_pipeline(graph, "hotel_001", later)
    assert llm.extracted_ids == ["new_1"]
    assert state["status"] == "skipped"
    assert repo.get_summary("hotel_001") == first_summary

    # 距上次摘要超過 7 天且有新評論：重算
    state = run_pipeline(graph, "hotel_001", NOW + timedelta(days=8))
    assert state["status"] == "published"
    assert llm.extracted_ids == ["new_1"]


def test_reaching_new_review_threshold_triggers_resummarize() -> None:
    repo = InMemoryRepository.from_json(DATA_DIR)
    llm = FakeReviewLLM()
    graph = build_graph(llm, repo)
    run_pipeline(graph, "hotel_001", NOW)

    repo.add_reviews([make_review(f"new_{i}", days_ago=-1) for i in range(15)])
    state = run_pipeline(graph, "hotel_001", NOW + timedelta(days=1))
    assert state["status"] == "published"


def test_too_few_mentions_produces_no_summary() -> None:
    repo = InMemoryRepository(
        [Entity(id="hotel_001", name="小飯店", type="hotel")],
        [make_review("r1", 1), make_review("r2", 2)],
    )
    state = run_pipeline(build_graph(FakeReviewLLM(), repo), "hotel_001", NOW)
    assert state["status"] == "insufficient_data"
    assert repo.get_summary("hotel_001") is None
