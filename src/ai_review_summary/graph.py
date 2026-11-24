"""LangGraph 管線：Step 1 ～ Step 5。

fetch_new_reviews ─▶ [Send ×N] extract_batch ─▶ store_extractions
                                                   │
                                    should_resummarize?
                                    ├─ 否 ─▶ skip
                                    └─ 是 ─▶ aggregate ─▶ summarize ─▶ qc
                                                           ▲          │
                                                           └─ 退回 ───┤
                                                     通過 ─▶ publish ◀┤
                                         重試用完 ─▶ keep_previous ◀──┘
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from .aggregate import AggregateConfig, aggregate_aspects
from .aspects import aspect_names
from .llm import ReviewLLM
from .models import (
    AspectExtraction,
    AspectStat,
    AspectSummary,
    Entity,
    EntitySummary,
    EntityType,
    PublishedAspect,
    QCResult,
    Review,
    StoredAspect,
)
from .repository import Repository

Status = Literal["published", "skipped", "insufficient_data", "qc_failed"]


@dataclass(frozen=True)
class PipelineConfig:
    batch_size: int = 25
    rerun_min_new_reviews: int = 15
    rerun_max_age_days: int = 7
    max_summary_attempts: int = 3
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)


class PipelineState(TypedDict, total=False):
    entity_id: str
    now: datetime
    entity: Entity
    new_reviews: list[Review]
    extractions: Annotated[list[AspectExtraction], operator.add]
    stats: list[AspectStat]
    summaries: list[AspectSummary]
    qc: QCResult
    qc_feedback: str
    summary_attempts: int
    status: Status
    result: EntitySummary
    trace: Annotated[list[str], operator.add]


class BatchState(TypedDict):
    entity_type: EntityType
    batch: list[Review]


def build_graph(
    llm: ReviewLLM,
    repo: Repository,
    config: PipelineConfig | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    config = config or PipelineConfig()

    # Step 1：只撈高水位線之後、有文字的評論。
    def fetch_new_reviews(state: PipelineState) -> dict[str, Any]:
        entity = repo.get_entity(state["entity_id"])
        progress = repo.get_progress(entity.id)
        reviews = repo.list_text_reviews_after(entity.id, progress.last_processed_at)
        return {
            "entity": entity,
            "new_reviews": reviews,
            "summary_attempts": 0,
            "qc_feedback": "",
            "trace": [f"Step 1 撈取：{len(reviews)} 則有文字的新評論"],
        }

    def fan_out_batches(state: PipelineState) -> list[Send] | str:
        reviews = state["new_reviews"]
        if not reviews:
            return "store_extractions"
        size = config.batch_size
        return [
            Send(
                "extract_batch",
                {"entity_type": state["entity"].type, "batch": reviews[i : i + size]},
            )
            for i in range(0, len(reviews), size)
        ]

    # Step 2（Map）：每批平行萃取，結果經 reducer 合併回 extractions。
    def extract_batch(state: BatchState) -> dict[str, Any]:
        allowed_ids = {review.id for review in state["batch"]}
        allowed_aspects = set(aspect_names(state["entity_type"]))
        extractions = [
            item
            for item in llm.extract(state["entity_type"], state["batch"])
            if item.review_id in allowed_ids and item.aspect in allowed_aspects
        ]
        return {
            "extractions": extractions,
            "trace": [
                f"Step 2 萃取：{len(state['batch'])} 則評論 → {len(extractions)} 筆面向"
            ],
        }

    def store_extractions(state: PipelineState) -> dict[str, Any]:
        entity = state["entity"]
        reviews = state["new_reviews"]
        if not reviews:
            return {}
        created_at = {review.id: review.created_at for review in reviews}
        repo.add_aspects(
            [
                StoredAspect(
                    **item.model_dump(),
                    entity_id=entity.id,
                    created_at=created_at[item.review_id],
                )
                for item in state.get("extractions", [])
            ]
        )
        progress = repo.get_progress(entity.id)
        progress.last_processed_at = max(created_at.values())
        progress.pending_review_count += len(reviews)
        repo.save_progress(entity.id, progress)
        return {
            "trace": [
                f"Step 2 寫入：處理進度更新到 "
                f"{progress.last_processed_at:%Y-%m-%d %H:%M}"
                f"，距上次摘要累積 {progress.pending_review_count} 則"
            ]
        }

    def should_resummarize(state: PipelineState) -> str:
        entity_id = state["entity"].id
        progress = repo.get_progress(entity_id)
        if progress.pending_review_count == 0:
            return "skip"
        if repo.get_summary(entity_id) is None or progress.last_summary_at is None:
            return "aggregate"
        if progress.pending_review_count >= config.rerun_min_new_reviews:
            return "aggregate"
        age = state["now"] - progress.last_summary_at
        if age >= timedelta(days=config.rerun_max_age_days):
            return "aggregate"
        return "skip"

    def skip(state: PipelineState) -> dict[str, Any]:
        return {"status": "skipped", "trace": ["未達重算門檻，沿用現有摘要"]}

    # Step 3：程式聚合，不呼叫 LLM。
    def aggregate(state: PipelineState) -> dict[str, Any]:
        entity = state["entity"]
        stats = aggregate_aspects(
            repo.list_aspects(entity.id),
            aspect_names(entity.type),
            state["now"],
            config.aggregate,
        )
        return {
            "stats": stats,
            "trace": [
                "Step 3 聚合："
                + "、".join(f"{s.aspect} {s.mention_review_count} 則" for s in stats)
            ],
        }

    def route_after_aggregate(state: PipelineState) -> str:
        return "summarize" if state["stats"] else "insufficient_data"

    def insufficient_data(state: PipelineState) -> dict[str, Any]:
        return {
            "status": "insufficient_data",
            "trace": ["每個維度提及數都不足，暫不產出摘要"],
        }

    # Step 4（Reduce）：只看統計與短句，不接觸原始評論。
    def summarize(state: PipelineState) -> dict[str, Any]:
        attempt = state["summary_attempts"] + 1
        stat_aspects = {stat.aspect for stat in state["stats"]}
        summaries = [
            item
            for item in llm.summarize(
                state["entity"],
                state["stats"],
                state["qc_feedback"],
            )
            if item.aspect in stat_aspects
        ]
        return {
            "summaries": summaries,
            "summary_attempts": attempt,
            "trace": [f"Step 4 摘要第 {attempt} 版：{len(summaries)} 個維度"],
        }

    def qc(state: PipelineState) -> dict[str, Any]:
        result = llm.review(state["stats"], state["summaries"])
        verdict = "通過" if result.passed else f"退回（{result.reason}）"
        return {
            "qc": result,
            "qc_feedback": result.reason,
            "trace": [f"Step 4 QC：{verdict}"],
        }

    def route_after_qc(state: PipelineState) -> str:
        if state["qc"].passed:
            return "publish"
        if state["summary_attempts"] < config.max_summary_attempts:
            return "summarize"
        return "keep_previous"

    def keep_previous(state: PipelineState) -> dict[str, Any]:
        return {
            "status": "qc_failed",
            "trace": ["QC 重試用完，不上線新版，保留現有摘要"],
        }

    # Step 5：數字由程式組裝，不交給 LLM。
    def publish(state: PipelineState) -> dict[str, Any]:
        entity = state["entity"]
        counts = {s.aspect: s.mention_review_count for s in state["stats"]}
        summary = EntitySummary(
            entity_id=entity.id,
            entity_type=entity.type,
            has_summary=True,
            generated_at=state["now"],
            model_version=llm.model_version,
            aspect_summaries=[
                PublishedAspect(
                    aspect=item.aspect,
                    mention_review_count=counts[item.aspect],
                    summary=item.summary,
                )
                for item in state["summaries"]
            ],
        )
        repo.save_summary(summary)
        progress = repo.get_progress(entity.id)
        progress.pending_review_count = 0
        progress.last_summary_at = state["now"]
        repo.save_progress(entity.id, progress)
        return {
            "status": "published",
            "result": summary,
            "trace": ["Step 5 寫入摘要"],
        }

    graph = StateGraph(PipelineState)
    graph.add_node("fetch_new_reviews", fetch_new_reviews)
    graph.add_node("extract_batch", extract_batch)
    graph.add_node("store_extractions", store_extractions)
    graph.add_node("skip", skip)
    graph.add_node("aggregate", aggregate)
    graph.add_node("insufficient_data", insufficient_data)
    graph.add_node("summarize", summarize)
    graph.add_node("qc", qc)
    graph.add_node("keep_previous", keep_previous)
    graph.add_node("publish", publish)

    graph.add_edge(START, "fetch_new_reviews")
    graph.add_conditional_edges(
        "fetch_new_reviews", fan_out_batches, ["extract_batch", "store_extractions"]
    )
    graph.add_edge("extract_batch", "store_extractions")
    graph.add_conditional_edges(
        "store_extractions",
        should_resummarize,
        {"aggregate": "aggregate", "skip": "skip"},
    )
    graph.add_conditional_edges(
        "aggregate",
        route_after_aggregate,
        ["summarize", "insufficient_data"],
    )
    graph.add_edge("summarize", "qc")
    graph.add_conditional_edges(
        "qc", route_after_qc, ["publish", "summarize", "keep_previous"]
    )
    for terminal in ("skip", "insufficient_data", "keep_previous", "publish"):
        graph.add_edge(terminal, END)
    return graph.compile()


def run_pipeline(
    graph: CompiledStateGraph[Any, Any, Any, Any], entity_id: str, now: datetime
) -> PipelineState:
    state = graph.invoke({"entity_id": entity_id, "now": now})
    return cast(PipelineState, state)
