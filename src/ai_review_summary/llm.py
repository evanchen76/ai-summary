"""管線用到的三種 LLM 呼叫。

`ReviewLLM` 是管線依賴的介面；`ClaudeReviewLLM` 用 Claude 實作，測試則換成
回傳固定結果的假物件，不需要 API key。
"""

from __future__ import annotations

import os
from typing import Any, Protocol, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .aspects import aspect_names
from .models import (
    AspectExtraction,
    AspectStat,
    AspectSummary,
    Entity,
    EntityType,
    QCResult,
    Review,
)
from .prompts import (
    QC_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    extract_system_prompt,
    extract_user_prompt,
    qc_user_prompt,
    summarize_user_prompt,
)

DEFAULT_MODEL = "claude-haiku-4-5"


class ReviewLLM(Protocol):
    model_version: str

    def extract(
        self, entity_type: EntityType, reviews: list[Review]
    ) -> list[AspectExtraction]: ...

    def summarize(
        self,
        entity: Entity,
        stats: list[AspectStat],
        feedback: str,
    ) -> list[AspectSummary]: ...

    def review(
        self, stats: list[AspectStat], summaries: list[AspectSummary]
    ) -> QCResult: ...


def _extraction_schema(aspects: list[str]) -> dict[str, Any]:
    """用 enum 鎖住固定維度，模型無法自創新維度。"""
    return {
        "title": "record_extractions",
        "description": "記錄從評論萃取出的面向、正負評價與短句",
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_id": {"type": "string"},
                        "aspect": {"type": "string", "enum": aspects},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"],
                        },
                        "snippet": {"type": "string", "description": "12 字以內"},
                    },
                    "required": ["review_id", "aspect", "sentiment", "snippet"],
                },
            }
        },
        "required": ["extractions"],
    }


def _summary_schema(aspects: list[str]) -> dict[str, Any]:
    return {
        "title": "record_aspect_summaries",
        "description": "記錄每個維度的分項摘要",
        "type": "object",
        "properties": {
            "aspect_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "aspect": {"type": "string", "enum": aspects},
                        "summary": {"type": "string"},
                    },
                    "required": ["aspect", "summary"],
                },
            }
        },
        "required": ["aspect_summaries"],
    }


class ClaudeReviewLLM:
    def __init__(self, model: str | None = None, qc_model: str | None = None) -> None:
        model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        qc_model = qc_model or os.environ.get("ANTHROPIC_QC_MODEL", model)
        self.model_version = model
        self._chat = ChatAnthropic(model=model, temperature=0, max_tokens=4096)
        # QC 是獨立的一次呼叫，可以設定成不同模型。
        self._qc_chat = ChatAnthropic(model=qc_model, temperature=0, max_tokens=2048)

    def extract(
        self, entity_type: EntityType, reviews: list[Review]
    ) -> list[AspectExtraction]:
        runnable = self._chat.with_structured_output(
            _extraction_schema(aspect_names(entity_type))
        )
        result = cast(
            dict[str, Any],
            runnable.invoke(
                [
                    SystemMessage(extract_system_prompt(entity_type)),
                    HumanMessage(extract_user_prompt(reviews)),
                ]
            ),
        )
        return [
            AspectExtraction.model_validate(item)
            for item in result.get("extractions", [])
        ]

    def summarize(
        self,
        entity: Entity,
        stats: list[AspectStat],
        feedback: str,
    ) -> list[AspectSummary]:
        runnable = self._chat.with_structured_output(
            _summary_schema([stat.aspect for stat in stats])
        )
        result = cast(
            dict[str, Any],
            runnable.invoke(
                [
                    SystemMessage(SUMMARIZE_SYSTEM_PROMPT),
                    HumanMessage(summarize_user_prompt(entity, stats, feedback)),
                ]
            ),
        )
        return [
            AspectSummary.model_validate(item)
            for item in result.get("aspect_summaries", [])
        ]

    def review(
        self, stats: list[AspectStat], summaries: list[AspectSummary]
    ) -> QCResult:
        runnable = self._qc_chat.with_structured_output(QCResult)
        return cast(
            QCResult,
            runnable.invoke(
                [
                    SystemMessage(QC_SYSTEM_PROMPT),
                    HumanMessage(qc_user_prompt(stats, summaries)),
                ]
            ),
        )
