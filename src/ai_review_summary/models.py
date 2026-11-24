"""管線各階段交換的資料結構。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["hotel", "attraction"]
Sentiment = Literal["positive", "negative", "neutral"]


class Entity(BaseModel):
    id: str
    name: str
    type: EntityType


class Review(BaseModel):
    """`reviews` 表的一筆原始評論。沒有文字的評分也會存在這裡。"""

    id: str
    entity_id: str
    rating: int = Field(ge=1, le=5)
    created_at: datetime
    text: str = ""


class AspectExtraction(BaseModel):
    """Step 2（Map）從單則評論萃取出的一個面向。"""

    review_id: str
    aspect: str
    sentiment: Sentiment
    snippet: str


class StoredAspect(AspectExtraction):
    """`review_aspects` 表的一筆紀錄。

    `created_at` 沿用原評論的時間，Step 3 用它做時間衰減與挑最新短句。
    """

    entity_id: str
    created_at: datetime


class AspectStat(BaseModel):
    """Step 3 聚合後送給 Step 4 的單一維度統計。"""

    aspect: str
    mention_review_count: int
    weighted_pos: float
    weighted_neg: float
    positive_ratio: float | None
    sample_snippets: list[str]

    def llm_view(self) -> dict[str, object]:
        """給 LLM 看的欄位。

        提及數由後端組裝進最終文字，加權分數是內部值，所以只給 LLM 換算後的
        `positive_ratio` 與代表短句。
        """
        return {
            "aspect": self.aspect,
            "positive_ratio": self.positive_ratio,
            "sample_snippets": self.sample_snippets,
        }


class AspectSummary(BaseModel):
    """Step 4（Reduce）由 LLM 產出的單一維度摘要，不含數字。"""

    aspect: str
    summary: str


class QCResult(BaseModel):
    """獨立 LLM 的品質把關結果。"""

    passed: bool
    safety: Literal["pass", "fail"]
    groundedness: Literal["pass", "fail"]
    reason: str = ""


class PublishedAspect(BaseModel):
    aspect: str
    mention_review_count: int
    summary: str


class EntitySummary(BaseModel):
    """Step 5 寫入 `travel_review_summaries` 的資料，商品頁直接讀取。"""

    entity_id: str
    entity_type: EntityType
    has_summary: bool
    generated_at: datetime
    model_version: str
    aspect_summaries: list[PublishedAspect]

    def render(self) -> str:
        return "\n".join(
            f"- **{item.aspect}**：{item.summary}"
            f"（根據 {item.mention_review_count} 則評論）"
            for item in self.aspect_summaries
        )


class Progress(BaseModel):
    """每個商品一筆的增量處理進度。"""

    last_processed_at: datetime | None = None
    pending_review_count: int = 0
    last_summary_at: datetime | None = None
