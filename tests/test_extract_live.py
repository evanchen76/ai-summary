"""用真實模型驗證 Step 2 的過濾。

Prompt 的過濾效果只有真實模型能驗證，所以這些測試預設跳過；
設定 `RUN_LIVE_TESTS=1` 並提供 `ANTHROPIC_API_KEY` 才會執行：

    RUN_LIVE_TESTS=1 uv run pytest tests/test_extract_live.py
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from ai_review_summary.llm import ClaudeReviewLLM
from ai_review_summary.models import AspectExtraction, Review

load_dotenv()

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="需要 RUN_LIVE_TESTS=1 與 ANTHROPIC_API_KEY",
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 沒有可參考資訊的極端評論：辱罵、誇大、極短、沒有具體意見、洗版、離題。
NOISE_IDS = [
    "h_x01",  # 純辱罵
    "h_x03",  # 誇大辱罵，沒有具體內容
    "h_x04",  # 還行
    "h_x05",  # ok
    "h_x06",  # 👍👍👍
    "h_x07",  # 沒有具體意見
    "h_x08",  # 洗版
    "h_x09",  # 離題
    "h_x10",  # 攻擊其他評論者
]
MIXED_ID = "h_x02"  # 夾帶辱罵的具體抱怨（隔音差）


@pytest.fixture(scope="module")
def extractions() -> list[AspectExtraction]:
    rows = json.loads((DATA_DIR / "reviews.json").read_text("utf-8"))
    reviews = [Review(**row) for row in rows if row["id"].startswith("h_x")]
    return ClaudeReviewLLM().extract("hotel", reviews)


def test_noise_reviews_produce_no_extractions(
    extractions: list[AspectExtraction],
) -> None:
    extracted_ids = {item.review_id for item in extractions}
    assert extracted_ids.isdisjoint(NOISE_IDS)


def test_abusive_review_with_concrete_complaint_keeps_only_the_fact(
    extractions: list[AspectExtraction],
) -> None:
    [item] = [e for e in extractions if e.review_id == MIXED_ID]
    assert item.aspect == "性價比與整體舒適度"
    assert item.sentiment == "negative"
    assert "隔音" in item.snippet

