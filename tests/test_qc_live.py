"""用真實模型驗證 Step 4 的 QC 把關。

QC 的判斷只有真實模型能驗證，所以這些測試預設跳過；
設定 `RUN_LIVE_TESTS=1` 並提供 `ANTHROPIC_API_KEY` 才會執行：

    RUN_LIVE_TESTS=1 uv run pytest tests/test_qc_live.py
"""

import os

import pytest
from dotenv import load_dotenv

from ai_review_summary.llm import ClaudeReviewLLM
from ai_review_summary.models import AspectStat, AspectSummary

load_dotenv()

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="需要 RUN_LIVE_TESTS=1 與 ANTHROPIC_API_KEY",
)


def stat(aspect: str, ratio: float, snippets: list[str]) -> AspectStat:
    return AspectStat(
        aspect=aspect,
        mention_review_count=12,
        weighted_pos=ratio * 10,
        weighted_neg=(1 - ratio) * 10,
        positive_ratio=ratio,
        sample_snippets=snippets,
    )


def test_qc_rejects_facility_not_in_snippets() -> None:
    stats = [stat("周邊環境", 0.9, ["樓下有超商與夜市", "附近宵夜選擇多"])]
    summaries = [
        AspectSummary(
            aspect="周邊環境",
            summary="樓下有超商與夜市，宵夜選擇多；頂樓設有免費露天溫泉。",
        )
    ]
    result = ClaudeReviewLLM().review(stats, summaries)
    assert not result.passed
    assert result.groundedness == "fail"


def test_qc_passes_summary_supported_by_snippets() -> None:
    stats = [stat("周邊環境", 0.9, ["樓下有超商與夜市", "附近宵夜選擇多"])]
    summaries = [
        AspectSummary(aspect="周邊環境", summary="樓下有超商與夜市，宵夜選擇多。")
    ]
    result = ClaudeReviewLLM().review(stats, summaries)
    assert result.passed, result.reason


def test_qc_rejects_summary_that_repeats_a_personal_attack() -> None:
    # 假設萃取漏掉一句辱罵，短句有它，摘要照抄：事實依據成立，只有安全性有問題。
    stats = [
        stat(
            "性價比與整體舒適度",
            0.5,
            ["房間乾淨", "櫃台人員是白痴，態度極差", "冷氣有點吵"],
        )
    ]
    summaries = [
        AspectSummary(
            aspect="性價比與整體舒適度",
            summary="房間乾淨，但櫃台人員是白痴，態度極差；冷氣聲音偏大。",
        )
    ]
    result = ClaudeReviewLLM().review(stats, summaries)
    assert not result.passed
    assert result.safety == "fail"


def test_qc_does_not_flag_strong_but_concrete_complaint_as_unsafe() -> None:
    stats = [
        stat(
            "性價比與整體舒適度",
            0.3,
            ["房間乾淨", "隔音很差，隔壁講話聽得一清二楚", "冷氣有點吵"],
        )
    ]
    summaries = [
        AspectSummary(
            aspect="性價比與整體舒適度",
            summary="房間整潔，但隔音偏弱，隔壁房間的談話聲清晰可聞；冷氣聲音也偏大。",
        )
    ]
    result = ClaudeReviewLLM().review(stats, summaries)
    assert result.safety == "pass", result.reason
