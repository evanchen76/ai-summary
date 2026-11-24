"""Step 3：聚合統計與代表短句挑選。全部是程式計算，不呼叫 LLM。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import AspectStat, StoredAspect


@dataclass(frozen=True)
class AggregateConfig:
    recent_days: int = 180
    recent_weight: float = 1.5
    old_weight: float = 1.0
    min_mentions: int = 3
    max_snippets: int = 8


def record_weight(
    created_at: datetime, now: datetime, config: AggregateConfig
) -> float:
    """近期評論權重較高，避免翻修前的舊負評蓋過近況。"""
    if now - created_at <= timedelta(days=config.recent_days):
        return config.recent_weight
    return config.old_weight


def _latest_unique(records: list[StoredAspect], limit: int) -> list[str]:
    snippets: list[str] = []
    for record in sorted(records, key=lambda item: item.created_at, reverse=True):
        if len(snippets) >= limit:
            break
        if record.snippet not in snippets:
            snippets.append(record.snippet)
    return snippets


def select_snippets(
    records: list[StoredAspect],
    weighted_pos: float,
    weighted_neg: float,
    limit: int,
) -> list[str]:
    """照正負分數比例分配名額，各自取最新的短句。

    只有中立紀錄的維度（例如「客人建議」）直接取最新的 `limit` 句。
    少數意見只要存在就至少保留 1 句，讓摘要能如實點出注意事項。
    """
    positives = [r for r in records if r.sentiment == "positive"]
    negatives = [r for r in records if r.sentiment == "negative"]
    neutrals = [r for r in records if r.sentiment == "neutral"]

    total = weighted_pos + weighted_neg
    if total == 0:
        return _latest_unique(records, limit)

    neg_quota = round(limit * weighted_neg / total)
    if negatives and positives:
        neg_quota = min(max(neg_quota, 1), limit - 1)
    pos_quota = limit - neg_quota

    picked_neg = _latest_unique(negatives, neg_quota)
    picked_pos = _latest_unique(positives, pos_quota)
    picked = picked_pos + [s for s in picked_neg if s not in picked_pos]

    # 某一邊不夠時，用剩下的紀錄補滿名額。
    if len(picked) < limit:
        rest = [r for r in positives + negatives + neutrals if r.snippet not in picked]
        picked += _latest_unique(rest, limit - len(picked))
    return picked


def aggregate_aspects(
    records: list[StoredAspect],
    aspect_order: list[str],
    now: datetime,
    config: AggregateConfig | None = None,
) -> list[AspectStat]:
    config = config or AggregateConfig()
    grouped: dict[str, list[StoredAspect]] = defaultdict(list)
    for record in records:
        grouped[record.aspect].append(record)

    stats: list[AspectStat] = []
    for aspect in aspect_order:
        items = grouped.get(aspect, [])
        mention_count = len({item.review_id for item in items})
        if mention_count < config.min_mentions:
            continue

        weighted_pos = sum(
            record_weight(item.created_at, now, config)
            for item in items
            if item.sentiment == "positive"
        )
        weighted_neg = sum(
            record_weight(item.created_at, now, config)
            for item in items
            if item.sentiment == "negative"
        )
        total = weighted_pos + weighted_neg
        stats.append(
            AspectStat(
                aspect=aspect,
                mention_review_count=mention_count,
                weighted_pos=weighted_pos,
                weighted_neg=weighted_neg,
                positive_ratio=round(weighted_pos / total, 2) if total else None,
                sample_snippets=select_snippets(
                    items, weighted_pos, weighted_neg, config.max_snippets
                ),
            )
        )
    return stats
