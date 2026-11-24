"""資料存取介面。

範例只實作記憶體版本，初始資料從 JSON 載入；正式環境可再實作一個連到
關聯式資料庫的版本，管線本身不用改。
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .models import Entity, EntitySummary, Progress, Review, StoredAspect


class Repository(Protocol):
    def get_entity(self, entity_id: str) -> Entity: ...

    def list_reviews(self, entity_id: str) -> list[Review]: ...

    def list_text_reviews_after(
        self, entity_id: str, after: datetime | None
    ) -> list[Review]: ...

    def add_aspects(self, records: list[StoredAspect]) -> None: ...

    def list_aspects(self, entity_id: str) -> list[StoredAspect]: ...

    def get_progress(self, entity_id: str) -> Progress: ...

    def save_progress(self, entity_id: str, progress: Progress) -> None: ...

    def get_summary(self, entity_id: str) -> EntitySummary | None: ...

    def save_summary(self, summary: EntitySummary) -> None: ...


class InMemoryRepository:
    def __init__(self, entities: list[Entity], reviews: list[Review]) -> None:
        self._entities = {entity.id: entity for entity in entities}
        self._reviews: dict[str, list[Review]] = defaultdict(list)
        self._aspects: dict[str, list[StoredAspect]] = defaultdict(list)
        self._progress: dict[str, Progress] = {}
        self._summaries: dict[str, EntitySummary] = {}
        self.add_reviews(reviews)

    @classmethod
    def from_json(cls, data_dir: Path) -> InMemoryRepository:
        entities = json.loads((data_dir / "entities.json").read_text("utf-8"))
        reviews = json.loads((data_dir / "reviews.json").read_text("utf-8"))
        return cls(
            [Entity.model_validate(item) for item in entities],
            [Review.model_validate(item) for item in reviews],
        )

    def add_reviews(self, reviews: list[Review]) -> None:
        for review in reviews:
            self._reviews[review.entity_id].append(review)

    def list_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def get_entity(self, entity_id: str) -> Entity:
        return self._entities[entity_id]

    def list_reviews(self, entity_id: str) -> list[Review]:
        return list(self._reviews[entity_id])

    def list_text_reviews_after(
        self, entity_id: str, after: datetime | None
    ) -> list[Review]:
        return sorted(
            (
                review
                for review in self._reviews[entity_id]
                if review.text.strip() and (after is None or review.created_at > after)
            ),
            key=lambda review: review.created_at,
        )

    def add_aspects(self, records: list[StoredAspect]) -> None:
        for record in records:
            self._aspects[record.entity_id].append(record)

    def list_aspects(self, entity_id: str) -> list[StoredAspect]:
        return list(self._aspects[entity_id])

    def get_progress(self, entity_id: str) -> Progress:
        return self._progress.get(entity_id, Progress()).model_copy()

    def save_progress(self, entity_id: str, progress: Progress) -> None:
        self._progress[entity_id] = progress

    def get_summary(self, entity_id: str) -> EntitySummary | None:
        return self._summaries.get(entity_id)

    def save_summary(self, summary: EntitySummary) -> None:
        self._summaries[summary.entity_id] = summary
