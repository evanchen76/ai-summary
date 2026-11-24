"""用 data/ 的 JSON 假資料跑一次完整管線。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .graph import build_graph, run_pipeline
from .llm import ClaudeReviewLLM
from .repository import InMemoryRepository

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="旅遊商品 AI 評論摘要範例")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--entity", action="append", help="只處理指定商品 id，可重複指定"
    )
    parser.add_argument("--json", action="store_true", help="輸出寫入 DB 的 JSON")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("找不到 ANTHROPIC_API_KEY，請先 cp .env.example .env 並填入 key。")

    repo = InMemoryRepository.from_json(args.data_dir)
    llm = ClaudeReviewLLM()
    graph = build_graph(llm, repo)
    now = datetime.now(UTC)

    entity_ids = args.entity or [entity.id for entity in repo.list_entities()]
    for entity_id in entity_ids:
        entity = repo.get_entity(entity_id)
        print(f"\n=== {entity.name}（{entity_id}）===")
        state = run_pipeline(graph, entity_id, now)
        print("執行軌跡：")
        for line in state.get("trace", []):
            print(f"- {line}")
        print(f"\n狀態：{state.get('status')}")
        summary = repo.get_summary(entity_id)
        if summary is None:
            continue
        print()
        print(summary.model_dump_json(indent=2) if args.json else summary.render())


if __name__ == "__main__":
    main()
