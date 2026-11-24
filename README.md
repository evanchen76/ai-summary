# 旅遊商品 AI 評論摘要

旅客挑飯店或一日遊時，最需要知道的是「各面向具體好在哪、差在哪」，再判斷缺點是不是自己在意的部分。這個專案用 LangGraph 實作兩階段的評論摘要管線：先依品類的固定維度把每則評論萃取成結構化 JSON，程式統計提及數與代表短句後，再交給 LLM 分項寫摘要，並由另一次獨立的 LLM 呼叫做品質把關。

```text
fetch_new_reviews ─▶ [Send ×N] extract_batch ─▶ store_extractions
                                                   │
                                    should_resummarize?
                                    ├─ 否 ─▶ skip
                                    └─ 是 ─▶ aggregate ─▶ summarize ─▶ qc
                                                           ▲          │
                                                           └─ 退回 ───┤
                                                     通過 ─▶ publish ◀┤
                                         重試用完 ─▶ keep_previous ◀──┘
```

對應的概念文章是[〈旅遊商品 AI 評論摘要怎麼做〉](../../../note/ai-idea/旅遊商品%20AI%20評論摘要怎麼做.md)。

## 執行

```bash
uv sync
cp .env.example .env   # 填入 ANTHROPIC_API_KEY
uv run review-summary
```

預設模型是 Claude Haiku 4.5（`claude-haiku-4-5`），可用 `ANTHROPIC_MODEL` 指定其他模型；QC 用的模型可另外用 `ANTHROPIC_QC_MODEL` 指定。其他參數：

```bash
uv run review-summary --entity hotel_001   # 只跑指定商品
uv run review-summary --json               # 印出寫入 DB 的 JSON
```

輸出範例（模型每次措辭會略有不同）：

```text
=== 東京新宿東口飯店（hotel_001）===
執行軌跡：
- Step 1 撈取：26 則有文字的新評論
- Step 2 萃取：25 則評論 → 39 筆面向
- Step 2 萃取：1 則評論 → 2 筆面向
- Step 2 寫入：處理進度更新到 2026-11-18 12:00，距上次摘要累積 26 則
- Step 3 聚合：地點交通 11 則、性價比與整體舒適度 12 則、周邊環境 11 則、客人建議 5 則
- Step 4 摘要第 1 版：4 個維度
- Step 4 QC：通過
- Step 5 寫入摘要

狀態：published

- **地點交通**：位置優越，新宿站東口步行5分鐘可達，鄰近地鐵新宿三丁目站與新宿御苑，機場巴士便利。飯店無停車場，需預約特約停車位。（根據 11 則評論）
- **周邊環境**：餐飲選擇豐富，24小時超市與便利店近在咫尺，治安感受良好。低樓層臨馬路車聲擾人，週末晚間歌舞伎町噪音較大。（根據 11 則評論）
...
```

## 假資料

`data/` 放兩個 JSON 檔，載入到記憶體版的 repository，不需要資料庫：

- `entities.json`：一間飯店（`hotel_001`）、一個一日遊行程（`tour_001`）。
- `reviews.json`：56 則評論，刻意混入沒有文字的評分、只寫「讚」「推推推」或「還行」的評論、沒有具體意見的評論、洗版與離題內容、加 LINE 的廣告、不帶資訊的辱罵，以及夾帶辱罵的具體抱怨與 1 星反諷評論，用來觀察管線怎麼處理。

## 各階段對應

| 筆記步驟 | 節點 | 說明 |
| --- | --- | --- |
| Step 1 撈取新評論 | `fetch_new_reviews` | 依每個商品的 `last_processed_at` 只撈之後、有文字的評論 |
| Step 2 Map 萃取 | `extract_batch`、`store_extractions` | 每 25 則一批，用 `Send` 平行處理；structured output 以 enum 鎖住固定維度，再過濾不在該批的 `review_id` |
| 重算門檻 | `should_resummarize` | 第一次一定產生；之後累積 15 則新評論，或距上次超過 7 天且有新評論才重算 |
| Step 3 聚合 | `aggregate` | 純程式：近 180 天權重 1.5、舊評論 1.0，算出 `weighted_pos`、`weighted_neg` 與 `positive_ratio`；少於 3 則的維度不產摘要；每維度最多 8 句短句，照正負比例分配名額 |
| Step 4 Reduce 與 QC | `summarize`、`qc` | LLM 只收到 `entity_type`，以及每個維度的 `aspect`、`positive_ratio`、`sample_snippets`（`AspectStat.llm_view()`），看不到提及數與加權分數；QC 是另一次獨立呼叫，只看維度與短句，檢查事實依據與安全性，退回時把原因帶回 `summarize` 重寫，最多 3 版 |
| Step 5 寫入 | `publish` | 「根據 N 則評論」的數字由程式從 `mention_review_count` 組裝，不交給 LLM |

各品類的固定維度定義在 `src/ai_review_summary/aspects.py`，三個 prompt 在 `prompts.py`。

## 換成真的資料庫

管線只依賴 `repository.py` 的 `Repository` Protocol。正式環境實作一個連到關聯式資料庫的版本（`reviews`、`review_aspects`、`travel_review_summaries` 與處理進度表），傳給 `build_graph` 即可。

## 開發

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

測試用假的 LLM 回傳固定結果，不需要 API key，涵蓋：聚合加權與短句挑選、批次切分、QC 退回後重寫、QC 重試用完保留舊摘要、增量萃取與重算門檻、提及數不足不產摘要。

`tests/test_extract_live.py` 用真實模型驗證 Step 2 對極端評論的過濾：純辱罵、極短、沒有具體意見、洗版、離題的評論不萃取；夾帶辱罵的具體抱怨只留下具體事實的那一筆。`tests/test_qc_live.py` 驗證 Step 4 的 QC：摘要出現短句沒有的設施（例如頂樓露天溫泉）時 `groundedness` 判定為 `fail`，有短句支持的摘要通過；摘要重複人身攻擊時 `safety` 判定為 `fail`，強烈但具體的抱怨（例如隔音很差）不會被當成不當用語。這些測試會呼叫 API，預設跳過，需要時才執行：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_extract_live.py tests/test_qc_live.py
```
