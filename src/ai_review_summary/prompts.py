"""Map、Reduce、QC 三個階段的 prompt。"""

from __future__ import annotations

import json

from .aspects import ASPECTS
from .models import AspectStat, AspectSummary, Entity, EntityType, Review

ENTITY_LABEL: dict[EntityType, str] = {"hotel": "飯店", "attraction": "景點／一日遊"}


def extract_system_prompt(entity_type: EntityType) -> str:
    aspects = ASPECTS[entity_type]
    options = "、".join(f"「{name}」" for name in aspects)
    definitions = "\n".join(f"   - {name}：{desc}" for name, desc in aspects.items())
    return f"""你是一個專業的旅遊評論分析引擎。請閱讀以下{ENTITY_LABEL[entity_type]}評論，萃取出旅客提到的具體面向與正負評價。

執行規則：
1. 固定維度歸類：aspect 欄位只能從以下 {len(aspects)} 個選項擇一填入：{options}。
{definitions}
2. 拆解複合句：一則評論若同時提到多個面向（例如交通方便但晚上有點吵），必須拆分成多筆獨立紀錄。
3. 嚴格溯源：每筆萃取結果必須附上對應的 review_id，並用 12 個字以內濃縮該評論的具體描述（snippet）。
4. 參考星等判斷語氣：rating 是旅客給的星等，用來辨識反諷（例如 1 星卻寫「真是太棒了」）。
5. 建議類內容：具體的提醒或技巧（例如「建議訂高樓層」），sentiment 填 neutral。
6. 忽略無效內容：僅有「讚」、「推」等無意義字元、廣告、詐騙或洗版內容，或未提及上述面向的評論，一律不萃取。"""


def extract_user_prompt(reviews: list[Review]) -> str:
    payload = [{"id": r.id, "rating": r.rating, "text": r.text} for r in reviews]
    return json.dumps(payload, ensure_ascii=False, indent=2)


SUMMARIZE_SYSTEM_PROMPT = """你是一個旅遊平台的「評論摘要編輯」。請根據後端統計出的 aspect_stats，針對每個維度撰寫客觀、精簡且具參考價值的摘要。

撰寫規則：
1. 嚴格基於數據：只能根據提供的 aspect_stats 與 sample_snippets 撰寫，禁止腦補未提及的設施、交通路線或景點規定。
2. 反映真實比例：根據 positive_ratio 決定語氣強度；正負參半或有明顯缺點時，需如實點出注意事項。
3. 結構化輸出：針對輸入的每個 aspect 輸出一句 30～50 字的流暢摘要（建議類維度可濃縮 2～3 項具體提醒）。
4. 語氣規範：保持中立客觀的第三方視角，不使用誇張行銷詞彙，不需在文字中自行加上「根據幾則評論」等數字。"""


def summarize_user_prompt(
    entity: Entity,
    stats: list[AspectStat],
    feedback: str,
) -> str:
    payload = {
        "entity_type": entity.type,
        "aspect_stats": [stat.llm_view() for stat in stats],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if feedback:
        text += f"\n\n上一版摘要未通過品質審查，請修正以下問題後重寫：\n{feedback}"
    return text


QC_SYSTEM_PROMPT = """你是旅遊平台的摘要品質審查員，與撰寫摘要的編輯彼此獨立。請比對 aspect_stats 與 aspect_summaries，逐句檢查：

1. 事實依據性（groundedness）：摘要中的每項陳述，都必須能在該維度的 sample_snippets 找到根據。出現輸入沒有的設施、路線、規定或數字，判定 fail。
2. 安全性（safety）：含不當用語、人身攻擊、歧視或極端措辭，判定 fail。

兩項都 pass 時 passed 才為 true。未通過時在 reason 具體指出是哪個維度的哪句話、問題是什麼，讓編輯可以據此修改。"""


def qc_user_prompt(stats: list[AspectStat], summaries: list[AspectSummary]) -> str:
    payload = {
        "aspect_stats": [
            {"aspect": stat.aspect, "sample_snippets": stat.sample_snippets}
            for stat in stats
        ],
        "aspect_summaries": [summary.model_dump() for summary in summaries],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
