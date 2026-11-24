"""各品類的固定關鍵字維度（Closed-Set），不隨評論動態新增。"""

from __future__ import annotations

from .models import EntityType

ASPECTS: dict[EntityType, dict[str, str]] = {
    "hotel": {
        "地點交通": "捷運／車站距離、機場接駁、停車",
        "性價比與整體舒適度": "房價划算程度、隔音、床鋪與空間清潔",
        "周邊環境": "夜市、超商、餐廳機能、治安與安靜程度",
        "客人建議": "選房型技巧、入住提醒、避雷建議",
    },
    "attraction": {
        "體驗及觀感": "景色、氣氛、動線、整體好玩程度",
        "遊玩建議": "最佳時段、停留時間、拍照點、門票預約",
        "配套及服務": "導覽員、接駁車、休息區、餐飲與寄物",
        "歷史文化": "背景故事、建築特色、文化導覽深度",
        "注意事項": "天候限制、季節停駛、排隊人潮、服裝與體力要求",
    },
}


def aspect_names(entity_type: EntityType) -> list[str]:
    return list(ASPECTS[entity_type])
