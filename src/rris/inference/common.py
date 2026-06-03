# -*- coding: utf-8 -*-
"""Shared inference utilities (ratings, colors, artifact checks)."""

from __future__ import annotations

import sys

import numpy as np

RATING_CLASSES = np.array([1, 2, 3, 4, 5], dtype=np.float32)

HEX_COLORS = {
    1: "#e53935",
    2: "#ff9800",
    3: "#fbc02d",
    4: "#4caf50",
    5: "#00bcd4",
}

ABSA_KEYWORDS = {
    "food": [
        "อร่อย", "รสชาติ", "เค็ม", "หวาน", "เผ็ด", "เนื้อ", "หมู", "ไก่", "เส้น",
        "น้ำซุป", "อาหาร", "จาน", "สด", "คาว", "หอม", "จืด", "เปรี้ยว", "รส", "แซ่บ",
        "นัว", "กรอบ", "นุ่ม",
    ],
    "service": [
        "บริการ", "พนักงาน", "เด็กเสิร์ฟ", "เสิร์ฟ", "พูดจา", "ดูแล", "ช้า", "เร็ว",
        "มารยาท", "เจ้าของ", "รับออเดอร์", "ต้อนรับ", "รอ", "คิว",
    ],
    "atmosphere": [
        "บรรยากาศ", "ร้าน", "แอร์", "ที่จอดรถ", "ห้องน้ำ", "สะอาด", "สกปรก", "ร้อน",
        "วิว", "ตกแต่ง", "เพลง", "โต๊ะ", "เก้าอี้", "กว้าง", "แคบ", "สบาย", "ที่นั่ง", "มุม",
    ],
}


def get_hex_color(rating: float) -> str:
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return HEX_COLORS[3]
    clamped = max(1, min(5, int(np.round(float(rating)))))
    return HEX_COLORS[clamped]


def expected_rating_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.dot(probs, RATING_CLASSES)


def gaussian_probs_from_expected(expected_vals: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Synthetic 5-class probs from regression-style expected ratings."""
    classes = np.arange(1, 6)
    dist = np.exp(-((classes - expected_vals[:, None]) ** 2) / sigma)
    return dist / np.sum(dist, axis=-1, keepdims=True)


def exit_missing_artifacts(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def head_tail_truncate_text(text: str, tokenizer, max_length: int) -> str:
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_length - 2:
        return text
    head_len = (max_length - 2) // 2
    tail_len = (max_length - 2) - head_len
    tokens = tokens[:head_len] + tokens[-tail_len:]
    return tokenizer.convert_tokens_to_string(tokens)
