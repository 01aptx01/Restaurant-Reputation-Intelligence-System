# -*- coding: utf-8 -*-
"""Text normalization and XLM-R preprocessing strategies."""

from __future__ import annotations

import re

import pandas as pd
from pythainlp.tokenize import word_tokenize

TEXT_ALIASES = ("review_body", "text", "review")
RATING_ALIASES = ("stars", "user_rating", "rating", "star")
NEGATION_PHRASES = ("ไม่แนะนำ", "ไม่ค่อย", "ไม่")

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "]+",
    flags=re.UNICODE,
)

THAI_EMOJI_MAP = {
    "😡": " โกรธ ", "😠": " โกรธ ", "🤬": " ด่า ",
    "🤮": " อ้วก ", "🤢": " แหวะ ", "👎": " แย่ ",
    "😞": " ผิดหวัง ", "😔": " ผิดหวัง ", "😢": " เศร้า ", "😭": " ร้องไห้ ",
    "😊": " ยิ้ม ", "😁": " ดีใจ ", "😍": " ชอบมาก ", "🥰": " รัก ",
    "❤️": " หัวใจ ", "💕": " หัวใจ ", "👍": " เยี่ยม ", "👌": " โอเค ",
    "🤤": " น่ากิน ", "😋": " อร่อย ", "🌟": " ดาว ", "⭐": " ดาว ",
    "✨": " ดีเยี่ยม ", "💯": " เต็มร้อย ", "💔": " ผิดหวัง ",
}

THAI_SLANG_MAP = {
    "นัว": " อร่อยกลมกล่อม ",
    "ฟิน": " มีความสุขมาก ",
    "ดีงาม": " ดีมาก ",
    "ต้องลอง": " แนะนำให้ทาน ",
    "ดรอป": " คุณภาพแย่ลง ",
    "เค็มปี๋": " เค็มมาก ",
    "จืดชืด": " รสชาติแย่ ",
    "ไม่สมราคา": " ราคาแพงเกินไป ",
    "หน้าเลือด": " ราคาแพงเกินไป ",
    "เหนียว": " เหนียวเคี้ยวยาก ",
    "คาว": " เหม็นคาว ",
    "บูด": " อาหารเสีย ",
    "ห่วย": " แย่มาก ",
}


def translate_emojis(text: str) -> str:
    for emoji_char, thai_text in THAI_EMOJI_MAP.items():
        text = text.replace(emoji_char, thai_text)
    return text


def translate_slang(text: str) -> str:
    for slang, standard in THAI_SLANG_MAP.items():
        text = text.replace(slang, standard)
    return text


def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = str(text).strip()
    return re.sub(r"\s+", " ", cleaned)


def extended_normalize_text(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = translate_emojis(cleaned)
    cleaned = translate_slang(cleaned)
    cleaned = _EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\d+", "0", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def xlmr_normalize_text(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"\d+", "0", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def xlmr_preprocess_minimal(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    return cleaned.lower()


def xlmr_preprocess_aggressive(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = translate_emojis(cleaned)
    cleaned = translate_slang(cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = _EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], cleaned)
    cleaned = re.sub(r"(ไม่ค่อย|ไม่น่า|ไม่เคย|ไม่)\s*([ก-ฮ]+)", r"\1_\2", cleaned)
    cleaned = re.sub(r"\d+", "0", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def xlmr_preprocess_keep_digits(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def xlmr_preprocess_emoji_tag(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = _EMOJI_PATTERN.sub(" [EMO] ", cleaned)
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"\d+", "0", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def xlmr_preprocess_segment(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"\d+", "0", cleaned)
    tokens = word_tokenize(cleaned, engine="newmm")
    cleaned = " ".join(tokens)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


PREPROCESS_REGISTRY: dict[str, callable] = {
    "default": xlmr_normalize_text,
    "minimal": xlmr_preprocess_minimal,
    "aggressive": xlmr_preprocess_aggressive,
    "keep_digits": xlmr_preprocess_keep_digits,
    "emoji_tag": xlmr_preprocess_emoji_tag,
    "segment": xlmr_preprocess_segment,
}
