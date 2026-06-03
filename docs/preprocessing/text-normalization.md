# Text Normalization Functions

ฟังก์ชันล้างข้อความทั้งหมดอยู่ใน `src/rris/data/text.py`  
Registry สำหรับ XLM-R / Embedding: `PREPROCESS_REGISTRY`

---

## `normalize_text` (พื้นฐาน)

ใช้เป็นขั้นแรกของฟังก์ชันอื่นๆ เกือบทั้งหมด

1. คืน `""` ถ้า input เป็น `None` หรือ NaN
2. `strip()` ช่องว่างหัวท้าย
3. รวม whitespace หลายตัวเป็นช่องว่างเดียว

---

## `extended_normalize_text` (Baseline)

ใช้ตอน **โหลดข้อมูล Baseline** และเป็น input ของ `thai_tokenizer`

| ลำดับ | ขั้นตอน |
|-------|---------|
| 1 | `normalize_text` |
| 2 | แปล emoji → คำไทย (`THAI_EMOJI_MAP`) เช่น 😡→"โกรธ", 👍→"เยี่ยม" |
| 3 | แปลสแลง → คำมาตรฐาน (`THAI_SLANG_MAP`) เช่น "นัว"→"อร่อยกลมกล่อม", "ห่วย"→"แย่มาก" |
| 4 | ลบ emoji ที่เหลือ (regex) |
| 5 | ตัวเลขทุกกลุ่ม → `"0"` |
| 6 | lowercase |

---

## `xlmr_normalize_text` (default / XLM-R inference)

| ลำดับ | ขั้นตอน |
|-------|---------|
| 1 | `normalize_text` |
| 2 | ลดตัวอักษรซ้ำ 3+ ตัว → 2 ตัว (`"ดีีี"` → `"ดี"`) |
| 3 | ตัวเลข → `"0"` |
| 4 | lowercase |

**ไม่** แปล emoji/สแลง · **ไม่** ลบ URL · **ไม่** ผูก negation

---

## กลยุทธ์ใน `PREPROCESS_REGISTRY`

ตั้งค่าด้วย `XLMR_PREPROCESS_STRATEGY` ใน `config.py` (default: `"aggressive"`)

### `minimal`

- `normalize_text` + lowercase

### `aggressive` (default ตอน train XLM-R / Embedding)

| ลำดับ | ขั้นตอน |
|-------|---------|
| 1 | `normalize_text` |
| 2 | แปล emoji → คำไทย |
| 3 | แปลสแลง → คำมาตรฐาน |
| 4 | ลบ URL (`http://`, `www.`) |
| 5 | ลบ emoji ที่เหลือ |
| 6 | ลดตัวอักษรซ้ำ 3+ → 2 |
| 7 | ลด `!!` / `??` ซ้ำ → ตัวเดียว |
| 8 | ผูก negation: `(ไม่ค่อย\|ไม่น่า\|ไม่เคย\|ไม่) + คำไทย` → `ไม่_คำ` |
| 9 | ตัวเลข → `"0"` |
| 10 | lowercase |

### `keep_digits`

- เหมือน `xlmr_normalize_text` แต่ **เก็บตัวเลขจริง** (ไม่แทนด้วย 0)

### `emoji_tag`

- emoji → token `"[EMO]"` แทนการแปลเป็นคำไทย
- ลดตัวอักษรซ้ำ + ตัวเลข → 0 + lowercase

### `segment`

- ลดตัวอักษรซ้ำ + ตัวเลข → 0
- `word_tokenize(..., engine="newmm")` แล้ว join ด้วยช่องว่าง
- lowercase

---

## Emoji และสแลงที่รองรับ

### Emoji (`THAI_EMOJI_MAP`) — ตัวอย่าง

| Emoji | คำไทย |
|-------|-------|
| 😡 😠 | โกรธ |
| 👍 | เยี่ยม |
| 😋 🤤 | อร่อย / น่ากิน |
| ⭐ 🌟 | ดาว |

### สแลง (`THAI_SLANG_MAP`) — ตัวอย่าง

| สแลง | คำมาตรฐาน |
|------|-----------|
| นัว | อร่อยกลมกล่อม |
| ฟิน | มีความสุขมาก |
| ห่วย | แย่มาก |
| ไม่สมราคา / หน้าเลือด | ราคาแพงเกินไป |

รายการเต็มดูใน `src/rris/data/text.py`

---

## Negation phrases (Extra features)

ใช้ใน `count_negations()` สำหรับ Baseline extra features — **ไม่** ใช้ใน text normalize โดยตรง (ยกเว้น `aggressive` ที่ผูก `ไม่_คำ`)

```python
NEGATION_PHRASES = ("ไม่แนะนำ", "ไม่ค่อย", "ไม่")
```
