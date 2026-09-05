# Evaluation Metrics

คู่มือเมตริกที่ใช้ใน `python -m rris evaluate` และรายงาน JSON / dashboard

โค้ดหลัก: `src/rris/evaluation/metrics.py`, `src/rris/evaluation/runner.py`

---

## เมตริกมาตรฐาน

| เมตริก | นิยาม |
|--------|--------|
| **MAE** | `mean(|y_true - expected|)` — คะแนนดาวจริงเทียบ expected rating (ทศนิยม 1–5) |
| **RMSE** | รากที่สองของ MSE บน expected rating |
| **Accuracy** | สัดส่วนที่ `round(expected) == y_true` |
| **F1-macro / weighted** | จาก classification report บนดาวปัดเศษ |

Expected rating มาจาก softmax หรือ `predict_proba` ผ่าน `expected_rating_from_probs()` ยกเว้น regression head ที่ clip กลับช่วง 1–5

---

## เมตริกขยาย (extended)

| เมตริก | นิยาม |
|--------|--------|
| **off_by_one_accuracy** | สัดส่วนที่ `\|y_true - round(expected)\| <= 1` |
| **recall_star_1**, **recall_star_2** | recall ต่อคลาสจาก `classification_report` |
| **anomaly_rate** | สัดส่วนที่ `\|y_true - expected\| >= ANOMALY_THRESHOLD` (default 2.0) |
| **severe_error_rate** | สัดส่วนที่ `\|y_true - round(expected)\| >= 2` |

### Anomaly precision / recall / F1

ใช้ **severe error เป็น pseudo ground truth**:

- `y_severe = |y_true - round(expected)| >= 2`
- `y_flag = |y_true - expected| >= ANOMALY_THRESHOLD`

รายงาน P/R/F1 ของการ flag anomaly เทียบ severe errors — **ไม่ใช่ fraud ที่มนุษย์ label** ใช้สำหรับวิเคราะห์ calibration ของ threshold เท่านั้น

---

## การเลือกโมเดล production

| บริบท | เกณฑ์ |
|-------|-------|
| Train baseline / embedding / optuna | **Val MAE ต่ำสุด** |
| XLM-R checkpoint | **Val MAE ต่ำสุด** (early stopping) |
| Web dashboard (`--model auto`) | MAE จาก `eval_report.json` (`WEB_MODEL_METRIC = "mae"`) |

Helper: `src/rris/evaluation/selection.py` — `classifier_val_mae`, `pick_lowest_mae`

---

## ไฟล์ผลลัพธ์

| ไฟล์ | เนื้อหา |
|------|---------|
| `outputs/eval/eval_report.json` | เมตริกทุกโมเดล + `best_model` |
| `outputs/reports/eval_report_viz.html` | กราฟจาก `eval_report.py` |
| `outputs/eval/errors_*_severe_delta_ge_*.csv` | รีวิวที่ model พลาดหนัก |
| `outputs/eval/errors_*_confusion_summary.json` | คู่ `(true_star, pred_star)` ที่ \|delta\| ≥ 2 |

---

## คำสั่ง

```powershell
python -m rris evaluate --model all --output outputs/eval/eval_report.json
python -m rris evaluate --model baseline --export-errors
python -m rris visualize
```

ดู preprocessing ที่สอดคล้อง train/inference: [preprocessing/README.md](preprocessing/README.md)
