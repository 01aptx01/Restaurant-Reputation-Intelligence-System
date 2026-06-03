# Getting Started

คู่มือรันทั้ง pipeline: ติดตั้ง → ข้อมูล → เทรน → ประเมิน → เว็บแอป

## ลำดับขั้นตอน

1. Setup (Python venv + `pip install -e .` + Bun)
2. Data (`scripts/download_wongnai.py` หรือ `generate_mock_data.py`)
3. Train (`python -m rris train <model>`)
4. Evaluate + visualize
5. Web cache (`scripts/initialize_web_data.py --model auto`)
6. Web server (`bun run web_app/index.ts`)

---

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Python 3.10+ แนะนำ

**GPU (NVIDIA RTX 50-series / sm_120):** การ์ดรุ่น Blackwell ต้องใช้ PyTorch ที่ build ด้วย CUDA 12.8 (`cu128`):

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

---

## 2. Data preparation

**Wongnai (แนะนำ):**

```powershell
python scripts/download_wongnai.py
```

**Smoke test:**

```powershell
python scripts/generate_mock_data.py
```

**เสริม (ถ้ามีข้อมูล merge เอง):**

```powershell
python scripts/merge.py
python scripts/reduce_merged.py
python scripts/augment_data.py
```

---

## 3. Train models

```powershell
python -m rris train baseline
python -m rris train xlmr
python -m rris train embedding
python -m rris train baseline_optuna   # optional: Optuna hyperparameter search
```

ทางเลือกหลัง `pip install -e .`: `rris-train baseline`

---

## 4. Evaluate and visualize

```powershell
python -m rris evaluate --model all --output outputs/eval/eval_report.json
python -m rris visualize --input outputs/eval/eval_report.json --output outputs/reports/eval_report_viz.html
```

รายงาน JSON จะมี `best_model` (เลือกจาก MAE ต่ำสุด) สำหรับ web dashboard

ประเมินทีละโมเดล: `--model baseline|xlmr|embedding|both`

---

## 5. Score / anomaly

```powershell
python -m rris score --model baseline --input data/wongnai/test.csv --output outputs/scores/scored_baseline.csv
```

---

## 6. Web app

```powershell
cd web_app; bun install; cd ..
python scripts/initialize_web_data.py --model auto
bun run web_app/index.ts
```

เปิด [http://127.0.0.1:8000](http://127.0.0.1:8000)

- `--model auto` อ่าน `best_model` จาก `outputs/eval/eval_report.json`
- หากยังไม่มี eval report จะ fallback เป็น `baseline` (ดู `WEB_MODEL_FALLBACK` ใน `config.py`)

---

## Research / tuning (optional)

```powershell
python scripts/eda_baseline_data.py
python scripts/tune_baseline.py --append-try-log
python scripts/summarize_tune_ceiling.py
python experiments/run_all_preprocess.py
```

---

## เอกสารเพิ่ม

| หัวข้อ | ไฟล์ |
|--------|------|
| ทุกโฟลเดอร์/ไฟล์ | [directory-guide.md](directory-guide.md) |
