# Restaurant Reputation Intelligence System (RRIS)

ระบบวิเคราะห์ความน่าเชื่อถือรีวิวร้านอาหารจากข้อความภาษาไทย — scope ย่อ: **3 โมเดล inference** (baseline, embedding, xlmr) + **baseline_optuna** สำหรับจูน hyperparameter

## เริ่มต้น

อ่าน [docs/getting-started.md](docs/getting-started.md) สำหรับขั้นตอนติดตั้งและรัน pipeline ทั้งหมด

## โมเดลที่รองรับ

| Train | Inference / Score / Eval |
|-------|--------------------------|
| `baseline` | ✓ |
| `baseline_optuna` | (artifact → baseline) |
| `embedding` | ✓ |
| `xlmr` | ✓ |

## คำสั่งหลัก

```powershell
python -m rris train baseline
python -m rris train embedding
python -m rris train xlmr
python -m rris evaluate --model all --output outputs/eval/eval_report.json
python scripts/initialize_web_data.py --model auto
bun run web_app/index.ts
```

Web UI เลือกโมเดลที่ **evaluate ได้ MAE ต่ำสุด** อัตโนมัติ (`--model auto`)

## โครงสร้างโปรเจกต์

ดู [docs/directory-guide.md](docs/directory-guide.md)

## Preprocessing

ดู [docs/preprocessing/README.md](docs/preprocessing/README.md) — อธิบายขั้นตอนล้างข้อมูลและแปลงฟีเจอร์ของแต่ละโมเดล

## Training

ดู [docs/training/README.md](docs/training/README.md) — อธิบายเทคนิคการเทรน, imbalance handling และ hyperparameters ของแต่ละโมเดล

## Smoke test

```powershell
$env:RRIS_SMOKE="1"
python scripts/generate_mock_data.py
python -m rris train baseline
python -m rris evaluate --model baseline --output outputs/eval/eval_report.json
```
