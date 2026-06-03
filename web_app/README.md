# Web App

แผนที่ร้านอาหาร + แดชบอร์ด eval — รายละเอียดเต็มที่ **[docs/web-app.md](../docs/web-app.md)**

```powershell
cd web_app && bun install && bun run build && cd ..
python scripts/initialize_web_data.py --model auto
bun run web_app/index.ts
```

`--model auto` เลือกโมเดลที่ evaluate ได้ **MAE ต่ำสุด** จาก `outputs/eval/eval_report.json`  
รองรับ: `baseline`, `xlmr`, `embedding`, `auto`
