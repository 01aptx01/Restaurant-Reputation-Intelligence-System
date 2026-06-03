# Web App

แผนที่ร้านอาหาร + แดชบอร์ด eval — รายละเอียดเต็มที่ **[docs/web-app.md](../docs/web-app.md)**

```powershell
cd web_app && bun install && bun run build && cd ..
python scripts/initialize_web_data.py --model hybrid_ensemble
bun run web_app/index.ts
```
