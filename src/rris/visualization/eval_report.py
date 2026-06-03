"""
Build an interactive HTML report comparing models from rris evaluate JSON output.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from rris import config

DEFAULT_INPUT = config.DEFAULT_EVAL_REPORT
DEFAULT_OUTPUT = config.DEFAULT_EVAL_VIZ

STAR_LABELS = ["1", "2", "3", "4", "5"]

MODEL_LABELS = {
    "baseline": "Baseline (TF-IDF + XGBoost)",
    "xlmr": "XLM-R",
    "embedding": "Embedding",
    "ensemble": "Hybrid Ensemble (XLM-R + BGE-M3)",
    "wangchan": "WangchanBERTa (Thai)",
    "hybrid_ensemble": "Hybrid Thai (Model 6)",
}

COLORS = {
    "baseline": "#5c6bc0",
    "xlmr": "#26a69a",
    "embedding": "#ab47bc",
    "ensemble": "#ff7043",
    "wangchan": "#00bcd4",
    "hybrid_ensemble": "#e91e63",
}

# Display order: higher-is-better vs lower-is-better
SUMMARY_METRICS = [
    ("mae", "MAE", "lower", "ดาว (ยิ่งต่ำยิ่งดี)"),
    ("rmse", "RMSE", "lower", "ดาว (ยิ่งต่ำยิ่งดี)"),
    ("off_by_one_accuracy", "Off-by-1 acc", "higher", "ทายผิดไม่เกิน 1 ดาว"),
    ("accuracy", "Accuracy", "higher", "สัดส่วนทายถูก (ปัด 1–5)"),
    ("f1_macro", "F1 (macro)", "higher", "F1 เฉลี่ยทุกดาว"),
    ("f1_weighted", "F1 (weighted)", "higher", "F1 ถ่วงตามจำนวนตัวอย่าง"),
    ("recall_star_1", "Recall ★1", "higher", "ดึงกลับรีวิว 1 ดาว"),
    ("recall_star_2", "Recall ★2", "higher", "ดึงกลับรีวิว 2 ดาว"),
    ("anomaly_f1", "Anomaly F1", "higher", "F1 การ flag delta≥threshold"),
    ("anomaly_rate", "Anomaly rate", "lower", "สัดส่วนรีวิวที่ flag"),
]


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _embedding_label_from_meta() -> str:
    """Build display label from embedding_meta.json (E5 vs fine-tuned BGE-M3)."""
    meta_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "embedding_meta.json")
    if not os.path.isfile(meta_path):
        return MODEL_LABELS["embedding"]
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return MODEL_LABELS["embedding"]
    if meta.get("finetune") and meta.get("finetuned_model_path"):
        base = meta.get("finetune_mode", "finetune")
        model = meta.get("embedding_model", "BGE-M3")
        short = str(model).split("/")[-1]
        return f"Embedding ({short}, {base})"
    model = meta.get("embedding_model", config.EMBEDDING_MODEL_NAME)
    short = str(model).split("/")[-1]
    return f"Embedding ({short})"


def get_model_label(model_key: str) -> str:
    if model_key == "embedding":
        return _embedding_label_from_meta()
    return MODEL_LABELS.get(model_key, model_key.upper().replace("_", "-"))


def get_model_color(model_key: str) -> str:
    return COLORS.get(model_key, "#78909c")


def _find_winner(model_values: dict[str, float], direction: str) -> str:
    if not model_values:
        return "tie"
    filtered_vals = {k: v for k, v in model_values.items() if v is not None}
    if not filtered_vals:
        return "tie"
    
    first_val = list(filtered_vals.values())[0]
    if all(v == first_val for v in filtered_vals.values()):
        return "tie"

    if direction == "lower":
        return min(filtered_vals, key=filtered_vals.get)
    else:
        return max(filtered_vals, key=filtered_vals.get)


def build_summary_rows(report: dict, available_models: list[str]) -> list[dict]:
    models = report["models"]
    rows = []
    for key, label, direction, hint in SUMMARY_METRICS:
        model_vals = {}
        for m in available_models:
            if key in models[m]:
                model_vals[m] = models[m][key]
        
        winner = _find_winner(model_vals, direction)
        rows.append(
            {
                "key": key,
                "metric": label,
                "hint": hint,
                "values": model_vals,
                "winner": winner,
                "direction": direction,
            }
        )
    return rows


def build_f1_rows(report: dict, available_models: list[str]) -> list[dict]:
    rows = []
    for star in STAR_LABELS:
        model_vals = {}
        for m in available_models:
            m_report = report["models"][m].get("classification_report", {})
            star_data = m_report.get(star, {})
            model_vals[m] = star_data.get("f1-score", 0.0)
        rows.append(
            {
                "star": star,
                "values": model_vals,
            }
        )
    return rows


def render_html(report: dict, source_path: str) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "plotly is required for visualization. Install with: pip install plotly"
        ) from exc

    models = report["models"]
    available_models = list(models.keys())
    if not available_models:
        raise ValueError("Report must contain at least one model under 'models'.")

    summary = build_summary_rows(report, available_models)
    f1_rows = build_f1_rows(report, available_models)
    n_samples = models[available_models[0]].get("n_samples", "—")
    test_input = report.get("input", "—")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Chart 1: summary grouped bars ---
    metric_names = [r["metric"] for r in summary]
    fig_summary = go.Figure()
    for m in available_models:
        fig_summary.add_trace(
            go.Bar(
                name=get_model_label(m),
                x=metric_names,
                y=[r["values"].get(m, 0.0) for r in summary],
                marker_color=get_model_color(m),
                text=[f"{r['values'].get(m, 0.0):.3f}" for r in summary],
                textposition="outside",
            )
        )
    fig_summary.update_layout(
        title="เปรียบเทียบ Metrics หลักของทุกโมเดล",
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
        yaxis_title="ค่า metric",
    )

    # --- Chart 2: F1 per star ---
    fig_f1 = go.Figure()
    for m in available_models:
        fig_f1.add_trace(
            go.Bar(
                name=get_model_label(m),
                x=[f"ดาว {r['star']}" for r in f1_rows],
                y=[r["values"].get(m, 0.0) for r in f1_rows],
                marker_color=get_model_color(m),
            )
        )
    fig_f1.update_layout(
        title="F1-score แยกตามดาว (1–5)",
        barmode="group",
        template="plotly_white",
        height=380,
        yaxis=dict(range=[0, 1], title="F1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
    )

    # --- Heatmaps: confusion matrices ---
    def heatmap(cm: list, title: str) -> go.Figure:
        z = cm
        text = [[str(v) for v in row] for row in z]
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=[f"ทาย {s}" for s in STAR_LABELS],
                y=[f"จริง {s}" for s in STAR_LABELS],
                text=text,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True,
            )
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=360,
            xaxis_title="ทาย (ปัดดาว)",
            yaxis_title="ดาวจริง",
            margin=dict(t=60, b=50),
        )
        return fig

    cm_divs = []
    for m in available_models:
        m_data = models[m]
        if "confusion_matrix" in m_data:
            fig_cm = heatmap(m_data["confusion_matrix"], get_model_label(m))
            cm_div_str = fig_cm.to_html(full_html=False, include_plotlyjs=False)
            cm_divs.append(
                f"""<div class="cm-card">
                  {cm_div_str}
                </div>"""
            )

    summary_div = fig_summary.to_html(full_html=False, include_plotlyjs=False)
    f1_div = fig_f1.to_html(full_html=False, include_plotlyjs=False)

    # Dynamic CSS Styling lines
    style_lines = []
    for m in available_models:
        color = get_model_color(m)
        style_lines.append(f"table.compare th.header-{m} {{ border-top: 3px solid {color}; background: {color}12; }}")
        style_lines.append(f"table.compare td.col-model.winner.{m} {{ background: {color}15; font-weight: 700; }}")
        style_lines.append(f".card.{m} {{ border-left-color: {color}; }}")
        style_lines.append(f".badge.{m} {{ background: {color}30; color: {color}; }}")

    # Comparison table headers
    th_cols = []
    col_width = int(54 / len(available_models))
    for m in available_models:
        th_cols.append(f'<th class="col-model header-{m}" style="width: {col_width}%;">{get_model_label(m)}</th>')
        
    table_headers = f"""
      <tr>
        <th class="col-metric" style="width: 32%;">Metric</th>
        {''.join(th_cols)}
        <th class="col-winner" style="width: 14%;">ดีกว่า</th>
      </tr>
    """

    # Comparison table rows
    table_rows = []
    for r in summary:
        w = r["winner"]
        row_cells = []
        row_cells.append(f'<td class="col-metric"><strong>{r["metric"]}</strong><br><span class="hint">{r["hint"]}</span></td>')
        for m in available_models:
            val = r["values"].get(m, 0.0)
            cls = f"winner {m}" if m == w else f"{m}"
            row_cells.append(f'<td class="col-model num {cls}">{val:.4f}</td>')
        winner_label = get_model_label(w) if w != "tie" else "เสมอ"
        badge_cls = w if w != "tie" else "tie"
        row_cells.append(f'<td class="col-winner"><span class="badge {badge_cls}">{winner_label}</span></td>')
        table_rows.append(f"<tr>{''.join(row_cells)}</tr>")

    # KPI cards
    kpi_cards = []
    for m in available_models:
        m_data = models[m]
        acc = m_data.get("accuracy", 0.0) * 100
        mae = m_data.get("mae", 0.0)
        rmse = m_data.get("rmse", 0.0)
        lbl = get_model_label(m)
        kpi_cards.append(
            f"""<div class="card {m}">
              <h3>{lbl}</h3>
              <div class="value">{acc:.1f}% Accuracy</div>
              <div class="sub">MAE: {mae:.3f} · RMSE: {rmse:.3f}</div>
            </div>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>รายงานผลลัพธ์การประเมินประสิทธิภาพตัวแบบจำแนกคะแนนดาว</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #f4f6f9;
      --card: #ffffff;
      --text: #1a237e;
      --muted: #5c6b7a;
      --accent: #26a69a;
      --winner-bg: #e8f5e9;
      --shadow: 0 4px 24px rgba(26, 35, 126, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: linear-gradient(160deg, #e8eaf6 0%, var(--bg) 40%, #e0f2f1 100%);
      color: var(--text);
      margin: 0;
      padding: 2rem 1.5rem 3rem;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    header {{
      text-align: center;
      margin-bottom: 2rem;
    }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0 0 0.5rem;
      letter-spacing: -0.02em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--card);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: var(--shadow);
      border-left: 4px solid var(--accent);
    }}
    .card h3 {{ margin: 0 0 0.5rem; font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .card .value {{ font-size: 1.4rem; font-weight: 700; }}
    .card .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }}
    section {{
      background: var(--card);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--shadow);
    }}
    section h2 {{
      font-size: 1.15rem;
      margin: 0 0 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid #e8eaf6;
    }}
    table.compare {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.95rem;
      border: 1px solid #cfd8dc;
      border-radius: 10px;
      overflow: hidden;
    }}
    table.compare th,
    table.compare td {{
      padding: 0.85rem 1.15rem;
      border-bottom: 1px solid #dce1e6;
      border-right: 1px solid #dce1e6;
      vertical-align: middle;
    }}
    table.compare th:last-child,
    table.compare td:last-child {{
      border-right: none;
    }}
    table.compare tbody tr:last-child td {{
      border-bottom: none;
    }}
    table.compare th {{
      color: var(--text);
      font-weight: 600;
      font-size: 0.9rem;
      letter-spacing: 0.02em;
    }}
    table.compare th.col-metric {{
      text-align: left;
      background: #eceff1;
    }}
    table.compare th.col-model {{
      text-align: right;
    }}
    table.compare th.col-winner {{
      text-align: center;
      background: #f5f5f5;
      border-top: 3px solid #90a4ae;
    }}
    table.compare td.col-metric {{
      background: #fafafa;
      text-align: left;
    }}
    table.compare td.col-model {{
      background: #fafbff;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    table.compare td.col-winner {{
      background: #ffffff;
      text-align: center;
    }}
    table.compare tbody tr:hover td.col-metric {{ background: #f0f0f0; }}
    table.compare tbody tr:hover td.col-model {{ background: #f5f6fc; }}
    table.compare tbody tr:hover td.col-winner {{ background: #fafafa; }}
    .badge {{
      display: inline-block;
      min-width: 5.5rem;
      padding: 0.25em 0.75em;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      text-align: center;
    }}
    .badge.tie {{ background: #eceff1; color: #546e7a; }}
    .hint {{ font-size: 0.75rem; color: var(--muted); font-weight: normal; }}
    
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }}
    @media (min-width: 900px) {{
      .charts-grid.two-col {{ grid-template-columns: 1fr 1fr; }}
      .charts-grid.multi-col {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
    }}
    .cm-card {{
      background: var(--card);
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 4px 16px rgba(0,0,0,0.03);
      border: 1px solid #e8eaf6;
    }}
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.8rem;
      margin-top: 2rem;
    }}
    
    /* Dynamic CSS generated based on models */
    {chr(10).join(style_lines)}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>รายงานผลลัพธ์การประเมินและเปรียบเทียบโมเดลทั้งหมด</h1>
      <p class="meta">
        ชุดทดสอบ: <code>{test_input}</code> · n = {n_samples} ·
        สร้างจาก <code>{os.path.basename(source_path)}</code> · {generated}
      </p>
    </header>

    <div class="cards">
      {chr(10).join(kpi_cards)}
    </div>

    <section>
      <h2>ตารางเปรียบเทียบประสิทธิภาพแบบแยกละเอียด</h2>
      <table class="compare">
        <thead>
          {table_headers}
        </thead>
        <tbody>
          {chr(10).join(table_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>กราฟเปรียบเทียบค่าความผิดพลาดและสัดส่วนความถูกต้อง (Summary Metrics)</h2>
      {summary_div}
    </section>

    <section>
      <h2>คะแนน F1-score แยกวิเคราะห์ตามรายระดับดวงดาว (1–5)</h2>
      {f1_div}
    </section>

    <section>
      <h2>วิเคราะห์การทายผิดกลุ่ม — Confusion Matrices ของแต่ละตัวแบบ</h2>
      <div class="charts-grid multi-col">
        {chr(10).join(cm_divs)}
      </div>
    </section>

    <footer>
      สร้างโดย python -m rris visualize · เปิดดูเอกสารนี้ด้วยเว็บบราวเซอร์ของคุณเพื่อความเสถียรสูงสุด
    </footer>
  </div>
</body>
</html>
"""
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize eval_report.json as an interactive HTML dashboard.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"JSON from rris evaluate (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}\nRun: python -m rris evaluate")

    report = load_report(args.input)
    html = render_html(report, args.input)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved visualization: {args.output}")
    print("Open this file in your browser to view charts and comparison table.")


if __name__ == "__main__":
    main()
