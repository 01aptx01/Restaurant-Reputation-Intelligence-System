# -*- coding: utf-8 -*-
# Model evaluation pipeline (python -m rris evaluate)

import argparse # ระบบช่วยวิเคราะห์พารามิเตอร์ CLI ตอนเรียกใช้งานผ่านหน้าจอ Terminal
import json     # เครื่องมือจัดทำข้อมูลสรุปและอ่านเขียนเอกสารในแบบมาตรฐาน JSON
import os       # เครื่องมือสำหรับติดต่อระบบไดเรกทอรีและตรวจสภาพแวดล้อมระบบไฟล์
import sys      # ตัวเข้าถึงตัวแปร Interpreter ของระบบ Python สำหรับควบคุมการออกจากสคริปต์

import numpy as np # ไลบรารีการประมวลผลเชิงตัวเลขและจัดสรรข้อมูล Array
import pandas as pd # ไลบรารีการจัดการจัดเรียงและวิเคราะห์ตารางข้อมูลแบบ 2 มิติ

from rris.inference.prep import prepare_scoring_for_model
from rris.evaluation.metrics import compute_extended_metrics
# นำเข้าตัววัดประสิทธิภาพระดับวิชาการของ Scikit-Learn ครบถ้วนตามมาตรฐานสากล
from sklearn.metrics import (
    accuracy_score,       # สัดส่วนความถูกต้องของการจำแนกประเภท (Accuracy)
    classification_report,# รายงานสรุปแจกแจงค่า Precision, Recall, F1 แยกแต่ละประเภทดาว
    confusion_matrix,     # แผนผังแมทริกซ์การจำแนกคำตอบถูก-ผิด (Confusion Matrix)
    f1_score,             # คะแนนชี้วัดความลงตัวระหว่างความถูกต้องและความครอบคลุม (F1-Score)
    mean_absolute_error,  # ส่วนเบี่ยงเบนสัมบูรณ์เฉลี่ยชี้วัดผลต่างระหว่างดาว (MAE)
    mean_squared_error,   # ส่วนเบี่ยงเบนกำลังสองเฉลี่ยเพื่อคำนวณหาระยะความผิดพลาด (MSE/RMSE)
)

from rris import config, utils
from rris.inference.baseline import predict_baseline, predict_baseline_with_probs
from rris.inference.embedding import predict_embedding_with_probs
from rris.inference.xlmr import predict_xlmr

# ค่ามาตรฐานชี้พิกัดไฟล์ประเมินหลักไปที่ Wongnai Test CSV
DEFAULT_EVAL_INPUT = config.TEST_PATH
# ลิสต์คะแนนดาวสำหรับการจัดระดับดัชนีคลาส
LABELS = [1, 2, 3, 4, 5]


def _baseline_uses_lsa() -> bool:
    """ตรวจสอบจากประวัติ Metadata บันทึกการฝึกสอน ว่ารอบนั้นได้ทำการเปิดใช้ LSA หรือไม่"""
    if os.path.isfile(config.BASELINE_META_PATH):
        with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
            meta = json.load(f) # อ่านไฟล์ JSON บันทึกระบบ
        return bool(meta.get("use_lsa", config.BASELINE_USE_LSA)) # คืนค่าผลการตั้งค่า
    return config.BASELINE_USE_LSA # หากไม่มีประวัติจะดึงค่ามาตรฐานจากคอนฟิกหลักแทน


def _baseline_meta() -> dict:
    """โหลดข้อมูลประวัติตั้งค่า Metadata ของโมเดล Baseline จากไฟล์ JSON"""
    if os.path.isfile(config.BASELINE_META_PATH):
        with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
            return json.load(f) # คืนดิกบันทึกประวัติการรัน
    return {} # คืนค่าดิกชันนารีเปล่าหากไม่พบประวัติ


def _baseline_artifact_paths() -> list[str]:
    """คำนวณตรวจสอบรายชื่อพาธไฟล์เวตและตัวสกัดฟีเจอร์ของ Baseline ที่ต้องใช้สำหรับการรันทำนายจริง"""
    meta = _baseline_meta() # ดึง Metadata
    best_model_type = meta.get("best_model_type", "xgboost")
    model_path = getattr(config, "SKLEARN_MODEL_PATH", os.path.join(config.BASELINE_ARTIFACTS_DIR, "sklearn_model.joblib")) if best_model_type != "xgboost" else config.XGB_MODEL_PATH
    
    paths = [config.TFIDF_VECTORIZER_PATH, model_path] # พาธ Word TF-IDF และโมเดลที่ชนะ

    # แนบตรวจเช็คพาสไฟล์ SVD (LSA) หากมีการเปิดใช้
    if meta.get("use_lsa", _baseline_uses_lsa()):
        paths.append(config.LSA_TRANSFORMER_PATH)
    # แนบตรวจเช็คพาสไฟล์ Char TF-IDF หากเปิดทำงาน
    if meta.get("use_char_tfidf", config.BASELINE_USE_CHAR_TFIDF):
        paths.append(config.CHAR_TFIDF_VECTORIZER_PATH)
    return paths # ส่งคืนลิสต์รวมพาธไฟล์จำเป็นทั้งหมด


def _xlmr_artifact_dir() -> str:
    """ส่งคืนไดเรกทอรีจัดเก็บเวตไฟล์และโทเคนไนเซอร์ของ Advanced XLM-R"""
    return config.XLMR_ARTIFACTS_DIR


def _baseline_ready() -> bool:
    return all(os.path.isfile(p) for p in _baseline_artifact_paths())


def _xlmr_ready() -> bool:
    model_dir = _xlmr_artifact_dir()
    return os.path.isdir(model_dir) and os.path.isfile(
        os.path.join(model_dir, "config.json")
    )


def _embedding_ready() -> bool:
    meta_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "embedding_meta.json")
    model_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "clf_model.joblib")
    return os.path.isfile(meta_path) and os.path.isfile(model_path)


_READINESS = {
    "baseline": _baseline_ready,
    "xlmr": _xlmr_ready,
    "embedding": _embedding_ready,
}


def _fail_missing(name: str) -> None:
    hints = {
        "baseline": "python -m rris train baseline",
        "xlmr": "python -m rris train xlmr",
        "embedding": "python -m rris train embedding",
    }
    print(f"{name} artifacts not found. Run: {hints.get(name, '')}", file=sys.stderr)
    sys.exit(1)


def resolve_eval_models(requested: str) -> set[str]:
    """Return model keys to evaluate; for 'all', skip models without artifacts."""
    if requested in _READINESS:
        if not _READINESS[requested]():
            _fail_missing(requested)
        return {requested}

    if requested == "both":
        want = ("baseline", "xlmr")
    else:
        want = tuple(_READINESS.keys())

    ready = {name for name in want if _READINESS[name]()}
    missing = set(want) - ready
    for name in sorted(missing):
        print(f"Skipping eval for {name}: artifacts not found", file=sys.stderr)
    if not ready:
        print("No models available to evaluate.", file=sys.stderr)
        sys.exit(1)
    return ready


def ensure_artifacts(model: str) -> None:
    """Strict check for a single-model evaluate request."""
    resolve_eval_models(model)


def rounded_stars(expected: np.ndarray) -> np.ndarray:
    """ปัดเศษคะแนนทำนายดาวเฉลี่ยของ AI (ทศนิยม) ให้เป็นจำนวนเต็ม (1 ถึง 5 ดาว) เพื่อคำนวณ Metrics เชิงจัดกลุ่ม"""
    return np.clip(np.round(expected), 1, 5).astype(int) # ปัดเศษดาวแล้วกักขอบเขตไม่ให้เกิดค่านอกคลาส 1-5


def majority_baseline_metrics(
    y_true: np.ndarray,
    majority_class: int = config.BASELINE_MAJORITY_CLASS,
) -> dict:
    """คำนวณ Metric มาตรฐานอ้างอิงฐานหลัก (Majority Baseline) โดยตั้งค่าสมมติว่าโมเดลทำนายเดาคะแนนฐานเดียวกันทั้งหมด (เช่น 4★)"""
    expected = np.full(len(y_true), float(majority_class), dtype=np.float64) # เวกเตอร์คะแนนทศนิยม (4.0 ดาวทั้งหมด)
    y_pred = np.full(len(y_true), majority_class, dtype=int)                 # เวกเตอร์คะแนนจัดกลุ่ม (4 ดาวจำนวนเต็ม)
    
    # รันรายงานสถิติจำแนกประเภทเพื่อประเมินความแม่นยำรายดวงดาว
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0, # เติมค่า 0 ทันทีหากพบคลาสดาวใดที่มีตัวหารร่วมเป็น 0 ป้องกันโปรแกรมค้าง
    )
    # รวบรวมตัวชี้วัดความแม่นยำมาตรฐานคืนกลับไปในแบบ JSON-Friendly
    return {
        "n_samples": int(len(y_true)),                              # จำนวนประชากรข้อมูลทดสอบ
        "predicted_class": majority_class,                          # ดัชนีดาวมาตรฐานที่ใช้เป็นฐานเปรียบเทียบ
        "mae": float(mean_absolute_error(y_true, expected)),        # ความเบี่ยงเบนสัมบูรณ์เฉลี่ย
        "rmse": float(np.sqrt(mean_squared_error(y_true, expected))),# รากที่สองความคลาดเคลื่อนกำลังสองเฉลี่ย
        "accuracy": float(accuracy_score(y_true, y_pred)),          # อัตราความถูกต้อง
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), # F1 เฉลี่ยธรรมดาทุกดาว
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0) # F1 ถ่วงน้ำหนักปริมาณประชากรดาว
        ),
        "per_class_recall": utils.per_class_recall(report),          # สถิติ Recall แยกดาวแต่ละกลุ่ม
        "classification_report": report,                            # รายงานดิกรวมสรุป Scikit-Learn
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(), # แผนผังจำแนกในรูป Python List
    }


def compute_metrics(y_true: np.ndarray, expected: np.ndarray) -> dict:
    """วิเคราะห์คำนวณเปรียบเทียบระหว่างคะแนนจริงและคะแนนที่ AI คาดการณ์"""
    return compute_extended_metrics(y_true, expected)


def print_metrics(
    name: str,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """พิมพ์สรุปค่าคะแนนสถิติชี้วัดทั้งหมดและ Confusion Matrix ออกจอแสดงผลอย่างงดงามและมองหาง่าย"""
    print(f"\n=== {name} ===")
    print(f"n_samples:   {metrics['n_samples']}")       # ยอดประชากร
    print(f"MAE:         {metrics['mae']:.4f}")         # ค่าผิดเพี้ยนทศนิยมเฉลี่ย
    print(f"RMSE:        {metrics['rmse']:.4f}")        # ค่าผิดเพี้ยนน้ำหนักยกกำลังสอง (ขยายเคสทายพลาดห่างไกล)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")    # ยอดจำแนกดาวถูกต้องทั้งหมด
    print(f"F1 macro:    {metrics['f1_macro']:.4f}")    # F1-Score เฉลี่ยความเท่าเทียมคลาสดาว
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    if "off_by_one_accuracy" in metrics:
        print(f"Off-by-1:    {metrics['off_by_one_accuracy']:.4f}")
    if "anomaly_rate" in metrics:
        print(f"Anomaly rate:{metrics['anomaly_rate']:.4f}")
        print(
            f"Anomaly P/R/F1: {metrics.get('anomaly_precision', 0):.4f} / "
            f"{metrics.get('anomaly_recall', 0):.4f} / "
            f"{metrics.get('anomaly_f1', 0):.4f}"
        )
    
    # วนลูปพิมพ์ค่าการดึงกลับ (Recall) ของดาวดวงระดับ 1 ถึง 5 เพื่อให้ตรวจจับจุดอ่อนของโมเดล
    if metrics.get("per_class_recall"):
        print("Per-class recall:")
        for star, rec in sorted(metrics["per_class_recall"].items()):
            print(f"  star {star}: {rec:.4f}")
            
    print("\nClassification report:")
    # แสดงรายงานจำแนกประเภทรายดาวของ Scikit-Learn ในรูปตารางพิมพ์มาตรฐาน
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    # พิมพ์แผนผังแนวตั้งแถวนอน (แถว = ดาวจริง, คอลัมน์ = ดาวที่ AI เดาตอบ) สำหรับดูเปรียบเทียบการกระจายความผิดเพี้ยน
    for row in metrics["confusion_matrix"]:
        print(" ", row)


def print_comparison(models_metrics: dict) -> None:
    """เปรียบเทียบสถิติของทุกโมเดลและตัดสินหาโมเดลที่ทำผลงานได้ดีที่สุดในแต่ละหัวข้อสถิติ"""
    print("\n=== Comparison (All Models) ===")
    models = list(models_metrics.keys())
    
    # สร้างส่วนหัวตาราง
    header = f"{'metric':<12} " + " ".join([f"{m:>15}" for m in models]) + "  better"
    print(header)
    print("-" * len(header))
    
    # วนลูปสลักพิมพ์ Metrics หลักทั้ง 4
    for key in ("mae", "rmse", "accuracy", "f1_macro"):
        row = f"{key:<12} "
        best_val = None
        best_model = None
        
        for m in models:
            val = models_metrics[m][key]
            row += f"{val:>15.4f} "
            
            # หาโมเดลที่ทำคะแนนได้ดีที่สุด
            if best_val is None:
                best_val = val
                best_model = m
            else:
                if key in ("mae", "rmse"):
                    # ตัวประเมินความคลาดเคลื่อน (MAE/RMSE): ยิ่งแต้มความสูญเสียต่ำยิ่งเป็นโมเดลที่เลิศ
                    if val < best_val:
                        best_val = val
                        best_model = m
                else:
                    # ตัวชี้วัดอัตราแม่นยำ (Accuracy/F1-Score): คะแนนยิ่งสูงยิ่งเก่ง
                    if val > best_val:
                        best_val = val
                        best_model = m
                        
        row += f"  {best_model}"
        print(row)


_LOWER_IS_BETTER = frozenset(("mae", "rmse"))


def pick_best_model(
    models_metrics: dict,
    metric: str | None = None,
) -> tuple[str, float]:
    """Return (model_key, metric_value) for the best model on the given metric."""
    metric = metric or config.WEB_MODEL_METRIC
    if metric not in ("mae", "rmse", "accuracy", "f1_macro"):
        raise ValueError(f"Unsupported metric: {metric}")

    best_model = None
    best_val = None
    for name, stats in models_metrics.items():
        val = stats[metric]
        if best_val is None:
            best_val = val
            best_model = name
        elif metric in _LOWER_IS_BETTER:
            if val < best_val:
                best_val = val
                best_model = name
        elif val > best_val:
            best_val = val
            best_model = name

    if best_model is None:
        raise ValueError("No models to compare")
    return best_model, float(best_val)


def resolve_best_model(report_path: str | None = None) -> str:
    """Read eval report JSON and return best_model, or WEB_MODEL_FALLBACK."""
    path = report_path or config.DEFAULT_EVAL_REPORT
    if not os.path.isfile(path):
        return config.WEB_MODEL_FALLBACK

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    if report.get("best_model") in _READINESS:
        return report["best_model"]

    models = report.get("models") or {}
    if not models:
        return config.WEB_MODEL_FALLBACK

    best_model, _ = pick_best_model(models)
    return best_model


def evaluate_model(name: str, df: pd.DataFrame, expected: np.ndarray) -> dict:
    """ประมวลผลขั้นตอนการวิเคราะห์ค่าความถูกต้องคำนวณและสกรีนพิมพ์สถิติออกจอสรุปครบวงจร"""
    y_true = df["user_rating"].values.astype(int) # สกัดเอาเรตติ้งดาวจริงมาเก็บ
    metrics = compute_metrics(y_true, expected)   # วิเคราะห์สถิติ Metric ต่างๆ
    y_pred = rounded_stars(expected)             # ปัดเศษคะแนนดาวให้เป็นจำนวนเต็ม
    print_metrics(name, metrics, y_true, y_pred)  # พิมพ์รายงานสรุปออกหน้าจอ
    return metrics # ส่งคืนค่าคำนวณสถิติเพื่อลงบันทึก


def save_predictions(
    df: pd.DataFrame,
    expected: np.ndarray,
    model_name: str,
) -> str:
    """จัดเตรียมสร้างบันทึกเขียนผลลัพธ์คำทำนายทุกข้อความรีวิวลงในรูปแบบตาราง CSV ลงไดเรกทอรี 'outputs/eval/'"""
    os.makedirs(config.EVAL_DIR, exist_ok=True) # ยืนยันไดเรกทอรีปลายทาง
    out_path = os.path.join(config.EVAL_DIR, f"eval_{model_name}_preds.csv") # กำหนดสกีมาชื่อพาธ
    
    # แนบข้อความ ดาวดิบจริง ดาวที่ AI คาดการณ์ทศนิยม และคำตอบปัดเศษจำนวนเต็ม
    out_df = pd.DataFrame(
        {
            "text": df["text"],
            "user_rating": df["user_rating"],
            "ai_expected_rating": expected,
            "pred_star_rounded": rounded_stars(expected),
        }
    )
    out_df.to_csv(out_path, index=False) # เซฟผลเป็นตารางผลวิเคราะห์
    print(f"Saved predictions: {out_path}")
    return out_path # ส่งคืนพาสเป้าหมาย


def parse_args() -> argparse.Namespace:
    """กำหนดวิเคราะห์ตัวแปรพารามิเตอร์ของ CLI สำหรับจัดสรรระบบประเมินผล"""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and/or XLM-R on a labeled test CSV.",
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "xlmr", "embedding", "both", "all"),
        default="both",
        help="Model to evaluate: baseline, xlmr, embedding, both, or all",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_EVAL_INPUT,
        help=f"ไฟล์ข้อความทดสอบหลัก CSV ที่บันทึกดาวเป้าหมายไว้ถูกต้อง (มาตรฐานชี้ไปที่: {DEFAULT_EVAL_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="พิกัดไฟล์รายงาน JSON รายงานสรุปคะแนนประเมิน (เช่น outputs/eval_report.json)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="บันทึกรายละเอียดคำทำนายข้อความของแต่ละโมเดลลงตาราง CSV เสริม",
    )
    parser.add_argument(
        "--export-errors",
        action="store_true",
        help="สั่งรันขั้นตอนส่งออกวิเคราะห์หาข้อผิดพลาดรุนแรงและเคสความไม่มั่นใจของโมเดล Baseline ลงไดเรกทอรีย่อย",
    )
    parser.add_argument(
        "--no-majority",
        action="store_true",
        help="ข้ามการวิเคราะห์ส่วนเปรียบเทียบฐานอ้างอิง Majority (4★) ในรายงานสรุป",
    )
    return parser.parse_args()


def main() -> None:
    """ท่อโครงสร้างประสานการทดสอบประเมินโมเดลคู่อย่างมีหลักเกณฑ์ (Prevent Cross-Contamination & Leakage)"""
    args = parse_args() # ประมวลผลฟิลเตอร์พารามิเตอร์ CLI

    # ตรวจสอบยืนยันความมีอยู่จริงของไฟล์ข้อมูลเป้าหมาย หากขาดจะแนะนําให้นำตัว Holdout มาสวมรันทันที
    if not os.path.isfile(args.input):
        print(
            f"Input file not found: {args.input}\n"
            "Use a held-out test set (e.g. data/wongnai/holdout.csv) that was not "
            "used for training.",
            file=sys.stderr,
        )
        sys.exit(1) # สั่งหยุดระบบ

    active = resolve_eval_models(args.model)

    print(f"--- Evaluation (model={args.model}, active={sorted(active)}) ---")
    print(f"Input: {args.input}")

    # โครงสร้าง JSON หลักที่จะใช้ลงประวัติผลลัพธ์
    results: dict = {"input": args.input, "models": {}, "skipped_models": []}
    if args.model == "all":
        results["skipped_models"] = sorted(set(_READINESS) - active)

    # ==============================================================================
    # 1. การตรวจสอบสำหรับโมเดลหลัก Baseline (TF-IDF + XGBoost)
    # ==============================================================================
    if "baseline" in active:
        df_baseline = prepare_scoring_for_model(args.input, "baseline")
        # ตรวจความสมดุลของการแจกแจงดาวในชุดข้อความประเมินเทียบกับดาวจริง
        utils.compare_rating_distributions(
            df_baseline["user_rating"].values,
            label_train="eval input (baseline clean)",
        )
        y_true = df_baseline["user_rating"].values.astype(int) # ดึงดาวจริง
        
        # 1.1 ประเมิน Majority Baseline สำหรับอ้างอิงระดับต่ำสุด
        if not args.no_majority and "majority_baseline" not in results:
            maj = majority_baseline_metrics(y_true)
            print_metrics(
                f"Majority baseline (predict {maj['predicted_class']} stars)",
                maj,
                y_true,
                np.full(len(y_true), maj["predicted_class"], dtype=int),
            )
            results["majority_baseline"] = maj # เก็บค่าสถิติอ้างอิง

        # 1.2 รันการทำนายผลลัพธ์ผ่านสคริปต์ทำนายของ XGBoost
        expected, probs = predict_baseline_with_probs(df_baseline)
        metrics = evaluate_model("Baseline (TF-IDF + Classification Model)", df_baseline, expected)
        results["models"]["baseline"] = metrics # เก็บค่าสถิติ

        if args.save_predictions:
            save_predictions(df_baseline, expected, "baseline") # ส่งออกตารางคำทำนาย
            
        if args.export_errors:
            prefix = os.path.splitext(os.path.basename(args.input))[0]
            # ส่งออกเคสสาหัสและโอกาสมั่นใจต่ำลงโฟลเดอร์สำหรับทำวิจัยเพิ่มเติม
            paths = utils.export_error_analysis(
                df_baseline,
                expected,
                probs,
                config.EVAL_DIR,
                min_delta=config.ERROR_EXPORT_MIN_DELTA,
                low_conf_threshold=config.LOW_CONFIDENCE_THRESHOLD,
                prefix=f"errors_{prefix}",
            )
            results["error_exports"] = paths # บันทึกเส้นทางไฟล์ผิดพลาด
            print(f"Error analysis exports: {paths}")

    # ==============================================================================
    # 2. การตรวจสอบสำหรับโมเดลระดับลึก Advanced (XLM-RoBERTa Base)
    # ==============================================================================
    if "xlmr" in active:
        df_xlmr = prepare_scoring_for_model(args.input, "xlmr")
        # ตรวจความสมดุลของการแจกแจงดาวในแบบของโมเดลระดับลึก
        utils.compare_rating_distributions(
            df_xlmr["user_rating"].values,
            label_train="eval input (xlmr clean)",
        )
        y_true_xlmr = df_xlmr["user_rating"].values.astype(int)
        
        # 2.1 ทำการประเมิน Majority Baseline สำหรับ XLM-R (หาก Baseline ไม่ได้วิ่งไว้ก่อนหน้า)
        if not args.no_majority and "majority_baseline" not in results:
            maj = majority_baseline_metrics(y_true_xlmr)
            print_metrics(
                f"Majority baseline (predict {maj['predicted_class']} stars)",
                maj,
                y_true_xlmr,
                np.full(len(y_true_xlmr), maj["predicted_class"], dtype=int),
            )
            results["majority_baseline"] = maj

        # 2.2 สั่งโมเดลประมวลผล NLP ทยอยสปอว์นแบทช์ทำนายดาวและ Head+Tail Truncation
        expected = predict_xlmr(df_xlmr)
        metrics = evaluate_model("XLM-R", df_xlmr, expected)
        results["models"]["xlmr"] = metrics # บันทึกสถิติ
        
        if args.save_predictions:
            save_predictions(df_xlmr, expected, "xlmr") # บันทึกผลลัพธ์คำทำนายละเอียด

    # ==============================================================================
    # 2.5 การตรวจสอบสำหรับโมเดล Embedding (BGE-M3 + Classifier)
    # ==============================================================================
    if "embedding" in active:
        df_embed = prepare_scoring_for_model(args.input, "embedding")
        y_true_embed = df_embed["user_rating"].values.astype(int)
        
        expected, probs = predict_embedding_with_probs(df_embed)
        metrics = evaluate_model("Embedding (BGE-M3)", df_embed, expected)
        results["models"]["embedding"] = metrics
        
        if args.save_predictions:
            save_predictions(df_embed, expected, "embedding")

    if results["models"]:
        if len(results["models"]) > 1:
            print_comparison(results["models"])
        best_model, best_val = pick_best_model(results["models"])
        results["best_model"] = best_model
        results["best_model_metric"] = {
            "name": config.WEB_MODEL_METRIC,
            "value": best_val,
        }
        print(
            f"\nBest model ({config.WEB_MODEL_METRIC}): {best_model} ({best_val:.4f})"
        )

    # 4. เขียนส่งออกค่าประเมินทั้งหมดลงไฟล์บันทึกประวัติการพัฒนา JSON
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics JSON: {args.output}")

    print("\nDone evaluation.")


if __name__ == "__main__":
    main() # ปลุกพลังสคริปต์หลักเมื่อรันใช้งานแบบตรงๆ
