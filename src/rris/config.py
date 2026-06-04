# -*- coding: utf-8 -*-
# ไฟล์ config.py: ศูนย์กลางการกำหนดคอนฟิกทั้งหมดของระบบ (Single Source of Truth)

from pathlib import Path  # ใช้จัดการพาธไฟล์และไดเรกทอรีแบบอิงออบเจกต์ (Cross-platform)
import os                # ใช้เข้าถึงตัวแปรสภาพแวดล้อม (Environment Variables) ของระบบปฏิบัติการ
import torch             # ไลบรารี PyTorch สำหรับการจัดการโมเดล Deep Learning และเรียกใช้ GPU

# กำหนดพาธรูทของโปรเจกต์ (repo root = สองระดับเหนือ src/rris/)
ROOT = Path(__file__).resolve().parent.parent.parent

def _p(*parts: str) -> str:
    """ฟังก์ชันตัวช่วยสร้างพาธสัมบูรณ์ (Absolute Path) จากรูทของโปรเจกต์"""
    return str(ROOT.joinpath(*parts))

# ==============================================================================
# 1. การตั้งค่าเกี่ยวกับข้อมูล (DATA CONFIGURATION)
# ==============================================================================
DATA_DIR = _p("data")                                      # โฟลเดอร์หลักสำหรับเก็บข้อมูลดิบและข้อมูลจำลอง
RAW_DATA_PATH = _p("data", "merge", "merged_reduce2.csv") # พาธเก็บไฟล์ข้อมูลสอนหลัก
TEST_PATH = _p("data", "70k", "70k_reduce.csv")        # พาธเก็บไฟล์ข้อมูลทดสอบทางการของ Wongnai
HOLDOUT_PATH = _p("data", "merge", "holdout.csv")          # พาธบันทึกข้อมูล Holdout (เก็บแยก 20% จาก Train สำหรับทดสอบระหว่างทาง)
HOLDOUT_FRACTION = 0.2                                     # สัดส่วนข้อมูลที่จะตัดไปเป็น Holdout (20%)

# พาธชุดข้อมูลย่อยตามประเภทข้อมูลดิบเพื่อรันสคริปต์สวีป/EDA
WONGNAI_TRAIN_PATH = _p("data", "wongnai", "train_reduce.csv")
WONGNAI_TEST_PATH = _p("data", "wongnai", "test.csv")
_70K_TRAIN_PATH = _p("data", "70k", "70k_reduce.csv")
MOCK_TRAIN_PATH = _p("data", "mock", "train.csv")
MOCK_TEST_PATH = _p("data", "mock", "test.csv")

# ==============================================================================
# 2. การล้างข้อมูลเบื้องต้นสำหรับ Baseline (DATA CLEANING STRATEGY)
# ==============================================================================
MIN_TEXT_LENGTH = 5             # กำหนดความยาวอักษรขั้นต่ำของรีวิว หากสั้นกว่านี้จะโดนกรองออก
DROP_DUPLICATE_TEXT = True      # ลบข้อความรีวิวที่ซ้ำซ้อนกันทิ้ง ป้องกันโมเดลจดจำแพทเทิร์นขยะ
DUPLICATE_KEEP = "first"        # หากเจอรีวิวซ้ำกัน ให้เก็บแถวแรกสุดที่พบไว้

# ==============================================================================
# 3. การตั้งค่าพาธสำหรับเก็บโมเดลและน้ำหนัก (ARTIFACTS CONFIGURATION)
# ==============================================================================
ARTIFACTS_DIR = _p("artifacts")                                          # โฟลเดอร์หลักสำหรับจัดเก็บโมเดลที่ฝึกฝนเสร็จแล้ว
BASELINE_ARTIFACTS_DIR = _p("artifacts", "baseline")                      # โฟลเดอร์จัดเก็บโมเดลกลุ่ม Baseline
XLMR_ARTIFACTS_DIR = _p("artifacts", "xlmr")                              # โฟลเดอร์จัดเก็บโมเดลกลุ่ม XLM-RoBERTa
XLMR_META_PATH = _p("artifacts", "xlmr", "xlmr_meta.json")
XLMR_LARGE_ARTIFACTS_DIR = _p("artifacts", "xlmr_large")                  # โฟลเดอร์จัดเก็บโมเดลกลุ่ม XLM-RoBERTa Large
XLMR_LARGE_META_PATH = _p("artifacts", "xlmr_large", "xlmr_meta.json")
TFIDF_VECTORIZER_PATH = _p("artifacts", "baseline", "tfidf_vectorizer.joblib") # พาธจัดเก็บ Word-level TF-IDF
CHAR_TFIDF_VECTORIZER_PATH = _p("artifacts", "baseline", "char_tfidf_vectorizer.joblib") # พาธจัดเก็บ Char-level TF-IDF
LSA_TRANSFORMER_PATH = _p("artifacts", "baseline", "lsa_transformer.joblib") # พาธจัดเก็บตัวลดมิติข้อมูล TruncatedSVD (LSA)
XGB_MODEL_PATH = _p("artifacts", "baseline", "xgb_model.json")            # พาธจัดเก็บไฟล์โครงสร้างและน้ำหนักของ XGBoost
SKLEARN_MODEL_PATH = _p("artifacts", "baseline", "sklearn_model.joblib")  # พาธจัดเก็บไฟล์โครงสร้างของ LinearSVC/sklearn
BASELINE_META_PATH = _p("artifacts", "baseline", "baseline_meta.json")    # ไฟล์ Metadata บันทึกคุณสมบัติการเทรนของ Baseline

# ==============================================================================
# 4. พาธไฟล์สำหรับส่วนส่งออกผลลัพธ์ (PRODUCTION PIPELINE OUTPUTS)
# ==============================================================================
OUTPUTS_DIR = _p("outputs")                                                 # โฟลเดอร์หลักสำหรับเก็บไฟล์ผลลัพธ์การทำงานของระบบ
SCORES_DIR = _p("outputs", "scores")                                        # โฟลเดอร์เก็บไฟล์ผลการทำนายเรตติ้งดาวแบบละเอียด
EVAL_DIR = _p("outputs", "eval")                                            # โฟลเดอร์เก็บข้อมูลสรุปตัววัดประสิทธิภาพในรูป JSON
REPORTS_DIR = _p("outputs", "reports")                                      # โฟลเดอร์เก็บรายงาน HTML แดชบอร์ดสรุปประสิทธิภาพโมเดล
DEFAULT_SCORED_OUTPUT = _p("outputs", "scores", "scored_output_minimal.csv") # พาธส่งออกไฟล์ทำนายดาวพร้อมสี Hex สำหรับเว็บแอป
DEFAULT_EVAL_REPORT = _p("outputs", "eval", "eval_report.json")             # พาธส่งออกรายงานตัววัดประสิทธิภาพโมเดลคู่อย่างละเอียด (JSON)
DEFAULT_EVAL_VIZ = _p("outputs", "reports", "eval_report_viz.html")         # พาธส่งออกหน้า HTML รายงานเปรียบเทียบประสิทธิภาพแบบ Interactive

# ==============================================================================
# 5. พาธไฟล์สำหรับส่วนงานวิจัยและทดลอง (EXPERIMENTS CONFIGURATION)
# ==============================================================================
EXPERIMENTS_DIR = _p("experiments")                                           # โฟลเดอร์จัดเก็บประวัติการทำวิจัย การ Sweep และการวิเคราะห์
BASELINE_EXPERIMENTS_DIR = _p("experiments", "baseline")                      # โฟลเดอร์เก็บรายงานวิเคราะห์ข้อผิดพลาดและบันทึกการจูน Baseline
EXPERIMENT_EDA_SUMMARY_PATH = _p("experiments", "baseline", "eda_summary.json") # ไฟล์สรุปสถิติความถี่คำและโครงสร้างชุดข้อมูลสอน
EXPERIMENT_TUNE_LOG_PATH = _p("experiments", "baseline", "tune_log.json")     # บันทึกประวัติและผลคะแนนการรัน Hyperparameter Grid Search
EXPERIMENT_TRY_LOG_PATH = _p("experiments", "baseline", "try_log.md")         # สมุดบันทึกประวัติการพัฒนาและผลลัพธ์การทดลองแต่ละรอบ (Try-Log)
EXPERIMENT_ERRORS_DIR = _p("experiments", "baseline", "errors")               # ไดเรกทอรีสำหรับส่งออกข้อผิดพลาดรุนแรงและเคสความมั่นใจต่ำ
EXPERIMENT_EVAL_DIR = _p("experiments", "baseline", "eval")                   # ไดเรกทอรีจัดเก็บประวัติผลประเมินย่อยของการ Sweep ตัวแปร
XLMR_EXPERIMENTS_DIR = _p("experiments", "xlmr")                              # โฟลเดอร์เก็บผลทดลอง XLM-R
XLMR_PREPROCESS_ABLATION_LOG = _p("experiments", "xlmr", "preprocess_ablation_log.json")  # บันทึกผลเปรียบเทียบ preprocessing strategies


# ==============================================================================
# 6. การจัดสรรฮาร์ดแวร์ประมวลผล (COMPUTATIONAL RESOURCES)
# ==============================================================================
def _cuda_runtime_ok() -> bool:
    """True when a small CUDA kernel runs successfully (any GPU, incl. RTX 50 sm_120)."""
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(8, 8, device="cuda")
        _ = x @ x.T
        torch.cuda.synchronize()
        del x, _
        return True
    except Exception:
        return False


def cuda_device_hint() -> str:
    """Human-readable hint when CUDA is visible but the runtime probe failed."""
    if not torch.cuda.is_available():
        return "PyTorch was built without CUDA or no NVIDIA driver is installed."
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    base = f"{name} (sm_{major}{minor}), torch {torch.__version__}"
    if major >= 12 and "+cu128" not in torch.__version__:
        return (
            f"{base}. RTX 50-series needs a CUDA 12.8 wheel "
            f"(e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128)."
        )
    return f"{base}. Reinstall a PyTorch build that matches your GPU/driver."


def _resolve_torch_device() -> str:
    forced = os.environ.get("FORCE_TORCH_DEVICE")
    if forced:
        return forced
    if os.environ.get("RRIS_FORCE_CUDA") == "1":
        return "cuda"
    return "cuda" if _cuda_runtime_ok() else "cpu"


def _resolve_xgb_device() -> str:
    forced = os.environ.get("FORCE_XGB_DEVICE")
    if forced:
        return forced
    return "cuda" if _cuda_runtime_ok() else "cpu"


# กำหนดอุปกรณ์ประมวลผลสำหรับ PyTorch / XGBoost (บังคับผ่าน FORCE_*_DEVICE ได้)
TORCH_DEVICE = _resolve_torch_device()
XGB_DEVICE = _resolve_xgb_device()

# ==============================================================================
# 7. กลยุทธ์การปรับปรุงความสมดุลของข้อมูลฝั่ง Baseline (BASELINE BALANCING STRATEGY)
# ==============================================================================
MAX_REVIEW_CHARS = 500                 # ตัดความยาวข้อความรีวิวไว้ที่ 500 ตัวอักษรเพื่อลดขนาดเวกเตอร์ TF-IDF
BASELINE_OVERSAMPLE_LOW_STARS = True   # เปิดการทำ Oversampling ข้อมูลกลุ่มรีวิวที่ให้คะแนนน้อย (1-2 ดาว) ที่มีสัดส่วนน้อย
BASELINE_OVERSAMPLE_FACTOR = 5         # อัตราการเพิ่มข้อมูลรีวิว 1-2 ดาวขึ้นเป็น 5 เท่าจากจำนวนเดิม
BASELINE_OVERSAMPLE_USE_WEIGHT = True  # ใช้การป้อนน้ำหนักตัวอย่างในการแก้ปัญหา Imbalance ร่วมกับการทำ Oversampling
BASELINE_MAJORITY_CLASS = 4            # คะแนนดาวฐานหลัก (4 ดาว) สำหรับการทำทำนายเดาแบบ Majority Baseline

# ==============================================================================
# 8. การแปลงคุณลักษณะของ Baseline (BASELINE FEATURE PIPELINE CONFIG)
# ==============================================================================
TFIDF_MAX_FEATURES = 8000              # จำนวนคำศัพท์สูงสุดที่จะสกัดจาก Word-level TF-IDF (จัดลำดับตามความถี่สูงสุด)
TFIDF_NGRAM_RANGE = (1, 2)             # ขนาดของกลุ่มคำที่จะวิเคราะห์ (Word Unigrams และ Bigrams เช่น "อร่อย", "อร่อย มาก")
TFIDF_MIN_DF = 2                       # คำที่นำมาวิเคราะห์ต้องพบอย่างน้อยใน 2 เอกสารขึ้นไปเพื่อกรองคำที่เกิดครั้งเดียวทิ้ง
TFIDF_MAX_DF = 0.9                     # คำที่พบในกว่า 90% ของรีวิวทั้งหมดจะถูกลบทิ้ง (คำที่โผล่เยอะจนไม่มีอำนาจแยกแยะกลุ่มดาว)
BASELINE_USE_LSA = False               # ไม่เปิดตัวลดมิติข้อมูลแฝง Latent Semantic Analysis (ปิดไว้ตามการทดลอง Ablation)
LSA_N_COMPONENTS = 400                 # หากเปิดใช้ LSA จะลดความละเอียดของเวกเตอร์ TF-IDF ลงเหลือ 400 มิติ
BASELINE_USE_EXTRA_FEATURES = False     # เปิดใช้ลักษณะพิเศษเฉพาะตัว (ความยาวอักษร, จำนวนคำ, และจำนวนคำปฏิเสธภาษาไทย)
BASELINE_USE_CHAR_TFIDF = True         # เปิดใช้ Character-level TF-IDF เพื่อแก้ปัญหาคำไทยสะกดผิดและเพิ่มอำนาจการแยกแยะคำศัพท์
BASELINE_CHAR_MAX_FEATURES = 4000      # สกัดคุณลักษณะย่อยระดับตัวอักษรสูงสุด 4,000 มิติ
BASELINE_CHAR_NGRAM_RANGE = (3, 5)     # ค้นหาแพทเทิร์นลำดับตัวอักษรสะกดตั้งแต่ 3 ถึง 5 ตัวอักษรต่อเนื่อง

# ==============================================================================
# 9. คอนฟิกการทดลองตัวแปรฝั่ง Baseline (BASELINE TRAINING STRATEGY)
# ==============================================================================
BASELINE_UNDERSAMPLE_STAR4_FRACTION = 0.65 # สัดส่วนการสุ่มตัดข้อมูลรีวิว 4 ดาวลงมาเหลือ 65% เพื่อช่วยให้ดาวอื่นๆ มีน้ำหนักเพิ่มขึ้น
BASELINE_MOCK_MIX_FRACTION = 0.2           # นำเข้าข้อมูลจำลองจำนวน 20% เข้ามาปะปนช่วยเสริมความแข็งแกร่งในการทำ Smoke Test
BASELINE_USE_3CLASS = False                # ไม่แบ่งข้อมูลออกเป็น 3 กลุ่มใหญ่ (คะแนนลบ 1-2★, กลาง 3★, คะแนนบวก 4-5★) ปิดไว้
BASELINE_USE_REGRESSION = False            # ปิดโหมดการทำนายแบบถดถอยค่าต่อเนื่อง (ใช้การแยกประเภทหลายคลาสโดยตรงเพื่อความเสถียร)
BASELINE_USE_OPTUNA = False                # เปิดใช้ Bayesian Hyperparameter Optimization (Optuna) เพื่อหาพารามิเตอร์ที่ดีที่สุดอัตโนมัติ
BASELINE_KFOLD = 5                         # จำนวนรอบพับการประเมิน Stratified K-Fold Cross Validation (เปลี่ยนจาก 1 เป็น 5)

# ==============================================================================
# 9.5 การตั้งค่า Data Augmentation (DATA AUGMENTATION STRATEGY)
# ==============================================================================
AUGMENT_ENABLED = True                     # เปิด/ปิดระบบขยายข้อมูลอัตโนมัติสำหรับคลาสดาวน้อย
AUGMENT_TARGET_STARS = (1, 2, 3)           # คลาสดาวเป้าหมายที่จะทำ augmentation (ดาวที่มีข้อมูลน้อย)
AUGMENT_TARGET_COUNT = 1400                 # จำนวนตัวอย่างเป้าหมายต่อคลาสหลัง augmentation (ถมให้ถึงหลักพัน)
AUGMENT_SYNONYM_PROB = 0.3                 # ความน่าจะเป็นในการสุ่มแทนที่คำด้วยคำพ้องความหมาย (30%)
AUGMENT_SHUFFLE_PROB = 0.2                 # ความน่าจะเป็นในการสลับลำดับคำในประโยค (20%)
AUGMENT_RANDOM_STATE = 42                  # ค่าความสุ่มคงที่สำหรับ reproducibility ของ augmentation
AUGMENT_FROM_ERRORS = False                # เปิด augment จาก error analysis export
AUGMENT_FROM_ERRORS_PATH = ""              # path ไป severe error CSV
AUGMENT_FROM_ERRORS_FACTOR = 2             # จำนวนคopies ต่อ error row

# ==============================================================================
# 10. การกำหนดค่าเพื่อสกัดข้อผิดพลาดและวิเคราะห์ (ERROR ANALYSIS SETTINGS)
# ==============================================================================
ERROR_EXPORT_MIN_DELTA = 2             # ส่งออกข้อผิดพลาดที่ทำนายห่างจากดาวจริงตั้งแต่ 2 ดาวขึ้นไป (เช่น ดาวจริง 5★ ทายได้ 3★)
LOW_CONFIDENCE_THRESHOLD = 0.4         # เกณฑ์สำหรับจำแนกความมั่นใจต่ำ (ผลความน่าจะเป็นความน่าเชื่อถือสูงสุดไม่ถึง 40%)

# ==============================================================================
# 11. พารามิเตอร์อัลกอริทึม XGBoost Native API (BASELINE MODEL HYPERPARAMETERS)
# ==============================================================================
XGB_PARAMS = {
    "objective": "multi:softprob",     # ปัญหาการแยกประเภทหลายคลาส โดยคำนวณเอาผลความน่าจะเป็นของแต่ละคลาสดาว
    "num_class": 5,                    # จำนวนคลาสของดาวเป้าหมาย (0 ถึง 4 แทนเรตติ้ง 1 ถึง 5★)
    "max_depth": 4,                    # ความลึกสูงสุดของต้นไม้ตัดสินใจแต่ละต้น (ความลึกต่ำป้องกันโมเดลโอเวอร์ฟิต)
    "eta": 0.03,                       # อัตราการเรียนรู้การอัปเดตน้ำหนักของแต่ละต้น (Learning Rate = 0.03)
    "eval_metric": "mlogloss",         # ตัวประเมินความคืบหน้าการสร้างความสูญเสียในโหมด Multi-class Logloss
    "tree_method": "hist",             # อัลกอริทึมการวิเคราะห์สร้างฮิสโตแกรมความถี่เพื่อประมวลผลอย่างรวดเร็วเป็นพิเศษ
    "device": XGB_DEVICE,              # ระบุชิปประมวลผลการ์ดจอ (GPU) หรือซีพียู (CPU) ตามที่ระบบค้นพบ
    "min_child_weight": 5,             # จำนวนตัวอย่างขั้นต่ำที่ต้องการในแต่ละปุ่มใบไม้ตัดสินใจเพื่อความยั่งยืน
    "subsample": 0.8,                  # อัตราส่วนการสุ่มเลือกตัวอย่างข้อมูลสอน (80%) ในการสร้างต้นไม้แต่ละต้น
    "colsample_bytree": 0.8,           # อัตราส่วนการสุ่มเลือกคอลัมน์คุณลักษณะ (80%) ในการวิเคราะห์แตกกิ่งก้าน
    "reg_lambda": 3.0,                 # อัตราความยืดหยุ่น L2 Regularization (ป้องกันต้นไม้ซับซ้อนเกินไป)
    "reg_alpha": 1.0,                  # อัตราความยืดหยุ่น L1 Regularization (ช่วยคัดตัวแปรเด่นออกไปใช้งาน)
}
XGB_ROUNDS = 800                       # จำนวนรอบการเทรนต้นไม้ตัดสินใจสูงสุด 800 รอบ
XGB_EARLY_STOPPING_ROUNDS = 50         # หยุดเทรนหากค่าสูญเสียความแม่นยำบน Holdout ไม่ลดลงเลยติดต่อกัน 50 รอบต้นไม้
XGB_USE_SAMPLE_WEIGHT = True           # เปิดการแนบน้ำหนักตัวอย่างในการเทรน XGBoost เพื่อความเที่ยงตรงดาวน้อย
XGB_LOW_STAR_BOOST = 3.0               # ค่าตัวคูณเร่งพิเศษสำหรับการทำโทษความผิดพลาดของรีวิวคะแนนดาวต่ำ 1-2 ดาว

# ==============================================================================
# 11.5 พารามิเตอร์การตั้งค่าโมเดล Embedding (SENTENCE EMBEDDING + CLASSIFIER)
# ==============================================================================
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"   # โมเดล Embedding (ตัวเลือกอื่น: 'BAAI/bge-m3')
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 128
EMBEDDING_ARTIFACTS_DIR = _p("artifacts", "embedding")
EMBEDDING_CACHE_PATH = _p("data", "embedding_cache.joblib")  # legacy; prefer hashed cache
EMBEDDING_CLF_TYPE = "xgb"              # 'xgb' หรือ 'lr' (Logistic Regression)
EMBEDDING_LR_MAX_ITER = 1000
EMBEDDING_FINETUNE = False
EMBEDDING_FINETUNE_MODEL = "BAAI/bge-m3"
EMBEDDING_FINETUNE_MODE = "supervised"  # supervised | contrastive
EMBEDDING_FINETUNE_EPOCHS = 2
EMBEDDING_FINETUNE_BATCH_SIZE = 16
EMBEDDING_FINETUNE_OUTPUT = _p("artifacts", "embedding", "finetuned_model")

# ==============================================================================
# 12. พารามิเตอร์การจูนโมเดล XLM-RoBERTa (ADVANCED MODEL HYPERPARAMETERS)
# ==============================================================================
XLMR_MODEL_NAME = "xlm-roberta-base"   # ชื่อพรีเทรนโมเดลบน Hugging Face Hub (ขนาด 125M พารามิเตอร์)
XLMR_LARGE_MODEL_NAME = "xlm-roberta-large" # ชื่อพรีเทรนโมเดลขนาดใหญ่ (ขนาด 355M พารามิเตอร์)
MAX_LENGTH = 128                       # ความยาวโทเคนสูงสุดต่อรีวิวที่รองรับ (หากยาวกว่านี้จะทำการตัดหั่นแบบ Head+Tail)
BATCH_SIZE = 24                        # ขนาดตัวอย่างต่อการก้าวรันหนึ่งครัง (ลดขนาดลงเพื่อป้องกันหน่วยความจำการ์ดจอแตกบน GPU 6GB)
XLMR_LARGE_BATCH_SIZE = 4               # ขนาดแบทช์สำหรับโมเดลใหญ่ (ลดลงเพื่อป้องกัน OOM)
XLMR_GRAD_ACCUM_STEPS = 1              # เพิ่มระดับสะสมเกรเดียนต์เพื่อรักษา Effective Batch Size = 8 เท่าเดิม
XLMR_LARGE_GRAD_ACCUM_STEPS = 4        # รอบสะสมเกรเดียนต์สำหรับโมเดลใหญ่
XLMR_USE_AMP = True                    # เปิดโหมด Automatic Mixed Precision ใช้ทศนิยม 16 บิต (FP16) ลดทอนแรมการ์ดจอลงเท่าตัว
XLMR_GRADIENT_CHECKPOINTING = False     # เปิดใช้ระบบฝากผลเกรเดียนต์ไว้คำนวณใหม่แทนการเก็บค้างค้างเพื่อเซฟแรมการ์ดจอขั้นสุด (เหลือความจุขั้นต่ำ 6GB)
XLMR_LARGE_GRADIENT_CHECKPOINTING = True # บังคับเปิดเพื่อโมเดลใหญ่เพื่อลดแรมการ์ดจอลง
LEARNING_RATE = 2e-5                   # อัตราความเร็วในการปรับตัวโมเดล NLP ปรับตัวโมเดลระดับสูง
EPOCHS = 3                             # จำนวนรอบการวิ่งสอนผ่านข้อมูลทั้งหมดสูงสุด 5 รอบ
WEIGHT_DECAY = 0.01                    # อัตราการลดทอนค่าน้ำหนักตัวแปร L2 Regularization ป้องกัน Overfitting
XLMR_USE_CLASS_WEIGHT = True           # เปิดใช้งาน Weighted Cross Entropy Loss เพื่อถ่วงน้ำหนักความถูกต้องให้ดาวน้อย
XLMR_LOW_STAR_BOOST = 1.5              # ตัวคูณเร่งพิเศษสำหรับโมเดลระดับสูงเมื่อเทรนกลุ่มดาว 1-2 ดาว
XLMR_EARLY_STOPPING_PATIENCE = 3       # ระบบจะสั่งหยุดทันทีหากความแม่นยำบน Holdout คงที่ต่อเนื่องกัน 3 รอบ Epoch
XLMR_USE_LR_SCHEDULER = True           # เปิดใช้งานตัวปรับแต่งค่าความเร็วในการเรียนรู้อัตโนมัติ (Linear Warmup Scheduler)
XLMR_USE_REGRESSION = True             # Ordinal Regression (MSE Loss); disables Focal Loss below

# --- Focal Loss สำหรับ XLM-R ---
XLMR_USE_FOCAL_LOSS = True             # Used only when XLMR_USE_REGRESSION=False (classification mode)
XLMR_FOCAL_ALPHA = None                # Alpha สำหรับ Focal Loss (None = ใช้ class_weights แทน, หรือกำหนด list 5 ค่า)
XLMR_FOCAL_GAMMA = 2.0                 # Gamma สำหรับ Focal Loss (ยิ่งสูง ยิ่งโฟกัสเคสยากมากขึ้น, ค่ามาตรฐาน = 2.0)

# --- Preprocessing Strategy สำหรับ XLM-R ---
XLMR_PREPROCESS_STRATEGY = "aggressive" # กลยุทธ์การล้างข้อมูลสำหรับ XLM-R ('default', 'minimal', 'aggressive', 'keep_digits', 'emoji_tag', 'segment')

# --- 3-Class Label Grouping สำหรับ XLM-R ---
XLMR_USE_3CLASS = False                # ยุบรวมคลาส: Negative (1-2★), Neutral (3★), Positive (4-5★)
XLMR_3CLASS_LABEL_NAMES = ["Negative", "Neutral", "Positive"]  # ชื่อเรียกคลาสเพื่อใช้ในรายงาน
XLMR_LR_SCHEDULER_FACTOR = 0.5         # ตัวแปรเสริม (ไม่ได้ใช้งานแล้วเนื่องจากเปลี่ยนไปใช้ Linear schedule with Warmup)
XLMR_LR_SCHEDULER_PATIENCE = 1         # ตัวแปรเสริม (ไม่ได้ใช้งานแล้วเนื่องจากเปลี่ยนไปใช้ Linear schedule with Warmup)

# ==============================================================================
# 13. การประเมินและการทำนายผลตรวจจับความผิดปกติ (SCORING SETTINGS)
# ==============================================================================
ANOMALY_THRESHOLD = 2.0                # กำหนดความต่างของดาวจริงกับดาวทำนายของ AI ที่เริ่มเห็นความไม่เข้าพวก (ตั้งแต่ 2.0 ขึ้นไป)
RANDOM_STATE = 42                      # ค่าตั้งสุ่มคงที่สากลเพื่อให้สามารถตรวจสอบและรันผลให้ตรงกันได้ทุกครั้ง (Reproducibility)
LSA_RANDOM_STATE = RANDOM_STATE        # ค่าความสุ่มสำหรับขั้นตอนการทำลดทอนมิติ LSA

# ==============================================================================
# Smoke / CI overrides (set RRIS_SMOKE=1 for fast end-to-end checks)
# ==============================================================================
import os as _os

if _os.environ.get("RRIS_SMOKE") == "1":
    # Fast smoke defaults; use GPU unless explicitly forced to CPU (CI / broken CUDA).
    if _os.environ.get("RRIS_SMOKE_FORCE_CPU") == "1":
        TORCH_DEVICE = "cpu"
        XGB_DEVICE = "cpu"
    EPOCHS = 1
    XLMR_EARLY_STOPPING_PATIENCE = 1
    XLMR_USE_AMP = False
    XLMR_GRADIENT_CHECKPOINTING = False
    BATCH_SIZE = max(4, BATCH_SIZE)
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_BATCH_SIZE = 32
    BASELINE_OVERSAMPLE_LOW_STARS = False
    AUGMENT_ENABLED = False

# ==============================================================================
# 14. Web dashboard model selection
# ==============================================================================
WEB_MODEL_METRIC = "mae"
WEB_MODEL_DEFAULT = "auto"
WEB_MODEL_FALLBACK = "baseline"
