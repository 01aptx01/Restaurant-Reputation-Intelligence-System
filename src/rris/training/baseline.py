# -*- coding: utf-8 -*-
# ไฟล์ train_baseline.py: สคริปต์ฝึกสอนโมเดล Baseline (TF-IDF + XGBoost Native API) ร่วมกับกลยุทธ์การปรับสมดุลดาวขั้นสูงสุด

import json # เครื่องมือจัดทำข้อมูลสรุปและเก็บประวัติประมวลผลโมเดล JSON
import os   # ตัวดำเนินการติดต่อเข้าใช้ไดเรกทอรีระบบปฏิบัติการ

import joblib      # ตัวเขียนบันทึกไฟล์ออบเจกต์ตัวสกัดฟีเจอร์ลงสู่ไดเรกทอรีเครื่อง
import numpy as np # ไลบรารีคำนวณทางคณิตศาสตร์และการประยุกต์ใช้อาเรย์เชิงเวกเตอร์
import pandas as pd # ไลบรารีจัดการและจัดแต่งตารางข้อมูล
import xgboost as xgb # อัลกอริทึมต้นไม้ตัดสินใจที่มีการไล่ระดับน้ำหนักความผิดพลาด (XGBoost Native API)
from scipy.sparse import csr_matrix, hstack # เครื่องมือสร้างเมทริกซ์กระจาย (Sparse) และแนบต่อแนวขวาง
from sklearn.decomposition import TruncatedSVD # ตัวช่วยวิเคราะห์ลดทอนมิติคุณลักษณะแบบ LSA
from sklearn.feature_extraction.text import TfidfVectorizer # ตัวสกัดฟีเจอร์ความถี่คำระดับประโยค (TF-IDF)
from sklearn.model_selection import train_test_split # ฟังก์ชันสำหรับแบ่งซอยตารางชุดเทรนและประเมินผล
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

from rris import config # นำเข้าศูนย์กลางพารามิเตอร์และคอนฟิกูเรชันหลัก
from rris import utils  # นำเข้าตัวช่วยล้างทำความสะอาดข้อความและฟังก์ชันคำนวณถ่วงน้ำหนัก
from rris.data.augmentation import apply_train_augmentation
from rris.data.normalize import BASELINE_PREPROCESS_STRATEGY
from rris.evaluation.selection import (
    classifier_val_mae,
    pick_lowest_mae,
    xgb_booster_val_mae,
)
from rris.inference.common import expected_rating_from_probs


def train_xgb_native(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict,
) -> tuple[xgb.Booster, dict, float]:
    """ฝึกสอนต้นไม้ตัดสินใจด้วย XGBoost Native API พร้อมรองรับสลับการรันกลับไปยัง CPU โดยอัตโนมัติหากการ์ดจอหน่วยความจำเต็ม"""
    watchlist = [(dtrain, "train"), (dval, "val")] # รายชื่อชุดข้อมูลที่จะส่องตรวจวัดค่าสูญเสียความแม่นยำระหว่างสร้างต้นไม้
    evals_result: dict = {} # ดิกชันนารีเก็บผลประวัติการลดค่าความผิดพลาด

    def _run(p: dict) -> tuple[xgb.Booster, dict]:
        """ฟังก์ชันภายในสั่งจ้างเทรน Booster อ้างอิงพารามิเตอร์ที่ระบุ"""
        er: dict = {}
        bst = xgb.train(
            params=p,                         # พารามิเตอร์คอนฟิกของอัลกอริทึม
            dtrain=dtrain,                     # ชุดข้อมูลฝึกสอนหลัก
            num_boost_round=config.XGB_ROUNDS, # จำนวนต้นไม้ตัดสินใจสูงสุดที่สร้าง (เช่น 800 ต้น)
            evals=watchlist,                   # ตารางเฝ้าส่องความคืบหน้า
            evals_result=er,                   # ล็อกค่าเก็บผลประเมินความก้าวหน้า
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS, # สั่งหยุดสร้างต้นไม้เพิ่มถ้าค่าความต่าง Holdout คงที่
            verbose_eval=10,                   # พิมพ์ยอดรายงานสถิติค่าสูญเสียความแม่นยำลงจอวิจัยทุกๆ 10 รอบ
        )
        return bst, er # คืนร่าง Booster และประวัติการประเมิน

    try:
        bst, evals_result = _run(params) # 1. พยายามสตาร์ทรันฝึกสอนด้วยชิป GPU การ์ดจอ NVIDIA CUDA ตามเป้าหมาย
    except xgb.core.XGBoostError as exc:
        # 2. ปลุกระบบกู้ภัยอัตโนมัติ (Fallback Strategy)
        # หากแรมการ์ดจอเต็มหรือเกิดบั๊กตัวประมวลผลการ์ดจอ จะดึงโค้ดพารามิเตอร์กลับมารันบน CPU ทันทีเพื่อให้งานไม่สะดุด
        if params.get("device") == "cuda":
            print(f"GPU training failed ({exc}); retrying on CPU.")
            cpu_params = {**params, "device": "cpu"} # บังคับเปลี่ยนฮาร์ดแวร์เป็น CPU
            bst, evals_result = _run(cpu_params)      # เทรนผ่านซีพียูทดแทน
        else:
            raise # ส่งผ่านข้อผิดพลาดอื่นๆ ออกไปข้างนอกหากไม่ได้เป็นปัญหาการ์ดจอ

    best_iter = bst.best_iteration # ค้นหารอบต้นไม้ตัดสินใจที่มีคะแนนดีที่สุด
    val_hist = evals_result.get("val", {}).get("mlogloss", []) # ดึงลิสต์ประวัติค่าผิดพลาดชุดประเมิน Holdout
    best_val = float(val_hist[best_iter]) if val_hist else float("inf") # โหลดค่าความสูญเสียต่ำสุดที่เป็นจริง
    return bst, evals_result, best_val # ส่งคืน Booster พร้อมรายละเอียดประวัติประเมิน


def _build_word_tfidf() -> TfidfVectorizer:
    """จัดสรรระบบคำนวณความถี่คำ Word-level TF-IDF อ้างอิงตัวตัดคำภาษาไทย PyThaiNLP"""
    return TfidfVectorizer(
        tokenizer=utils.thai_tokenizer,          # ดึงตัวล้างและตัดคำภาษาไทย (newmm)
        token_pattern=None,                      # ปิดค่าแพทเทิร์น regex มาตรฐานสำหรับภาษาอังกฤษทิ้ง
        max_features=config.TFIDF_MAX_FEATURES,  # จำนวนคำศัพท์สูงสุด (เช่น 8,000 คำศัพท์เด่น)
        ngram_range=config.TFIDF_NGRAM_RANGE,    # วิเคราะห์ลากเชื่อมคำ (Unigrams & Bigrams)
        min_df=config.TFIDF_MIN_DF,              # ดึงคำศัพท์ที่พบมากกว่า 1 ข้อความขึ้นไป
        max_df=config.TFIDF_MAX_DF,              # ลบคำศัพท์ที่มีสัดส่วนโผล่เยอะเกิน 90% ของรีวิวทั้งหมด
    )


def _build_char_tfidf() -> TfidfVectorizer:
    """จัดสรรระบบคำนวณ Character-level TF-IDF เพื่อสกัดโครงสร้างตัวอักษรและพยางค์ช่วยแก้ปัญหาคำไทยเขียนผิด"""
    return TfidfVectorizer(
        analyzer="char_wb",                             # สกัดฟีเจอร์ระดับตัวสะกดเฉพาะภายในขอบเขตคำ
        ngram_range=config.BASELINE_CHAR_NGRAM_RANGE,   # หาแพทเทิร์นตัวสะกด 3 ถึง 5 อักขระต่อเนื่อง
        max_features=config.BASELINE_CHAR_MAX_FEATURES, # สกัดคุณลักษณะอักษรเด่นสุด 4,000 มิติ
        min_df=config.TFIDF_MIN_DF,                     # ขีดจำกัดขั้นต่ำความถี่
        max_df=config.TFIDF_MAX_DF,                     # ขีดจำกัดเพดานความถี่
    )


def _stack_features(parts: list) -> object:
    """รวมเวกเตอร์ความถี่หลายมิติเข้าขนานกันทางด้านขวา (Horizontal Stack) ในแบบ CSR Sparse Matrix"""
    if len(parts) == 1:
        return parts[0] # ส่งชิ้นส่วนเดี่ยวคืนทันทีไม่ต้องประกบ
    return hstack(parts, format="csr") # ประกบแนวนอนจัดเก็บโครงสร้าง CSR Sparse Matrix มิติคุ้มแรม


def fit_vectorizer_and_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[
    TfidfVectorizer,
    TfidfVectorizer | None,
    TruncatedSVD | None,
    object,
    object,
    float | None,
]:
    """สกัดฟีเจอร์หลัก (Word/Char TF-IDF, LSA, Extras) โดยสร้างโครงสร้างจากชุดสอนและแปลงผลชุด Holdout ป้องกันข้อมูลรั่วไหล (Data Leakage)"""
    vectorizer = _build_word_tfidf() # สร้างตัววิเคราะห์ TF-IDF คำศัพท์
    X_train_vec = vectorizer.fit_transform(df_train["text"]) # สร้างคำศัพท์และแปลงความถี่ในชุดสอน (Fit & Transform)
    X_val_vec = vectorizer.transform(df_val["text"])         # แปลงความถี่ชุด Holdout โดยห้ามสร้างคำใหม่เพิ่ม (Transform only)

    char_vectorizer = None
    # ทำการแปลงคุณลักษณะเสริมระดับตัวอักษรหากมีการตั้งค่า
    if config.BASELINE_USE_CHAR_TFIDF:
        char_vectorizer = _build_char_tfidf() # สร้างตัววิเคราะห์ TF-IDF ตัวอักษร
        X_train_char = char_vectorizer.fit_transform(df_train["text"]) # เรียนรู้และแปลงคุณลักษณะในชุดสอน
        X_val_char = char_vectorizer.transform(df_val["text"])         # แปลงคุณลักษณะชุด Holdout
        # เชื่อมประสานเวกเตอร์คำศัพท์และเวกเตอร์ระดับตัวสะกดเข้าด้วยกันทางขวา
        X_train_vec = _stack_features([X_train_vec, X_train_char])
        X_val_vec = _stack_features([X_val_vec, X_val_char])

    svd = None
    explained = None
    use_lsa = config.BASELINE_USE_LSA                             # ดึงนโยบายลดทอนมิติ LSA
    use_extra = config.BASELINE_USE_EXTRA_FEATURES               # ดึงนโยบายเพิ่มลักษณะพิเศษเฉพาะตัว

    # กรณี 1: ทำระบบวิเคราะห์มิติลดรูป Latent Semantic Analysis (LSA)
    if use_lsa:
        svd = TruncatedSVD(
            n_components=config.LSA_N_COMPONENTS,              # มิติเป้าหมาย (เช่น 400 มิติ)
            random_state=config.LSA_RANDOM_STATE,              # ล็อกความสุ่มตรวจสอบ
        )
        X_train_lsa = svd.fit_transform(X_train_vec)            # คำนวณแกนหมุนและแปลงขนาดชุดสอน
        explained = float(np.sum(svd.explained_variance_ratio_)) # คำนวณเปอร์เซ็นต์การรักษาเนื้อความรู้เดิม (Explained Variance)
        
        if use_extra:
            # โหลดคำนวณลักษณะเฉพาะ (ความยาวอักษร, จำนวนคำตัด, และคำปฏิเสธภาษาไทย) มาประกบแนวขวาง
            extra_train = utils.compute_extra_features(df_train["text"])
            extra_val = utils.compute_extra_features(df_val["text"])
            X_train = np.hstack([X_train_lsa, extra_train]) # เชื่อมต่อแบบอาเรย์หนาแน่น (Dense Array)
            X_val = np.hstack([svd.transform(X_val_vec), extra_val])
        else:
            X_train = X_train_lsa # ใช้ผล LSA โดยตรง
            X_val = svd.transform(X_val_vec)
            
    # กรณี 2: ไม่เปิด LSA แต่ประยุกต์ลักษณะเฉพาะเสริม (Extra Features) (Winner Strategy ในรอบ Ablation)
    elif use_extra:
        # สกัดฟีเจอร์ความรู้สึกเสริมจากเนื้อความรีวิวภาษาไทย
        extra_train = utils.compute_extra_features(df_train["text"])
        extra_val = utils.compute_extra_features(df_val["text"])
        # ประกบเมทริกซ์ความถี่กระจาย (Sparse) และฟีเจอร์เสริมในรูปแบบ CSR Matrix ปลอดภัยแรม
        X_train = hstack(
            [X_train_vec, csr_matrix(extra_train)],
            format="csr",
        )
        X_val = hstack(
            [X_val_vec, csr_matrix(extra_val)],
            format="csr",
        )
        
    # กรณี 3: ใช้เวกเตอร์ความถี่ TF-IDF เพียวๆ
    else:
        X_train = X_train_vec
        X_val = X_val_vec

    # คืนโครงสร้างผลลัพธ์เวกเตอร์สกัดคำพร้อมสถิติ variance ที่ได้
    return vectorizer, char_vectorizer, svd, X_train, X_val, explained


def _training_labels(ratings: np.ndarray) -> np.ndarray:
    """แปรรูปคะแนนดาวจริงของลูกค้า ให้กลายเป็นป้ายเป้าหมาย (Labels) ตามสเปกการเทรนของโมเดลแต่ละแบบ"""
    if config.BASELINE_USE_3CLASS:
        # ยุบดาว 5 ระดับ เหลือกลุ่มประเมิน 3 คลาสใหญ่ (ลบ 0, กลาง 1, บวก 2)
        return utils.rating_to_3class(ratings)
    if config.BASELINE_USE_REGRESSION:
        # ปรับสเกลดาว 1-5 ให้อยู่ในช่วงทศนิยม 0.0 - 4.0 สำหรับโมเดลทำนายแบบถดถอย
        return ratings.astype(np.float32) - 1.0
    # ปรับสเกลดาว 1-5 ให้อยู่ช่วงดัชนีคลาสจำนวนเต็ม 0..4 สำหรับโหมดแยกประเภท 5 คลาส
    return ratings - 1


def make_dmatrices(
    X_train,
    X_val,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    """แพ็กคุณลักษณะเวกเตอร์และป้ายเป้าหมายจัดสรรลงในรูป DMatrix คู่น้ำหนักชดเชย Imbalance สำหรับส่งป้อน XGBoost"""
    y_train = _training_labels(df_train["user_rating"].values) # แปลงเป้าหมายชุดสอน
    y_val = _training_labels(df_val["user_rating"].values)     # แปลงเป้าหมายชุดประเมิน

    # ประเมินการถ่วงน้ำหนักตัวอย่างแบบละเอียดยิบ (Sample Weights) ร่วมกับ XGBoost
    if config.XGB_USE_SAMPLE_WEIGHT and not config.BASELINE_USE_REGRESSION:
        # คำนวณเวกเตอร์ลงโทษความกว้างดาว โดยเร่งทวีโทษให้ดาวน้อย (1-2 ดาว) หนักหน่วงพิเศษ 3 เท่า
        class_w = utils.compute_class_weights(
            df_train["user_rating"].values,
            low_star_boost=config.XGB_LOW_STAR_BOOST,
        )
        # ถ่ายทอดน้ำหนักระดับคลาสลงไปยังแต่ละแถวตัวอย่างรีวิวต้นทาง
        sample_w = utils.compute_sample_weights_from_ratings(
            df_train["user_rating"].values,
            class_w,
        )
        # แพ็กเข้าโครงสร้าง DMatrix พร้อมข้อมูลน้ำหนักถ่วงโทษ (Sample weights)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
    else:
        # แพ็กข้อมูล DMatrix แบบไม่มีการชดเชยน้ำหนัก
        dtrain = xgb.DMatrix(X_train, label=y_train)

    dval = xgb.DMatrix(X_val, label=y_val) # แพ็ก Holdout DMatrix สำหรับวัดเกณฑ์การหยุดสร้างต้นไม้
    return dtrain, dval # ส่งคู่ข้อมูล DMatrix กลับคืน


def save_holdout_csv(df_holdout: pd.DataFrame) -> None:
    """บันทึกตัดตาราง Holdout 20% นอกวงเวียนเทรนไปเขียนลงในฮาร์ดดิสก์ไฟล์ CSV เพื่อเป็นตัวตรวจวัดประสิทธิภาพอย่างเที่ยงตรง"""
    os.makedirs(os.path.dirname(config.HOLDOUT_PATH), exist_ok=True) # คุ้มครองการสร้างโฟลเดอร์ปลายทาง
    df_holdout.to_csv(config.HOLDOUT_PATH, index=False)              # เซฟลงตารางข้อมูลหลัก
    print(f"Saved holdout split: {config.HOLDOUT_PATH} (n={len(df_holdout)})")


def save_artifacts(
    vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer | None,
    svd: TruncatedSVD | None,
    bst: xgb.Booster,
    meta: dict,
) -> None:
    """จัดเก็บตัวแปลงฟีเจอร์โมเดล และไฟล์ Metadata ทั้งหมดลงเครื่อง เพื่อให้พร้อมใช้งานทันทีฝั่ง Inference และฝั่งเว็บ"""
    os.makedirs(config.BASELINE_ARTIFACTS_DIR, exist_ok=True) # ตรวจจับและสร้างเส้นทางไดเรกทอรี
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)     # เซฟ Word TF-IDF
    
    # ดำเนินการจัดการบันทึกหรือลบไฟล์ Char TF-IDF ตามโครงสร้างการจูนจริง
    if char_vectorizer is not None:
        joblib.dump(char_vectorizer, config.CHAR_TFIDF_VECTORIZER_PATH)
    elif os.path.isfile(config.CHAR_TFIDF_VECTORIZER_PATH):
        os.remove(config.CHAR_TFIDF_VECTORIZER_PATH) # ขจัดไฟล์เก่าค้างออกเพื่อไม่ให้เกิดความสับสน
        
    # ดำเนินการจัดการบันทึกหรือลบไฟล์ SVD ตามความสอดคล้อง LSA
    if config.BASELINE_USE_LSA and svd is not None:
        joblib.dump(svd, config.LSA_TRANSFORMER_PATH)
    elif os.path.isfile(config.LSA_TRANSFORMER_PATH):
        os.remove(config.LSA_TRANSFORMER_PATH)
        
    best_model_type = meta.get("best_model_type", "xgboost")
    if best_model_type == "xgboost":
        bst.save_model(config.XGB_MODEL_PATH) # เขียนส่งออกไฟล์โครงสร้างต้นไม้บูสเตอร์ดิบของ XGBoost (.json)
    else:
        # บันทึกโมเดล Sklearn ลง Joblib
        sklearn_model_path = os.path.join(config.BASELINE_ARTIFACTS_DIR, "sklearn_model.joblib")
        joblib.dump(bst, sklearn_model_path)
    
    # บันทึกจดประวัติลักษณะฟีเจอร์และตัวแปรควบคุม (Metadata) ลงตาราง JSON เพื่อคุ้มครองความเข้ากันได้
    with open(config.BASELINE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    """แกนโปรแกรมหลักควบคุมขั้นตอนประมวลผลฝึกสอน Baseline NLP สลักขจัด Noise และความอสมมาตรข้อมูล"""
    print("--- Step 1: Loading data ---")
    # 1. โหลดข้อมูลข้อความรีวิวและระดับดาวแท้จากไฟล์ Wongnai Train CSV
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH)

    # 2. ทำการล้างข้อความกรองรีวิวว่างเปล่า รีวิวที่สั้นเกินไป และรีวิวที่ทำซ้ำซ้อนกันเพื่อลดขนาดประชากรและสลาย Noise
    df, clean_stats = utils.clean_review_dataframe(
        df,
        min_text_length=config.MIN_TEXT_LENGTH,
        drop_duplicates=config.DROP_DUPLICATE_TEXT,
        duplicate_keep=config.DUPLICATE_KEEP,
    )
    utils.log_cleaning_stats(clean_stats, label="train_reduce") # แสดงสถิติการล้างขยะในหน้า Terminal

    # 3. แยกชุดข้อมูลออกเป็นสองส่วนหลัก: ชุดขุมพลังในการฝึกเทรน/Sweep (80%) และชุดประเมินตรวจสอบภายนอก Holdout (20%)
    df_pool, df_holdout = train_test_split(
        df,
        test_size=config.HOLDOUT_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"], # ประกันสัดส่วนระดับดวงดาวให้มีความเที่ยงตรงสม่ำเสมอในทุกส่วน
    )
    # ซอยแบ่งชุด Pool 80% ด้านในออกเป็นชุดฝึกสอนย่อย (Train Split 64%) และชุดทดสอบความคืบหน้า (Validation Split 16%)
    df_train, df_val = train_test_split(
        df_pool,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df_pool["user_rating"],
    )
    save_holdout_csv(df_holdout) # เซฟตารางทดสอบปลายทางออกเครื่อง

    test_ratings = None
    # ตรวจสอบเพื่อดึงชุดข้อมูลทดสอบภายนอกทางการมาร่วมส่องดูความต่าง EDA
    if os.path.isfile(config.TEST_PATH):
        df_test = utils.load_and_standardize_data(config.TEST_PATH)
        df_test, _ = utils.clean_review_dataframe(
            df_test,
            min_text_length=config.MIN_TEXT_LENGTH,
            drop_duplicates=config.DROP_DUPLICATE_TEXT,
            duplicate_keep=config.DUPLICATE_KEEP,
        )
        test_ratings = df_test["user_rating"].values

    # รันการวิเคราะห์พิกัด EDA ความแตกต่างเปอร์เซ็นต์ดาวชุดสอนเปรียบชุดทดสอบเพื่อป้องกัน Data Drift
    utils.compare_rating_distributions(
        df_train["user_rating"].values,
        test_ratings,
        label_train="train split (after clean)",
        label_test="HF test (after clean)",
    )

    # 4-7. ประยุกต์ใช้นโยบายเสริมความแข็งแกร่งชุดข้อมูลทั้งหมด (Mock Mix, Undersample, Oversample, Augmentation, Truncation)
    train_df = utils.prepare_baseline_train_df(df_train, verbose=True)
    df_val = utils.apply_text_truncation(df_val, config.MAX_REVIEW_CHARS)

    print("--- Step 2: TF-IDF + features ---")
    # 8. สกัดแปลงคุณลักษณะชุดฝึกและชุด Holdout ออกเป็นก้อนเวกเตอร์ฟีเจอร์เด่น
    vectorizer, char_vectorizer, svd, X_train, X_val, explained = (
        fit_vectorizer_and_features(train_df, df_val)
    )
    print(
        f"Feature shape (train): {X_train.shape} | (val): {X_val.shape}"
    )
    if explained is not None:
        print(f"LSA explained variance ratio (sum): {explained:.6f}")

    print("--- Step 3: Training Models (XGBoost, Logistic, LinearSVM, RandomForest) ---")
    
    y_train = _training_labels(train_df["user_rating"].values)
    y_val_true = _training_labels(df_val["user_rating"].values)
    y_val_stars = df_val["user_rating"].values.astype(int)
    
    # 9. เตรียม DMatrix สำหรับ XGBoost
    dtrain, dval = make_dmatrices(X_train, X_val, train_df, df_val)
    xgb_params = dict(config.XGB_PARAMS)
    
    if config.BASELINE_USE_REGRESSION:
        xgb_params["objective"] = "reg:squarederror"
        xgb_params.pop("num_class", None)
        xgb_params["eval_metric"] = "rmse"
    elif config.BASELINE_USE_3CLASS:
        xgb_params["num_class"] = 3
        
    print("\n[1/4] Training XGBoost...")
    bst_xgb, evals_result, best_val = train_xgb_native(dtrain, dval, xgb_params)
    y_val_xgb_prob = bst_xgb.predict(dval)
    if config.BASELINE_USE_REGRESSION:
        y_val_xgb_pred = np.clip(np.round(y_val_xgb_prob), 0, 4)
    else:
        y_val_xgb_pred = np.argmax(y_val_xgb_prob, axis=1) if y_val_xgb_prob.ndim > 1 else np.round(y_val_xgb_prob)
    f1_xgb = f1_score(y_val_true, y_val_xgb_pred, average="macro")
    mae_xgb = xgb_booster_val_mae(
        bst_xgb, dval, y_val_stars, use_regression=config.BASELINE_USE_REGRESSION
    )
    print(f"XGBoost F1-Macro: {f1_xgb:.4f} | Val MAE: {mae_xgb:.4f}")
    
    # 10. เตรียม Weights สำหรับ Sklearn Models
    sample_weights = None
    if config.XGB_USE_SAMPLE_WEIGHT and not config.BASELINE_USE_REGRESSION:
        class_w = utils.compute_class_weights(train_df["user_rating"].values, low_star_boost=config.XGB_LOW_STAR_BOOST)
        sample_weights = utils.compute_sample_weights_from_ratings(train_df["user_rating"].values, class_w)

    print("\n[2/4] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    lr.fit(X_train, y_train, sample_weight=sample_weights)
    f1_lr = f1_score(y_val_true, lr.predict(X_val), average="macro")
    mae_lr = classifier_val_mae(lr, X_val, y_val_stars)
    print(f"Logistic Regression F1-Macro: {f1_lr:.4f} | Val MAE: {mae_lr:.4f}")
    
    print("\n[3/4] Training Linear SVM...")
    svm = LinearSVC(max_iter=1000, class_weight='balanced', dual=False)
    svm.fit(X_train, y_train, sample_weight=sample_weights)
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated_svm = CalibratedClassifierCV(FrozenEstimator(svm))
        calibrated_svm.fit(X_val, y_val_true)
    except ImportError:
        calibrated_svm = CalibratedClassifierCV(svm, cv=3)
        calibrated_svm.fit(X_train, y_train, sample_weight=sample_weights)
    f1_svm = f1_score(y_val_true, calibrated_svm.predict(X_val), average="macro")
    mae_svm = classifier_val_mae(calibrated_svm, X_val, y_val_stars)
    print(f"Linear SVM F1-Macro: {f1_svm:.4f} | Val MAE: {mae_svm:.4f}")
    
    print("\n[4/4] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=config.RANDOM_STATE)
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    f1_rf = f1_score(y_val_true, rf.predict(X_val), average="macro")
    mae_rf = classifier_val_mae(rf, X_val, y_val_stars)
    print(f"Random Forest F1-Macro: {f1_rf:.4f} | Val MAE: {mae_rf:.4f}")
    
    # 11. เปรียบเทียบและเลือกโมเดลที่ดีที่สุด (MAE ต่ำสุด)
    models_mae = {
        "xgboost": mae_xgb,
        "logistic_regression": mae_lr,
        "linear_svm": mae_svm,
        "random_forest": mae_rf,
    }
    models_f1 = {
        "xgboost": f1_xgb,
        "logistic_regression": f1_lr,
        "linear_svm": f1_svm,
        "random_forest": f1_rf,
    }
    best_model_name = pick_lowest_mae(models_mae)
    best_mae = models_mae[best_model_name]
    best_f1 = models_f1[best_model_name]
    
    print("\n" + "="*50)
    print(f"WINNER: {best_model_name.upper()} (Val MAE: {best_mae:.4f}, F1-Macro: {best_f1:.4f})")
    print("="*50 + "\n")
    
    best_model_obj = None
    if best_model_name == "xgboost":
        best_model_obj = bst_xgb
    elif best_model_name == "logistic_regression":
        best_model_obj = lr
    elif best_model_name == "linear_svm":
        best_model_obj = calibrated_svm
    elif best_model_name == "random_forest":
        best_model_obj = rf

    # 12. จดบันทึกรายงานพารามิเตอร์ของระบบทั้งหมด (Metadata) ลงตัวแปรกองเพื่อบันทึกลงในไดเรกทอรี
    meta = {
        "best_model_type": best_model_name,
        "best_val_mae": best_mae,
        "best_f1_macro": best_f1,
        "preprocess_strategy": BASELINE_PREPROCESS_STRATEGY,
        "tfidf_max_features": config.TFIDF_MAX_FEATURES,
        "use_lsa": config.BASELINE_USE_LSA,
        "lsa_n_components": config.LSA_N_COMPONENTS if config.BASELINE_USE_LSA else None,
        "use_extra_features": config.BASELINE_USE_EXTRA_FEATURES,
        "use_char_tfidf": config.BASELINE_USE_CHAR_TFIDF,
        "use_3class": config.BASELINE_USE_3CLASS,
        "use_regression": config.BASELINE_USE_REGRESSION,
        "mock_mix_fraction": config.BASELINE_MOCK_MIX_FRACTION,
        "undersample_star4_fraction": config.BASELINE_UNDERSAMPLE_STAR4_FRACTION,
        "low_star_boost": config.XGB_LOW_STAR_BOOST,
        "max_review_chars": config.MAX_REVIEW_CHARS,
        "oversample_low_stars": config.BASELINE_OVERSAMPLE_LOW_STARS,
        "oversample_factor": config.BASELINE_OVERSAMPLE_FACTOR,
        "use_sample_weight": config.XGB_USE_SAMPLE_WEIGHT,
        "explained_variance": explained,
    }

    print("--- Step 4: Saving artifacts ---")
    save_artifacts(vectorizer, char_vectorizer, svd, best_model_obj, meta) # บันทึกไฟล์เวตและตัวสกัดทั้งหมดลงเครื่องสำรอง
    print("Done baseline training & evaluation!")


if __name__ == "__main__":
    main() # ปลุกสคริปต์หลักเมื่อทำงานตรงๆ