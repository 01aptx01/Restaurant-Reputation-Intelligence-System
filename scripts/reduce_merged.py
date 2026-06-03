import pandas as pd
import os

def reduce_dataset(input_file, output_file, target_rows=10000):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    total_rows = len(df)
    print(f"Original dataset has {total_rows} rows.")
    
    if total_rows <= target_rows:
        print("Dataset is already smaller than or equal to target rows. No reduction needed.")
        df.to_csv(output_file, index=False)
        return
        
    # หาอัตราส่วนที่ต้องสุ่ม
    frac = target_rows / total_rows
    
    # สุ่มข้อมูลแบบ Stratified Sampling เพื่อรักษาสัดส่วนของ Rating
    print(f"Reducing to approximately {target_rows} rows (stratified by Rating)...")
    
    # ใช้ groupby และ sample เพื่อสุ่มตามสัดส่วน
    # ถ้าหากไม่มีคอลัมน์ Rating จะใช้การสุ่มแบบปกติแทน
    if 'Rating' in df.columns:
        df_reduced = df.groupby('Rating', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
    elif 'user_rating' in df.columns:
        df_reduced = df.groupby('user_rating', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
    else:
        df_reduced = df.sample(n=target_rows, random_state=42)
        
    print(f"Reduced dataset has {len(df_reduced)} rows.")
    
    # ตรวจสอบสัดส่วน
    if 'Rating' in df.columns:
        orig_dist = df['Rating'].value_counts(normalize=True).sort_index() * 100
        new_dist = df_reduced['Rating'].value_counts(normalize=True).sort_index() * 100
        print("\nDistribution comparison (%):")
        dist_df = pd.DataFrame({'Original': orig_dist, 'Reduced': new_dist})
        print(dist_df)
    
    # บันทึกไฟล์
    print(f"\nSaving to {output_file}...")
    df_reduced.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    input_path = "data/merge/merged_restaurant_reviews.csv"
    output_path = "data/merge/merged_reduce2.csv"
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    reduce_dataset(input_path, output_path, target_rows=7000)
