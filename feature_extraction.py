import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load data
df = pd.read_csv('amazon_dataset.csv')

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def cleanup_backup_files():
    """백업 파일들 삭제"""
    backup_files = glob.glob('backup_batch_*.csv')
    for file in backup_files:
        try:
            os.remove(file)
            print(f"Deleted backup file: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
    if backup_files:
        print(f"Cleaned up {len(backup_files)} backup files")
    else:
        print("No backup files to clean up")

def analyze_image_batch(image_urls, prompt, max_tokens=32):
    """배치로 여러 이미지 동시 처리"""
    if not image_urls:
        return []
    
    try:
        # 여러 이미지에 대한 메시지 생성
        messages_batch = []
        for url in image_urls:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": url},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)
        
        results = []
        for messages in messages_batch:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.append(output_text[0].strip())
        
        return results
    except Exception as e:
        print(f"Batch processing error: {e}")
        return ["Error"] * len(image_urls)

def process_single_row(row_data):
    """단일 행의 모든 피처를 한번에 처리"""
    idx, images_str = row_data
    
    if pd.isna(images_str) or images_str == "":
        return idx, [""] * 7, "0"  # 7개 VL 피처 + num_images
    
    image_urls = [url.strip() for url in images_str.split(',') if url.strip()]
    num_images = str(len(image_urls))
    
    # 모든 프롬프트를 한번에 처리
    feature_prompts = [
        ("Describe this image. Answer in one sentence.", 32),  # 첫 번째 이미지만
        ("Is there a person in the image? Answer Yes or No", 5),
        ("Is a human face visible? Answer Yes or No", 5),
        ("Does this image contain any visible text (e.g., signs or labels)? Answer Yes or No", 5),
        ("Identify the main environment of this image. Answer in one vocabulary (e.g., 'unsure', 'indoor', 'outdoor', 'beach').", 8),
        ("How many products are visible in this image? Answer only number", 8),
        ("What is the dominant color in this image? Answer only color", 8)
    ]
    
    all_results = []
    for i, (prompt, max_tokens) in enumerate(feature_prompts):
        if i == 0:  # captioning - 첫 번째 이미지만
            if image_urls:
                results = analyze_image_batch([image_urls[0]], prompt, max_tokens)
                all_results.append(results[0])
            else:
                all_results.append("")
        else:  # 나머지 피처들 - 모든 이미지
            results = analyze_image_batch(image_urls, prompt, max_tokens)
            all_results.append(", ".join(results))
    
    return idx, all_results, num_images

def save_backup(df, batch_num):
    """백업 저장 (최근 백업만 유지)"""
    # 이전 백업 파일 삭제
    cleanup_backup_files()
    
    # 새 백업 저장
    backup_filename = f'backup_batch_{batch_num}.csv'
    df.to_csv(backup_filename, index=False)
    print(f"Backup saved: {backup_filename} (previous backups cleaned)")

def load_existing_progress():
    """기존 진행상황 로드"""
    if os.path.exists('amazon_dataset_optimized_full.csv'):
        print("Found existing progress file. Loading...")
        return pd.read_csv('amazon_dataset_optimized_full.csv')
    return None

def find_resume_point(df):
    """처리되지 않은 첫 번째 행 찾기"""
    for i in range(len(df)):
        # captioning이 비어있거나 NaN이거나 공백만 있는 경우
        captioning_val = df.iloc[i]['captioning']
        if pd.isna(captioning_val) or str(captioning_val).strip() == "":
            return i
    return len(df)  # 모든 행이 처리된 경우

# 시작 전 기존 백업 파일들 모두 정리
print("Cleaning up existing backup files...")
cleanup_backup_files()

# 기존 진행상황 확인
existing_df = load_existing_progress()
if existing_df is not None:
    sample_df = existing_df.copy()
    print(f"Loaded existing data with {len(sample_df)} rows")
    
    # 어느 행부터 시작할지 확인 (더 정확한 방법)
    start_idx = find_resume_point(sample_df)
    
    # 현재 진행 상황 확인
    completed_rows = 0
    for i in range(len(sample_df)):
        captioning_val = sample_df.iloc[i]['captioning']
        if not pd.isna(captioning_val) and str(captioning_val).strip() != "":
            completed_rows += 1
    
    print(f"Current progress: {completed_rows}/{len(sample_df)} rows completed ({completed_rows/len(sample_df)*100:.1f}%)")
    print(f"Resuming from row {start_idx}")
    
    if start_idx >= len(sample_df):
        print("All rows are already processed!")
else:
    # 전체 데이터셋 처리
    print("Processing full dataset...")
    sample_df = df.copy()
    start_idx = 0

# 새로운 칼럼들 초기화
feature_names = ['captioning', 'is_person', 'face_attribute', 'is_text', 'scene', 'object_count', 'dominant_color', 'num_images']
for feature in feature_names:
    if feature not in sample_df.columns:
        sample_df[feature] = ""

print(f"Starting optimized processing for {len(sample_df)} rows...")
start_time = time.time()

# 행별로 순차 처리 (GPU 메모리 관리)
for i in tqdm(range(start_idx, len(sample_df)), desc="Processing rows"):
    # 이미 처리된 행은 스킵 (더 정확한 체크)
    captioning_val = sample_df.iloc[i]['captioning']
    if not pd.isna(captioning_val) and str(captioning_val).strip() != "":
        continue
        
    row_data = (i, sample_df.iloc[i]['images'])
    idx, vl_results, num_images = process_single_row(row_data)
    
    # 결과 저장
    sample_df.iloc[i, sample_df.columns.get_loc('captioning')] = vl_results[0]
    sample_df.iloc[i, sample_df.columns.get_loc('is_person')] = vl_results[1]
    sample_df.iloc[i, sample_df.columns.get_loc('face_attribute')] = vl_results[2]
    sample_df.iloc[i, sample_df.columns.get_loc('is_text')] = vl_results[3]
    sample_df.iloc[i, sample_df.columns.get_loc('scene')] = vl_results[4]
    sample_df.iloc[i, sample_df.columns.get_loc('object_count')] = vl_results[5]
    sample_df.iloc[i, sample_df.columns.get_loc('dominant_color')] = vl_results[6]
    sample_df.iloc[i, sample_df.columns.get_loc('num_images')] = num_images
    
    # 매 100개마다 백업 저장 (이전 백업은 자동 삭제됨)
    if (i + 1) % 100 == 0:
        sample_df.to_csv('amazon_dataset_optimized_full.csv', index=False)
        save_backup(sample_df, (i + 1) // 100)
        print(f"Progress saved at row {i + 1}")
        
        # 현재까지 진행률 출력 (더 정확한 계산)
        completed = 0
        for j in range(len(sample_df)):
            captioning_val = sample_df.iloc[j]['captioning']
            if not pd.isna(captioning_val) and str(captioning_val).strip() != "":
                completed += 1
        print(f"Completed: {completed}/{len(sample_df)} rows ({completed/len(sample_df)*100:.1f}%)")

end_time = time.time()
processing_time = end_time - start_time

print(f"\n=== Full Dataset Processing Completed! ===")
print(f"Processing time: {processing_time:.2f} seconds ({processing_time/3600:.1f} hours)")
print(f"Average time per row: {processing_time/(len(sample_df)-start_idx):.2f} seconds")

# 마지막 백업 파일도 삭제
print("\nCleaning up final backup files...")
cleanup_backup_files()

# 최종 저장
sample_df.to_csv('amazon_dataset_optimized_FULL_FINAL.csv', index=False)
print("Results saved to 'amazon_dataset_optimized_FULL_FINAL.csv'")

# 결과 요약
print("\n=== Summary ===")
for feature in feature_names:
    completed_count = 0
    for j in range(len(sample_df)):
        feature_val = sample_df.iloc[j][feature]
        if not pd.isna(feature_val) and str(feature_val).strip() != "":
            completed_count += 1
    print(f"{feature}: {completed_count}/{len(sample_df)} rows completed")

# 샘플 결과 출력
print("\n=== Sample Results (First Row) ===")
for feature in feature_names:
    print(f"{feature}: {sample_df[feature].iloc[0]}")

print("\n=== Optimized Features ===")
print("1. Resume capability - automatically continues from last checkpoint")
print("2. Backup management - only keeps latest backup, auto-cleanup")
print("3. Optimized processing - all features per row in single pass") 
print("4. Captioning only for first image per row")
print("5. Automatic cleanup of all backup files during processing")
print("6. Memory efficient - no accumulation of backup files")