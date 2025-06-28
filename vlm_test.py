import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load data
df = pd.read_csv('amazon_dataset.csv')

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def analyze_image(image_url, prompt, max_tokens=32):
    """단일 이미지에 대해 VL 모델 분석 수행"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

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
        return output_text[0].strip()
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return "Error"

def process_images_for_feature(images_str, prompt, max_tokens=32):
    """콤마로 구분된 이미지들에 대해 특정 피처 분석"""
    if pd.isna(images_str) or images_str == "":
        return ""
    
    image_urls = [url.strip() for url in images_str.split(',') if url.strip()]
    results = []
    
    for url in image_urls:
        result = analyze_image(url, prompt, max_tokens)
        results.append(result)
    
    return ", ".join(results)

def count_images(images_str):
    """이미지 개수 세기 (알고리즘 처리)"""
    if pd.isna(images_str) or images_str == "":
        return "0"
    
    image_urls = [url.strip() for url in images_str.split(',') if url.strip()]
    return str(len(image_urls))

# 처리할 데이터 (첫 10개)
sample_df = df.iloc[0:10].copy()

print("Starting feature extraction for 10 samples...")

# 각 피처별 프롬프트 정의
feature_prompts = {
    'captioning': ("Describe this image. Answer in one sentence.", 32),
    'is_person': ("Is there a person in the image? Answer Yes or No", 5),
    'face_attribute': ("Is a human face visible? Answer Yes or No", 5),
    'is_text': ("Does this image contain any visible text (e.g., signs or labels)? Answer Yes or No", 5),
    'scene': ("Identify the main environment of this image. Answer in one vocabulary (e.g., 'unsure', 'indoor', 'outdoor', 'beach').", 8),
    'object_count': ("How many products are visible in this image? Answer only number", 8),
    'dominant_color': ("What is the dominant color in this image? Answer only color", 8)
}

# 각 피처 처리
for feature_name, (prompt, max_tokens) in feature_prompts.items():
    print(f"Processing {feature_name}...")
    
    feature_results = []
    for idx in range(len(sample_df)):
        images_str = sample_df.iloc[idx]['images']
        result = process_images_for_feature(images_str, prompt, max_tokens)
        feature_results.append(result)
        print(f"  Row {idx}: {result}")
    
    sample_df[feature_name] = feature_results

# num_images 처리 (알고리즘)
print("Processing num_images...")
num_images_results = []
for idx in range(len(sample_df)):
    images_str = sample_df.iloc[idx]['images']
    result = count_images(images_str)
    num_images_results.append(result)
    print(f"  Row {idx}: {result}")

sample_df['num_images'] = num_images_results

print("\nFeature extraction completed!")
print("New columns added:", ['captioning', 'is_person', 'num_images', 'face_attribute', 'is_text', 'scene', 'object_count', 'dominant_color'])

# 결과 저장
sample_df.to_csv('amazon_dataset_with_features_first10.csv', index=False)
print("Results saved to 'amazon_dataset_with_features_first10.csv'")

# 결과 미리보기
print("\nSample results:")
for col in ['captioning', 'is_person', 'num_images', 'face_attribute', 'is_text', 'scene', 'object_count', 'dominant_color']:
    print(f"\n{col}:")
    print(sample_df[col].iloc[0])  # 첫 번째 행만 출력