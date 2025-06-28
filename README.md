# ecommerce
'''
feature_prompts = {
    'captioning': ("Describe this image. Answer in one sentence.", 32),
    'is_person': ("Is there a person in the image? Answer Yes or No", 5),
    'face_attribute': ("Is a human face visible? Answer Yes or No", 5),
    'is_text': ("Does this image contain any visible text (e.g., signs or labels)? Answer Yes or No", 5),
    'scene': ("Identify the main environment of this image. Answer in one vocabulary (e.g., 'unsure', 'indoor', 'outdoor', 'beach').", 8),
    'object_count': ("How many products are visible in this image? Answer only number", 8),
    'dominant_color': ("What is the dominant color in this image? Answer only color", 8)
}
'''

1. 캡션 생성
2. 사람 있는가?
3. 사람 얼굴 있는가?
4. 텍스트 있는가?
5. 배경 장면 어디?
6. 개수
7. 메인 컬러

* 목표: VLM을 사용해 이미지에서 feature들을 뽑아내서 average_rating prediction의 accuracy를 펌핑하자!