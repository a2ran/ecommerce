from datasets import load_dataset

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
print(dataset[0])

import pandas as pd

filtered_data = []

for item in dataset:
    images_list = item.get('images', {}).get('large', [])
    images_list_cleaned = [str(x) for x in images_list if x is not None]
    
    filtered_item = {
        'title': item.get('title', ''),
        'average_rating': item.get('average_rating', ''),
        'images': ', '.join(images_list_cleaned),
        'details': item.get('details', '')
    }
    filtered_data.append(filtered_item)

df = pd.DataFrame(filtered_data)

df.to_csv('amazon_dataset.csv', index=False)
