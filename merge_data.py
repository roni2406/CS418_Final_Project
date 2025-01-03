import json
import random

def process_dataset(split):
    random.seed(42)
    
    verse_file = f'datasets/dataset_chinese_{split}.json'
    prose_file = f'Kieu.json'
    output_file = f'datasets/dataset_chinese_nom.json'
    
    with open(verse_file, 'r', encoding='utf-8') as f:
        verse_data = json.load(f)
    
    with open(prose_file, 'r', encoding='utf-8') as f:
        prose_data = json.load(f)
    
    merged_data = {**verse_data, **prose_data}
    
    items = list(merged_data.items())
    random.shuffle(items)
    
    shuffled_data = {index: value for index, (_, value) in enumerate(items)}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shuffled_data, f, ensure_ascii=False, indent=4)

for split in ['train']:
    process_dataset(split)