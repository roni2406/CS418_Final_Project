import json
import random

import re

def remove_punctuation_and_spaces(text):
    # Define a regex pattern for punctuation (both ASCII and Chinese)
    pattern = r'[，。！？、；：“”‘’.,!?;:"\'`~@#$%^&*()\-_=+[\]{}|\\<>/。]'
    # Remove punctuation
    text = re.sub(pattern, '', text)
    # Remove spaces
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    return text

def process_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    for key, value in data.items():
        original_content = value.get('content', '')
        cleaned_content = remove_punctuation_and_spaces(original_content)
        # data[key]['content'] = cleaned_content
        # For rule-based model, we don't need to clean the content
        data[key]['content'] = original_content
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

def process_dataset(split):
    random.seed(42)
    
    verse_file = f'datasets/verse/expanded_data_{split}.json'
    prose_file = f'datasets/prose/DVSKTT_data_prose_{split}.json'
    output_file = f'datasets/rule_based_new_dataset_{split}.json'
    verse_file_2 = f'datasets/verse/kieu_{split}.json'
    prose_file_2 = f'datasets/prose/thuyhu_data_prose_{split}.json'
    verse_file_3 = f'datasets/verse/thivien_{split}.json'
    prose_file_3 = f'datasets/prose/tamquoc_data_prose_{split}.json'

    
    with open(verse_file, 'r', encoding='utf-8') as f:
        verse_data = json.load(f)
    
    with open(prose_file, 'r', encoding='utf-8') as f:
        prose_data = json.load(f)
    
    with open(verse_file_2, 'r', encoding='utf-8') as f:
        verse_data_2 = json.load(f)
    
    with open(prose_file_2, 'r', encoding='utf-8') as f:
        prose_data_2 = json.load(f)
    
    with open(verse_file_3, 'r', encoding='utf-8') as f:
        verse_data_3 = json.load(f)
    
    with open(prose_file_3, 'r', encoding='utf-8') as f:
        prose_data_3 = json.load(f)

    # Convert data to lists if they are dictionaries
    verse_data = list(verse_data.values()) if isinstance(verse_data, dict) else verse_data
    prose_data = list(prose_data.values()) if isinstance(prose_data, dict) else prose_data
    verse_data_2 = list(verse_data_2.values()) if isinstance(verse_data_2, dict) else verse_data_2
    prose_data_2 = list(prose_data_2.values()) if isinstance(prose_data_2, dict) else prose_data_2
    verse_data_3 = list(verse_data_3.values()) if isinstance(verse_data_3, dict) else verse_data_3
    prose_data_3 = list(prose_data_3.values()) if isinstance(prose_data_3, dict) else prose_data_3
    
    merged_data = verse_data + prose_data + verse_data_2 + prose_data_2 + verse_data_3 + prose_data_3
    
    random.shuffle(merged_data)
    
    shuffled_data = {index: value for index, value in enumerate(merged_data)}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shuffled_data, f, ensure_ascii=False, indent=4)
    
    process_json(output_file, output_file)


for split in ['train', 'test']:
    process_dataset(split)