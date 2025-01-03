import json
import random

def label_thivien(input_data):
    labeled_data = {}
    
    # Filter out invalid entries
    valid_thivien = {k: v for k, v in input_data.items() 
                  if k != "Title not found" and v != "Content not found"}
    
    # Create numbered entries
    for idx, (_, content) in enumerate(valid_thivien.items()):
        labeled_data[str(idx)] = {
            "content": content,
            "label": "verse"
        }
    
    return labeled_data

# Example usage:
with open('datasets/verse/thivien.json', 'r', encoding='utf-8') as file:
    input_data = json.load(file)

labeled = label_thivien(input_data)

# Save the labeled data to a new JSON file
with open('datasets/verse/thivien_labeled.json', 'w', encoding='utf-8') as file:
    json.dump(labeled, file, ensure_ascii=False, indent=2)

# Load the labeled data
with open('datasets/verse/thivien_labeled.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

## split data to train and test with ratio 0.8
def split_train_test(train_ratio=0.8, seed=42,
                     train_path='datasets/verse/thivien_train.json',
                     test_path='datasets/verse/thivien_test.json'):
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)

    split_index = int(train_ratio * len(keys))
    train_keys = keys[:split_index]
    test_keys = keys[split_index:]

    train_data = {key: data[key] for key in train_keys}
    test_data = {key: data[key] for key in test_keys}

    with open(train_path, 'w', encoding='utf-8') as outfile:
        json.dump(train_data, outfile, ensure_ascii=False, indent=4)

    with open(test_path, 'w', encoding='utf-8') as outfile:
        json.dump(test_data, outfile, ensure_ascii=False, indent=4)

split_train_test()