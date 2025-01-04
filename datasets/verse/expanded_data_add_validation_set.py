# # convert the expanded_data.xlsx to expanded_data.txt
# import pandas as pd

# # Read the Excel file
# df = pd.read_excel('datasets/verse/expanded_data.xlsx')

# # Extract the first column
# entries = df.iloc[:, 0]

# # Write to a text file
# entries.to_csv('datasets/verse/expanded_data.txt', index=False, header=False)

# import json

# ## label data
# # Define Chinese punctuation marks
# punctuations = {'，', '。', '！', '？', '、', '：', '；', '“', '”', '‘', '’', '（', '）'}

# data = {}
# index = 0
# current_verse = []

# with open('datasets/verse/expanded_data.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         line = line.strip()
#         if not line:
#             continue
#         if line[-1] in punctuations:
#             current_verse.append(line)
#         else:
#             # If current_verse has collected lines, save it
#             if current_verse:
#                 content = ''.join(current_verse)
#                 data[str(index)] = {
#                     "content": content,
#                     "label": "verse"
#                 }
#                 index += 1
#                 current_verse = []
#             # Skip the title line
#             continue
#     # Add the last verse if exists
#     if current_verse:
#         content = ''.join(current_verse)
#         data[str(index)] = {
#             "content": content,
#             "label": "verse"
#         }

# # Split the data into train and test sets
# import random
# with open('datasets/verse/kieu_labeled.json', 'w', encoding='utf-8') as json_file:
#     json.dump(data, json_file, ensure_ascii=False, indent=4)

import json
import random

with open('datasets/verse/kieu_labeled.json', 'r', encoding='utf-8') as file:
    kieu = json.load(file)

with open('datasets/verse/thivien_labeled.json', 'r', encoding='utf-8') as file:
    thivien = json.load(file)

with open('datasets/verse/expanded_data_labeled.json', 'r', encoding='utf-8') as file:
    expanded = json.load(file)

def split_data(data = kieu, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
               train_path='datasets/verse/val_kieu_train.json',
               val_path='datasets/verse/val_kieu_val.json',
               test_path='datasets/verse/val_kieu_test.json'):
    # Verify that ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"
    
    # Get and shuffle keys
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)
    
    # Calculate split indices
    train_end = int(train_ratio * len(keys))
    val_end = train_end + int(val_ratio * len(keys))
    
    # Split keys
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]
    
    # Create data splits
    train_data = {key: data[key] for key in train_keys}
    val_data = {key: data[key] for key in val_keys}
    test_data = {key: data[key] for key in test_keys}
    
    # Save splits to files
    splits = [
        (train_data, train_path),
        (val_data, val_path),
        (test_data, test_path)
    ]
    
    for split_data, path in splits:
        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(split_data, outfile, ensure_ascii=False, indent=4)
    
    # Print split sizes
    print(f"Total samples: {len(keys)}")
    print(f"Train samples: {len(train_keys)} ({len(train_keys)/len(keys):.1%})")
    print(f"Validation samples: {len(val_keys)} ({len(val_keys)/len(keys):.1%})")
    print(f"Test samples: {len(test_keys)} ({len(test_keys)/len(keys):.1%})")

split_data()
split_data(thivien, train_path='datasets/verse/val_thivien_train.json',
           val_path='datasets/verse/val_thivien_val.json',
           test_path='datasets/verse/val_thivien_test.json')
split_data(expanded, train_path='datasets/verse/val_expanded_data_train.json',
              val_path='datasets/verse/val_expanded_data_val.json',
              test_path='datasets/verse/val_expanded_data_test.json')