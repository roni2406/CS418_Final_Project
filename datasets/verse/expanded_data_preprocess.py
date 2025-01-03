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
    data = json.load(file)

## split data to train and test with ratio 0.8
def split_train_test(train_ratio=0.8, seed=42,
                     train_path='datasets/verse/kieu_train.json',
                     test_path='datasets/verse/kieu_test.json'):
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