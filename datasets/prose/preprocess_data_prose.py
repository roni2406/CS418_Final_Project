import json
import random

def split_and_label(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    
    data = {}
    index = 0
    current_item = ""
    punctuation_mark = '.'

    for sentence in sentences:
        sentence = sentence.strip().replace(' ', '')
        if not sentence:
            continue
        if not sentence.endswith(punctuation_mark):
            sentence += punctuation_mark
        if current_item:
            current_item += sentence
        else:
            current_item = sentence
        # Calculate length without spaces and punctuation
        clean_length = len(current_item.replace(punctuation_mark, ''))
        if clean_length > 50:
            data[str(index)] = {
                "content": current_item,
                "label": "prose"
            }
            index += 1
            current_item = ""
            
    # Add any remaining text
    if current_item:
        data[str(index)] = {
            "content": current_item,
            "label": "prose"
        }
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

def split_train_test(train_ratio=0.8, seed=42,
                    data_path='datasets/prose/thuyhu_labeled_data_prose.json',
                    train_path='datasets/prose/thuyhu_data_prose_train.json',
                    test_path='datasets/prose/thuyhu_data_prose_test.json'):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)

    split_index = int(train_ratio * len(keys))
    train_keys = keys[:split_index]
    test_keys = keys[split_index:]

    train_data = {key: data[key] for key in train_keys}
    test_data = {key: data[key] for key in test_keys}

    # Save data
    with open(train_path, 'w', encoding='utf-8') as outfile:
        json.dump(train_data, outfile, ensure_ascii=False, indent=2)
    with open(test_path, 'w', encoding='utf-8') as outfile:
        json.dump(test_data, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    split_and_label('datasets/prose/thuyhu.txt', 'datasets/prose/thuyhu_labeled_data_prose.json')
    split_train_test()
    split_and_label('datasets/prose/DVSKTT_self_aligned_cn_sv_vi_20240927_cleaned_official.no_punct.line.train.txt', 'datasets/prose/DVSKTT_labeled_data_prose.json')
    split_train_test(data_path='datasets/prose/DVSKTT_labeled_data_prose.json',
                     train_path='datasets/prose/DVSKTT_data_prose_train.json',
                     test_path='datasets/prose/DVSKTT_data_prose_test.json')
    split_and_label('datasets/prose/tamquoc.txt', 'datasets/prose/tamquoc_labeled_data_prose.json')
    split_train_test(data_path='datasets/prose/tamquoc_labeled_data_prose.json',
                     train_path='datasets/prose/tamquoc_data_prose_train.json',
                     test_path='datasets/prose/tamquoc_data_prose_test.json')