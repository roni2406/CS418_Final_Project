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

def split_train_test(train_ratio=0.8, val_ratio = 0.1, test_ratio = 0.1, seed=42,
                    data_path='datasets/prose/thuyhu_labeled_data_prose.json',
                    train_path='datasets/prose/val_thuyhu_data_prose_train.json',
                    test_path='datasets/prose/val_thuyhu_data_prose_test.json',
                    val_path='datasets/prose/val_thuyhu_data_prose_val.json',):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

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

if __name__ == "__main__":
    split_and_label('datasets/prose/thuyhu.txt', 'datasets/prose/thuyhu_labeled_data_prose.json')
    split_train_test()
    split_and_label('datasets/prose/DVSKTT_self_aligned_cn_sv_vi_20240927_cleaned_official.no_punct.line.train.txt', 'datasets/prose/DVSKTT_labeled_data_prose.json')
    split_train_test(data_path='datasets/prose/DVSKTT_labeled_data_prose.json',
                     train_path='datasets/prose/val_DVSKTT_data_prose_train.json',
                     test_path='datasets/prose/val_DVSKTT_data_prose_test.json',
                     val_path='datasets/prose/val_DVSKTT_data_prose_val.json')
    split_and_label('datasets/prose/tamquoc.txt', 'datasets/prose/tamquoc_labeled_data_prose.json')
    split_train_test(data_path='datasets/prose/tamquoc_labeled_data_prose.json',
                     train_path='datasets/prose/val_tamquoc_data_prose_train.json',
                     test_path='datasets/prose/val_tamquoc_data_prose_test.json',
                     val_path='datasets/prose/val_tamquoc_data_prose_val.json')