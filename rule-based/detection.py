import json
import random
from collections import defaultdict

def clean_chinese_text(text):
    """
    Better preprocessing for Chinese text that preserves verse structure.
    """
    # Remove spaces while preserving original line breaks
    text = ''.join(text.split())
    
    # Standardize line breaks
    text = text.replace('。。。', '…').replace('...', '…')  # Handle ellipsis
    
    # Only split on actual line breaks and specific end-of-verse punctuation
    text = text.replace('。', '\n').replace('！', '\n').replace('？', '\n')
    
    # Remove other punctuation that shouldn't affect verse structure
    text = text.replace('，', '').replace('、', '').replace('；', '')
    text = text.replace('：', '').replace('「', '').replace('」', '')
    text = text.replace('"', '').replace('"', '').replace('…', '')
    
    # Clean up multiple newlines and empty lines
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)

def classify_verse(text):
    """
    Improved verse classification with better structure detection.
    """
    # Clean and split text
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    line_lengths = [len(line) for line in lines]
    
    if not lines:
        return "prose"  # Empty text is considered prose

    # Minimum length requirement for verse classification
    if len(lines) < 2:
        return "prose"

    # Check for Thất ngôn tứ tuyệt (7-character, 4 lines)
    if len(lines) == 4 and all(length == 7 for length in line_lengths):
        return "Thất ngôn tứ tuyệt"

    # Check for Thất ngôn bát cú (7-character, 8 lines)
    if len(lines) == 8 and all(length == 7 for length in line_lengths):
        return "Thất ngôn bát cú"

    # Check for Lục Bát pattern
    luc_bat_pattern = True
    if len(lines) >= 4 and len(lines) % 2 == 0:
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                if not (line_lengths[i] == 6 and line_lengths[i + 1] == 8):
                    luc_bat_pattern = False
                    break
        if luc_bat_pattern:
            return "Lục Bát"

    # Check for Song Thất Lục Bát
    if len(lines) >= 4 and len(lines) % 4 == 0:
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                segment = line_lengths[i:i + 4]
                if segment == [7, 7, 6, 8]:
                    return "Song Thất Lục Bát"

    # Check for regular character-count verses
    char_counts = set(line_lengths)
    if char_counts.issubset({4, 5, 6, 7, 8}):
        # If more than 80% of lines have the same length, it's a regular verse
        most_common_length = max(set(line_lengths), key=line_lengths.count)
        if line_lengths.count(most_common_length) / len(line_lengths) >= 0.8:
            return f"{most_common_length}-character verse"
        return f"{min(line_lengths)}-{max(line_lengths)}-character verse"

    # Improved Semi-Free Verse detection
    sequence_counts = defaultdict(int)
    current_length = line_lengths[0]
    count = 1

    for length in line_lengths[1:]:
        if length == current_length:
            count += 1
        else:
            sequence_counts[current_length] += count
            current_length = length
            count = 1
    sequence_counts[current_length] += count

    total_sequences = sum(sequence_counts.values())
    most_common = max(sequence_counts.values())
    
    # More stringent criteria for Semi-Free Verse
    if most_common / total_sequences >= 0.4:  # At least 40% consistent pattern
        return "Semi-Free Verse"

    return "Free verse"

def detect_prose_verse(text):
    cleaned_text = clean_chinese_text(text)
    verse_type = classify_verse(cleaned_text)
    
    # Consider as prose if:
    # 1. It's free verse with very long lines
    # 2. It has very few lines
    # 3. Lines are too variable in length
    lines = cleaned_text.split('\n')
    if verse_type == "Free verse":
        if any(len(line) > 20 for line in lines):  # Very long lines suggest prose
            return "prose"
        if len(lines) < 3:  # Too few lines suggest prose
            return "prose"
    
    return "verse" if verse_type != "Free verse" else "prose"

# Main processing
with open("new_dataset_test.json", "r", encoding="utf-8") as file:
    data = json.load(file)

detected_data = {}
for key, value in data.items():
    content = value["content"]
    detected_value = {"content": content, "label": detect_prose_verse(content)}
    detected_data[key] = detected_value

with open("new_dataset_detected.json", "w", encoding="utf-8") as file:
    json.dump(detected_data, file, ensure_ascii=False, indent=4)