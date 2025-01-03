import json
import random
from collections import defaultdict

with open("dataset_chinese_merged.json", "r", encoding="utf-8") as file:
    data = json.load(file)

detected_data = {}


def classify_verse(text):
    """
    Classifies the given text into one of the specified verse types:
    - Thất ngôn tứ tuyệt
    - Thất ngôn bát cú
    - Lục Bát
    - Song Thất Lục Bát
    - 4/5/6/7/8-character verses
    - Semi-Free Verse
    - Free verse

    Parameters:
        text (str): The input text, with each line representing a verse.

    Returns:
        str: The detected verse type.
    """
    # Split text into lines and calculate character count for each line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    line_lengths = [len(line) for line in lines]

    # Check for Thất ngôn tứ tuyệt (7-character, 4 lines)
    if len(lines) == 4 and all(length == 7 for length in line_lengths):
        return "Thất ngôn tứ tuyệt"

    # Check for Thất ngôn bát cú (7-character, 8 lines)
    if len(lines) == 8 and all(length == 7 for length in line_lengths):
        return "Thất ngôn bát cú"

    # Check for Lục Bát (alternating 6-character and 8-character lines)
    if len(lines) >= 4:
        for i in range(len(lines) - 3):
            segment = line_lengths[i : i + 4]
            if segment == [6, 8, 6, 8]:
                return "Lục Bát"

    # Check for Song Thất Lục Bát (two 7-character lines followed by 6-character and 8-character lines)
    if len(lines) >= 4:
        for i in range(len(lines) - 3):
            segment = line_lengths[i : i + 4]
            if segment == [7, 7, 6, 8]:
                return "Song Thất Lục Bát"

    # Check for Semi-Free Verse
    if len(lines) >= 4:
        from collections import defaultdict

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
        if most_common / total_sequences > 0: # set a reasonable threshold, I find that 0 is the best
            return "Semi-Free Verse"

    # Check for 4/5/6/7/8-character verses
    if all(length in {4, 5, 6, 7, 8} for length in line_lengths):
        return f"{min(line_lengths)}-{max(line_lengths)}-character verse"

    # If none match, classify as free verse
    return "Free verse"


def detect_prose_verse(text):
    cleaned_text = text.replace(" ", "").replace("，", "。").replace("。", "\n")
    if classify_verse(cleaned_text) == "Free verse":
        # return "verse" if random.random() < 0.1 else "prose"
        return "prose"
    return "verse"


for key, value in data.items():
    content = value["content"]
    detected_value = {"content": content, "label": detect_prose_verse(content)}
    detected_data[key] = detected_value

# dump cleaned data to a new file
with open("dataset_chinese_detected.json", "w", encoding="utf-8") as file:
    json.dump(detected_data, file, ensure_ascii=False, indent=4)
