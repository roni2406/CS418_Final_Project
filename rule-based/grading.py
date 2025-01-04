import json

with open("new_dataset_test.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with open("new_dataset_detected.json", "r", encoding="utf-8") as file:
    detected_data = json.load(file)


def grade_classification(data, detected_data):
    length = len(data)
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for i in range(0, length):
        expected_label = data[str(i)]["label"]
        detected_label = detected_data[str(i)]["label"]

        if expected_label == "verse" and detected_label == "verse":
            true_pos += 1
        elif expected_label == "prose" and detected_label == "verse":
            false_pos += 1
        elif expected_label == "verse" and detected_label == "prose":
            false_neg += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f1


print(grade_classification(data, detected_data))
