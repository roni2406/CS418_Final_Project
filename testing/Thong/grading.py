from sklearn.metrics import f1_score

import json

with open("dataset_chinese_merged.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with open("dataset_chinese_detected.json", "r", encoding="utf-8") as file:
    detected_data = json.load(file)


def grade_classification(data, detected_data):
    true_labels = [item["label"] for item in data.values()]
    predicted_labels = [item["label"] for item in detected_data.values()]

    f1 = f1_score(true_labels, predicted_labels, average="micro")
    return f1

print(grade_classification(data, detected_data))
