import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the answers.json
with open('datasets/dataset_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load the results.json
with open('statistics/rule_based_result_non_punctuation.json', 'r', encoding='utf-8') as f:
    detected_data = json.load(f)

# Extract true and predicted labels
expected_labels = []
detected_labels = []
for i in range(0, len(data)):
    expected_labels.append(data[str(i)]["label"])
    detected_labels.append(detected_data[str(i)]["label"])

# Compute metrics
precision = precision_score(expected_labels, detected_labels, average='macro')
recall = recall_score(expected_labels, detected_labels, average='macro')
f1 = f1_score(expected_labels, detected_labels, average='macro')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')