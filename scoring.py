import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the answers.json
with open('datasets/dataset_test.json', 'r', encoding='utf-8') as f:
    answers = json.load(f)

# Load the results.json
with open('statistics/results_siku_fold_1.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extract true and predicted labels
true_labels = [item['label'] for item in answers.values()]
predicted_labels = [results[key] for key in sorted(results.keys(), key=int)]

# Compute metrics
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')