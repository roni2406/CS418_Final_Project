import json
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class VerseProseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        # Separate texts and labels during initialization
        texts = [item['content'] for item in data.values()]
        labels = [1 if item['label'] == 'verse' else 0 for item in data.values()]
        
        # Tokenize texts during initialization
        encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def evaluate_fold(model, val_dataset, device):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    return {
        'f1': f1_score(all_labels, all_predictions, average='macro'),
        'precision': precision_score(all_labels, all_predictions, average='macro'),
        'recall': recall_score(all_labels, all_predictions, average='macro')
    }

def test_model(model, tokenizer, test_data_path, device, fold_num, batch_size=16, max_length=128):
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_texts = [item['content'] for item in test_data.values()]
    
    test_encodings = tokenizer(
        test_texts, 
        truncation=True, 
        padding=True, 
        max_length=max_length, 
        return_tensors='pt'
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        test_encodings['input_ids'], 
        test_encodings['attention_mask']
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            torch.cuda.empty_cache()
    
    label_mapping = {0: "prose", 1: "verse"}
    test_predictions = {str(idx): label_mapping.get(pred, "unknown") for idx, pred in enumerate(all_predictions)}
    
    # Save results for this fold
    with open(f'statistics/results_guwen_fold_{fold_num + 1}.json', 'w', encoding='utf-8') as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=4)
    
    return test_predictions

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = load_data('datasets/dataset_train.json')
    
    # Setup k-fold cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Convert data dictionary to list for easier splitting
    data_items = list(data.items())
    
    # Store metrics for each fold
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_items)):
        print(f"\nTraining fold {fold + 1}/{k_folds}")
        
        # Create train and validation data for this fold
        train_data = dict([data_items[i] for i in train_idx])
        val_data = dict([data_items[i] for i in val_idx])
        
        # Load model and tokenizer for this fold
        model_name = "ethanyt/guwenbert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        
        # Create datasets
        train_dataset = VerseProseDataset(train_data, tokenizer, max_length=128)
        val_dataset = VerseProseDataset(val_data, tokenizer, max_length=128)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f'./checkpoints/fold-{fold + 1}',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            save_strategy='epoch',
            save_total_limit=1,
            logging_dir=f'./logs/fold-{fold + 1}',
            logging_steps=100,
        )
        
        # # Initialize and train
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        # )
        
        # trainer.train()
        model = AutoModelForSequenceClassification.from_pretrained(f'model_guwen_kfold/verse_prose_model_fold_{fold + 1}')
        tokenizer = AutoTokenizer.from_pretrained(f'model_guwen_kfold/verse_prose_model_fold_{fold + 1}')
        model.to(device)

        # Evaluate the fold
        metrics = evaluate_fold(model, val_dataset, device)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold + 1} metrics:")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # # Save the model for this fold
        # model.save_pretrained(f'model_guwen_kfold/verse_prose_model_fold_{fold + 1}')
        # tokenizer.save_pretrained(f'model_guwen_kfold/verse_prose_model_fold_{fold + 1}')

        # Test the model on test dataset
        print(f"Testing fold {fold + 1} model on test dataset...")
        test_model(model, tokenizer, 'datasets/dataset_test.json', device, fold)
    
    # Calculate and save average metrics
    avg_metrics = {
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics])
    }
    
    print("\nAverage metrics across all folds:")
    print(f"F1: {avg_metrics['f1']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    
    # Save metrics to file
    with open('statistics/guwen_kfold_metrics.json', 'w') as f:
        json.dump({
            'fold_metrics': fold_metrics,
            'average_metrics': avg_metrics
        }, f, indent=4)

if __name__ == "__main__":
    main()