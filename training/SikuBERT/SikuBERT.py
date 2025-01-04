import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader



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

def test_model(model, tokenizer, test_data_path, device, batch_size=16, max_length=128):
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
    
    with open('statistics/results_siku.json', 'w', encoding='utf-8') as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=4)


def main():
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "SIKU-BERT/sikubert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Load and prepare data
    data = load_data('datasets/dataset_train.json')
    dataset = VerseProseDataset(data, tokenizer, max_length=128)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_strategy='epoch',
        save_total_limit=2,  # Keep only the last 2 checkpoints
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()
    
    # Save the final model and tokenizer
    model.save_pretrained('model_siku/verse_prose_model')
    tokenizer.save_pretrained('model_siku/verse_prose_model')
    
    print("Model and tokenizer saved to 'model_siku/verse_prose_model'")
    
    # Test the model
    test_model(model, tokenizer, 'datasets/dataset_test.json', device=device, batch_size=16, max_length=128)

if __name__ == "__main__":
    main()