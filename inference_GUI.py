import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import re
from typing import List, Tuple, Dict
import math
from collections import defaultdict

# Add the rule-based classifier class
class RuleBasedClassifier:
    @staticmethod
    def clean_chinese_text(text):
        # Remove spaces while preserving original line breaks
        text = ''.join(text.split())
        
        # Standardize line breaks
        text = text.replace('。。。', '…').replace('...', '…')
        
        # Split on actual line breaks and specific end-of-verse punctuation
        text = text.replace('。', '\n').replace('！', '\n').replace('？', '\n')
        
        # Remove other punctuation
        text = text.replace('，', '').replace('、', '').replace('；', '')
        text = text.replace('：', '').replace('「', '').replace('」', '')
        text = text.replace('"', '').replace('"', '').replace('…', '')
        
        # Clean up multiple newlines and empty lines
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)

    @staticmethod
    def classify_verse(text):
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        line_lengths = [len(line) for line in lines]
        
        if not lines or len(lines) < 2:
            return "prose"

        # Check for specific verse patterns
        if len(lines) == 4 and all(length == 7 for length in line_lengths):
            return "verse"  # Thất ngôn tứ tuyệt

        if len(lines) == 8 and all(length == 7 for length in line_lengths):
            return "verse"  # Thất ngôn bát cú

        # Check for regular character-count verses
        char_counts = set(line_lengths)
        if char_counts.issubset({4, 5, 6, 7, 8}):
            most_common_length = max(set(line_lengths), key=line_lengths.count)
            if line_lengths.count(most_common_length) / len(line_lengths) >= 0.8:
                return "verse"

        # Check for prose indicators
        if any(len(line) > 20 for line in lines):
            return "prose"
        
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
        
        return "verse" if most_common / total_sequences >= 0.4 else "prose"

    def predict(self, text):
        cleaned_text = self.clean_chinese_text(text)
        print(text)
        prediction = self.classify_verse(cleaned_text)
        
        # Simulate confidence scores for consistency with the ML model
        confidence = 0.8 if prediction == "verse" else 0.75
        
        return {
            'prediction': prediction,
            'winning_confidence': confidence,
            'average_confidence': confidence,
            'total_chunks': 1,
            'chunk_distribution': {'verse': 1, 'prose': 0} if prediction == 'verse' else {'verse': 0, 'prose': 1},
            'chunk_predictions': [(prediction, confidence)]
        }

class VerseProseMajorityClassifier:
    def __init__(
        self,
        model_path: str,
        max_length: int = 128,
        overlap_percent: float = 0.2,
        min_chunk_size: int = 50,
        device: str = None
    ):
        self.max_length = max_length
        self.overlap_percent = overlap_percent
        self.min_chunk_size = min_chunk_size
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_mapping = {0: "prose", 1: "verse"}

    def clean_text(self, text: str) -> str:
        pattern = r'[，。！？、；：""''.,!?;:"\'`~@#$%^&*()\-_=+[\]{}|\\<>/。]'
        text = re.sub(pattern, '', text)
        text = text.replace(' ', '').replace('\n', '')
        text = text.replace(' ', '')
        text = re.sub(r'\s+', '', text)
        return text

    def split_text(self, text: str) -> List[str]:
        cleaned_text = self.clean_text(text)
        print(cleaned_text)
        if len(cleaned_text) <= self.max_length:
            return [cleaned_text]
        
        chunk_size = self.max_length
        overlap_size = int(chunk_size * self.overlap_percent)
        
        chunks = []
        start = 0

        if len(cleaned_text) <= 128:
            return [cleaned_text]
        
        while start < len(cleaned_text):
            end = start + chunk_size
            chunk = cleaned_text[start:end]
            
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            start = end - overlap_size
            
        return chunks

    def predict_chunk(self, chunk: str) -> Tuple[str, float]:
        inputs = self.tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1)
            confidence = probabilities[0][prediction[0]].item()
        
        predicted_label = self.label_mapping[prediction.item()]
        return predicted_label, confidence

    def majority_vote(self, predictions: List[Tuple[str, float]]) -> Dict:
        weighted_votes = {'prose': 0.0, 'verse': 0.0}
        for label, confidence in predictions:
            weighted_votes[label] += confidence
        
        final_label = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        total_chunks = len(predictions)
        chunk_distribution = {
            'prose': len([p for p, _ in predictions if p == 'prose']),
            'verse': len([p for p, _ in predictions if p == 'verse'])
        }
        
        avg_confidence = sum(conf for _, conf in predictions) / total_chunks
        winning_confidence = weighted_votes[final_label] / sum(weighted_votes.values())
        
        return {
            'prediction': final_label,
            'winning_confidence': winning_confidence,
            'average_confidence': avg_confidence,
            'total_chunks': total_chunks,
            'chunk_distribution': chunk_distribution,
            'chunk_predictions': predictions
        }

    def predict(self, text: str) -> Dict:
        chunks = self.split_text(text)
        for chunk in chunks:
            print("chunk", chunk)
        chunk_predictions = [self.predict_chunk(chunk) for chunk in chunks]
        results = self.majority_vote(chunk_predictions)
        return results

class UndoRedoText(scrolledtext.ScrolledText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.undo_stack = []
        self.redo_stack = []
        self.last_text = ""
        
        self.bind('<<Modified>>', self._on_modified)
        self.bind('<Control-z>', self._undo)
        self.bind('<Control-y>', self._redo)
        self.bind('<Control-Shift-Z>', self._redo)
        self.bind('<Control-a>', self._select_all)
        self.bind('<Control-x>', self._cut)
        self.bind('<Control-c>', self._copy)
        self.bind('<Control-v>', self._paste)

    def _on_modified(self, event):
        current_text = self.get("1.0", tk.END)
        if current_text != self.last_text:
            self.undo_stack.append(self.last_text)
            self.redo_stack.clear()
            self.last_text = current_text
        self.edit_modified(False)
        
    def _undo(self, event=None):
        if self.undo_stack:
            current_text = self.get("1.0", tk.END)
            self.redo_stack.append(current_text)
            previous_text = self.undo_stack.pop()
            self.delete("1.0", tk.END)
            self.insert("1.0", previous_text[:-1])
            self.last_text = previous_text
        return "break"
        
    def _redo(self, event=None):
        if self.redo_stack:
            current_text = self.get("1.0", tk.END)
            self.undo_stack.append(current_text)
            next_text = self.redo_stack.pop()
            self.delete("1.0", tk.END)
            self.insert("1.0", next_text[:-1])
            self.last_text = next_text
        return "break"
    
    def _select_all(self, event=None):
        self.tag_add(tk.SEL, "1.0", tk.END)
        self.mark_set(tk.INSERT, "1.0")
        self.see(tk.INSERT)
        return "break"
    
    def _cut(self, event=None):
        if self.tag_ranges(tk.SEL):
            self.event_generate("<<Cut>>")
        return "break"
    
    def _copy(self, event=None):
        if self.tag_ranges(tk.SEL):
            self.event_generate("<<Copy>>")
        return "break"
    
    def _paste(self, event=None):
        self.event_generate("<<Paste>>")
        return "break"

class DetailedResultsWindow:
    def __init__(self, parent, results):
        self.window = tk.Toplevel(parent)
        self.window.title("Detailed Prediction Results")
        self.window.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for results
        self.text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=70, height=20)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Format and display results
        self.display_results(results)
        
        # Add close button
        close_button = ttk.Button(main_frame, text="Close", command=self.window.destroy)
        close_button.pack(pady=10)
        
    def display_results(self, results):
        text = f"""Detailed Classification Results:

Final Prediction: {results['prediction'].title()}
Winning Confidence: {results['winning_confidence']:.2%}
Average Confidence: {results['average_confidence']:.2%}

Number of Chunks Analyzed: {results['total_chunks']}
Distribution:
- Verse chunks: {results['chunk_distribution']['verse']}
- Prose chunks: {results['chunk_distribution']['prose']}

Individual Chunk Predictions:
"""
        for i, (pred, conf) in enumerate(results['chunk_predictions'], 1):
            text += f"Chunk {i}: {pred.title()} (confidence: {conf:.2%})\n"
            
        self.text_widget.insert(tk.END, text)
        self.text_widget.config(state=tk.DISABLED)

# Modify the main GUI class
class VerseProsePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Verse/Prose Classifier")
        self.root.geometry("800x600")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        try:
            model_path = 'model_siku_kfold/verse_prose_model_fold_2'
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "Model not found! Please ensure you have trained the model."
                )
            
            self.ml_classifier = VerseProseMajorityClassifier(
                model_path=model_path,
                max_length=50,
                overlap_percent=0.2,
                min_chunk_size=50
            )
            self.rule_classifier = RuleBasedClassifier()
            self.current_classifier = self.ml_classifier  # Default to ML classifier
            
            self.create_widgets()
            self.create_menu()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo (Ctrl+Z)", command=lambda: self.text_input._undo())
        edit_menu.add_command(label="Redo (Ctrl+Y)", command=lambda: self.text_input._redo())
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut (Ctrl+X)", command=lambda: self.text_input.event_generate("<<Cut>>"))
        edit_menu.add_command(label="Copy (Ctrl+C)", command=lambda: self.text_input.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste (Ctrl+V)", command=lambda: self.text_input.event_generate("<<Paste>>"))
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All (Ctrl+A)", command=lambda: self.text_input._select_all())
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        
    def show_shortcuts(self):
        shortcuts = """
        Keyboard Shortcuts:
        
        Text Editing:
        - Undo: Ctrl+Z
        - Redo: Ctrl+Y or Ctrl+Shift+Z
        - Cut: Ctrl+X
        - Copy: Ctrl+C
        - Paste: Ctrl+V
        - Select All: Ctrl+A
        
        Application:
        - Predict: Enter (when Predict button is focused)
        - Clear: Escape (when Clear button is focused)
        - Show Details: Ctrl+D (when results are available)
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Classifier selection frame
        classifier_frame = ttk.LabelFrame(main_frame, text="Classifier Settings", padding="5")
        classifier_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Add classifier selection
        ttk.Label(classifier_frame, text="Method:").grid(row=0, column=0, padx=5)
        self.classifier_var = tk.StringVar(value="ml")
        ml_radio = ttk.Radiobutton(classifier_frame, text="Model-based", 
                                  variable=self.classifier_var, value="ml",
                                  command=self.on_classifier_change)
        rule_radio = ttk.Radiobutton(classifier_frame, text="Rule-Based", 
                                    variable=self.classifier_var, value="rule",
                                    command=self.on_classifier_change)
        ml_radio.grid(row=0, column=1, padx=5)
        rule_radio.grid(row=0, column=2, padx=5)
        
        # ML settings (initially visible)
        self.ml_settings_frame = ttk.Frame(classifier_frame)
        self.ml_settings_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        ttk.Label(self.ml_settings_frame, text="Chunk Size:").grid(row=0, column=0, padx=5)
        self.chunk_size_var = tk.StringVar(value="50")
        chunk_size_entry = ttk.Entry(self.ml_settings_frame, textvariable=self.chunk_size_var, width=10)
        chunk_size_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(self.ml_settings_frame, text="Overlap %:").grid(row=0, column=2, padx=5)
        self.overlap_var = tk.StringVar(value="20")
        overlap_entry = ttk.Entry(self.ml_settings_frame, textvariable=self.overlap_var, width=10)
        overlap_entry.grid(row=0, column=3, padx=5)
        
        # Rest of the widgets remain the same
        # [Previous widget creation code...]
        
        # Input area
        input_label = ttk.Label(main_frame, text="Enter Chinese text:")
        input_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.text_input = UndoRedoText(main_frame, width=80, height=15)
        self.text_input.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(0, 10))
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_input)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.details_button = ttk.Button(button_frame, text="Show Details", command=self.show_details)
        self.details_button.pack(side=tk.LEFT, padx=5)
        self.details_button.state(['disabled'])
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.result_display = ttk.Label(result_frame, text="", font=('TkDefaultFont', 12, 'bold'))
        self.result_display.pack(fill=tk.X, padx=5)
        
        self.confidence_display = ttk.Label(result_frame, text="")
        self.confidence_display.pack(fill=tk.X, padx=5)
        
        self.summary_display = ttk.Label(result_frame, text="")
        self.summary_display.pack(fill=tk.X, padx=5)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Bind keyboard shortcuts
        self.root.bind('<Return>', lambda e: self.predict_button.invoke())
        self.root.bind('<Escape>', lambda e: self.clear_button.invoke())
        self.root.bind('<Control-d>', lambda e: self.show_details())
        
        # Store the latest results
        self.latest_results = None

    def on_classifier_change(self):
        if self.classifier_var.get() == "ml":
            self.current_classifier = self.ml_classifier
            self.ml_settings_frame.grid()
        else:
            self.current_classifier = self.rule_classifier
            self.ml_settings_frame.grid_remove()
            self.details_button.state(['disabled'])

    def predict(self):
        self.result_display.config(text="")
        self.confidence_display.config(text="")
        self.summary_display.config(text="")

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            self.result_display.config(text="Please enter some text")
            return
        
        try:
            self.status_bar.config(text="Processing...")
            self.root.update()
            
            # Update ML classifier settings if using ML
            if self.classifier_var.get() == "ml":
                self.ml_classifier.max_length = int(self.chunk_size_var.get())
                self.ml_classifier.overlap_percent = float(self.overlap_var.get()) / 100.0
            
            # Get prediction from current classifier
            self.latest_results = self.current_classifier.predict(text)
            
            # Update display
            self.result_display.config(
                    text=f"Prediction: {self.latest_results['prediction'].title()}"
                )
            
            if self.classifier_var.get() == "ml":
                self.confidence_display.config(
                    text=f"Confidence: {self.latest_results['winning_confidence']:.2%}"
                )
                self.summary_display.config(
                    text=(f"Chunks: {self.latest_results['total_chunks']} "
                        f"(Verse: {self.latest_results['chunk_distribution']['verse']}, "
                        f"Prose: {self.latest_results['chunk_distribution']['prose']})")
                )
                self.details_button.state(['!disabled'])
            
            self.status_bar.config(text="Ready")
            
        except Exception as e:
            self.status_bar.config(text="Error occurred")
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

    def clear_input(self):
            self.text_input.delete("1.0", tk.END)
            self.result_display.config(text="")
            self.confidence_display.config(text="")
            self.summary_display.config(text="")
            self.latest_results = None
            self.details_button.state(['disabled'])
            self.status_bar.config(text="Ready")
        
    def show_details(self):
        if self.latest_results:
            DetailedResultsWindow(self.root, self.latest_results)

def main():
    root = tk.Tk()
    app = VerseProsePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()