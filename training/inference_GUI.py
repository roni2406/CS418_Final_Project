import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import re


def remove_punctuation_and_spaces(text):
    # Define a regex pattern for punctuation (both ASCII and Chinese)
    pattern = r'[，。！？、；：“”‘’.,!?;:"\'`~@#$%^&*()\-_=+[\]{}|\\<>/。]'
    # Remove punctuation
    text = re.sub(pattern, '', text)
    # Remove spaces
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    return text


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
        if most_common / total_sequences > 0.2: # set a reasonable threshold, I find that 0 is the best
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

class VerseProsePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Verse/Prose Classifier")
        self.root.geometry("800x600")

        # Configure root window to be resizable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize model and tokenizer
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = 'new_model_siku/verse_prose_model'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "Model not found! Please ensure you have trained the model first "
                    "and the files are saved in the 'model_save/verse_prose_model' directory."
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create GUI elements
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
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Make the main frame expandable
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Classification method selection
        method_frame = ttk.LabelFrame(main_frame, text="Classification Method", padding="5")
        method_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.method_var = tk.StringVar(value="model")
        self.method_var.trace_add('write', self.on_method_change)

        ttk.Radiobutton(method_frame, text="Model Approach", variable=self.method_var, 
                       value="model").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="Rule-based Approach", variable=self.method_var, 
                       value="rule").pack(side=tk.LEFT, padx=5)
        
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
        self.root.bind('<Return>', lambda e: self.predict_button.invoke())
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_input)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.root.bind('<Escape>', lambda e: self.clear_button.invoke())
        
        # Result area
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.result_display = ttk.Label(result_frame, text="", font=('TkDefaultFont', 12, 'bold'))
        self.result_display.pack(side=tk.TOP, anchor=tk.W)
        
        self.detailed_result = ttk.Label(result_frame, text="")
        self.detailed_result.pack(side=tk.TOP, anchor=tk.W)
        
        self.confidence_display = ttk.Label(result_frame, text="")
        self.confidence_display.pack(side=tk.TOP, anchor=tk.W)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def on_method_change(self, *args):
        """Clear input and output when classification method changes"""
        self.clear_input(delete_text=False)
        self.status_bar.config(text="Classification method changed - Ready")

    def predict(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            self.result_display.config(text="Please enter some text")
            self.confidence_display.config(text="")
            self.detailed_result.config(text="")
            return
        
        try:
            self.status_bar.config(text="Processing...")
            self.root.update()
            
            if self.method_var.get() == "model":
                # Model-based prediction
                text = remove_punctuation_and_spaces(text)
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                    prediction = torch.argmax(outputs.logits, dim=1)
                    confidence = probabilities[0][prediction[0]].item()
                
                label_mapping = {0: "Prose", 1: "Verse"}
                predicted_label = label_mapping[prediction.item()]
                
                self.result_display.config(text=f"Prediction: {predicted_label}")
                self.confidence_display.config(text=f"Confidence: {confidence:.2%}")
                self.detailed_result.config(text="")
                
            else:
                # Rule-based prediction
                predicted_label = detect_prose_verse(text)
                verse_type = "N/A"
                if predicted_label == "verse":
                    cleaned_text = text.replace(" ", "").replace("，", "。").replace("。", "\n")
                    verse_type = classify_verse(cleaned_text)
                
                self.result_display.config(text=f"Prediction: {predicted_label.title()}")
                self.detailed_result.config(text=f"Verse Type: {verse_type}" if predicted_label == "verse" else "")
                self.confidence_display.config(text="")
            
            self.status_bar.config(text="Ready")
            
        except Exception as e:
            self.status_bar.config(text="Error occurred")
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
    
    def clear_input(self, delete_text=True):
        if delete_text:
            self.text_input.delete("1.0", tk.END)
        self.result_display.config(text="")
        self.confidence_display.config(text="")
        self.detailed_result.config(text="")
        self.status_bar.config(text="Ready")

def main():
    root = tk.Tk()
    app = VerseProsePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()