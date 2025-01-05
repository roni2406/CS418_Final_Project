A machine learning and rule-based application for classifying Sinonom/Chinese texts as either verse or prose.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Rule-based Approach](#rule-based-approach)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [GUI Application](#gui-application)


## Features
- Rule-based classification system
- Model-based classification using GuwenBERT and SikuBERT
- Interactive GUI for real-time classification
- Support for both single-text and batch classification
- Performance evaluation metrics

## Project Structure
```
.
├── dataset/                 # Training and testing datasets
├── rule_based/             # Rule-based classification implementation
├── statistics/             # Model evaluation results
├── training/               # Model training code
├── inference_GUI.py        # GUI application
├── inference_test.txt      # External test data
└── scoring.py              # Evaluation metrics calculation
```

## Requirements
- Python <= 3.11
- CUDA-compatible GPU with >4GB VRAM
- CUDA Toolkit
- Miniconda (optional, but recommended)

## Usage

### Rule-based Approach
1. Prepare your input file in the same format as `datasets/dataset_test.json`
2. Run the classifier:
```bash
python rule_based/detection.py
```
3. The answer will be saved in folder  `statistics`

### Model Training
1. Prepare your training data in the specified format
2. Run the training script:
```bash
python training/{train_model}.py
```

The model will be saved to model folder (e.g `model_guwen/verse_prose_model/`), and the answer will be saved in folder `statistics`.

### Evaluation
To evaluate model performance: run the scoring.py file, replace the file results and answers with appropriate link.

The result on the test set will be written to the console as format below:
- Precision: 0.XX
- Recall: 0.XX
- F1 Score: 0.XX

### GUI Application
1. Ensure you have a trained model in the specified directory
2. Launch the GUI:
```bash
python inference_GUI.py
```
3. Input your text and see the result.