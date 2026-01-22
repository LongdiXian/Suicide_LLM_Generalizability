# Suicidal Ideation Detection using BERT on HAMD Transcripts

## Overview
This project extracts interview question text from HAMD (Hamilton Depression Rating Scale) translated transcripts and uses BERT-based models to detect suicidal ideation.  

The workflow consists of two main steps:

1. **Data Processing**: Extract H11 question text from transcript CSV files, including full question text and only the interviewee's responses.  
2. **Model Training**: Train transformer-based models (e.g., BERT) using the extracted text with cross-validation, chunking, and early stopping. Evaluation metrics include AUC, F1-score, Sensitivity, Specificity, PPV, and NPV.

---

## Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `extract_H11.py` | Script to process HAMD transcript CSV files and generate `H11_data.csv`. |
| `combine.csv` | Metadata CSV containing case IDs and labels. |
| `H11_data.csv` | Extracted dataset containing H11 text, interviewee text, and labels. |
| `train_model.py` | Script for training BERT-based classifiers on the processed dataset. Supports chunking, majority voting, cross-validation, and early stopping. |
| `saved_models/` | Folder where the trained model weights are stored. Each fold's best model is saved here. |
| `requirements.txt` | List of Python dependencies required to run the scripts. |
| `README.md` | Project documentation (this file). |

---

## Data Processing

### Steps
1. Load `combine.csv`, which contains case IDs and labels (`no suicidal`, `passive`, `active`).  
2. For each case, read the corresponding transcript file from `Y:/Rong/transcripts/HAMD/translated_can_HAMD/`.  
3. Identify the H11 question section and extract:
   - **Full text** of the H11 section.
   - **Interviewee-only text**.  
4. Handle missing files or columns gracefully by logging and inserting `NaN` values.  
5. Save the processed data into `H11_data.csv`.

### Run
```bash
python extract_H11.py
