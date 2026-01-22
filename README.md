# Exploring the Generalizability and Explainability of LLMs in Detecting Suicidal Ideation: The Impact of Data Heterogeneity
## Overview
This project extracts interview question text from HAMD (Hamilton Depression Rating Scale) translated transcripts and uses BERT-based models to detect suicidal ideation using H11 and extract H11 Content.  

The workflow consists of two main steps:

1. **Data Processing**: Extract H11 question text from transcript CSV files, including full question text.  
2. **Model Training**: Train transformer-based models (e.g., BERT) using the extracted text with 5 fold cross-validation, chunking(>512 token),, and early stopping. Evaluation metrics include AUC, F1-score, Sensitivity, Specificity, PPV, and NPV.

---

## Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `data_processing.py` | Script to process HAMD transcript CSV files and generate `H11_data.csv`. |
| `combine.csv` | Metadata CSV containing case IDs, text,question number and suicide ideationlabels. |
| `H11_data.csv` | Extracted dataset only H11 text and labels. |
| `model_traning.py` | Script for training BERT-based classifiers on the processed dataset. Supports chunking, majority voting, cross-validation, and early stopping. |
| `requirements.txt` | List of Python dependencies required to run the scripts. |
| `README.md` | Project documentation (this file). |

---

## Data Processing

### Steps
1. Load `combine.csv`, which contains case IDs ,question number (H1 to H14),text and labels (`no suicidal`, `passive`, `active`).  
2. For each case, read the corresponding transcript file from `data folder`.  
3. Identify the H11 question section and extract H11 question
4. Handle missing files or columns gracefully by logging and inserting `NaN` values.  
5. Save the processed data into `H11_data.csv`.

### Run
#### Data Processing Run
```bash
#### Data Processing Run
python data_processing.py

#### Model Traning  Run
python data_processing.py
