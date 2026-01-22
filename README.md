# Exploring the Generalizability and Explainability of LLMs in Detecting Suicidal Ideation: The Impact of Data Heterogeneity

## Overview
With the recent advancement of artificial intelligence (AI) and large language models (LLMs), the use of text analysis to detect suicidal ideation can be a promising tool. However, the performance of such detection system could be influenced by the language use difference caused by individuals’ alexithymic characteristics (difficulties in expressing emotion with unique language pattern), resulting in the subgroup disparity. The current study aims to explore the capability of a detection system on a clinical sample of heterogeneous language use (i.e., systematic difference in language use as influenced by patient characteristics and the language context).

The workflow consists of two main steps:

1. **Data Processing**: Extract H11 question text from transcript CSV files, including full question text and interviewee responses.  
2. **Model Training**: Train transformer-based models (e.g., BERT) using the extracted text with 10-fold cross-validation, chunking (for sequences >512 tokens), overlapping (128 tokens), and early stopping (patience = 5). Evaluation metrics include AUC, Sensitivity, Specificity, PPV, and NPV.

---

## Repository Structure

| File / Folder            | Description |
|--------------------------|-------------|
| `data_processing.py`     | Script to process HAMD transcript CSV files and generate `H11_data.csv`. |
| `combine.csv`            | Metadata CSV containing case IDs, question numbers, text, and suicidal ideation labels. |
| `H11_data.csv`           | Processed dataset containing only H11 text and labels. |
| `model_training.py`      | Script for training BERT-based classifiers on the processed dataset. Supports chunking, majority voting, cross-validation, and early stopping. |
| `requirements.txt`       | List of Python dependencies required to run the scripts. |
| `README.md`              | Project documentation (this file). |

---

## Data Processing

### Steps
1. Load `combine.csv`, which contains case IDs, question numbers (H1 to H14), text, and labels (`no suicidal`, `passive`, `active`).  
2. For each case, read the corresponding transcript CSV file from the `data` folder.  
3. Identify the H11 question section and extract its text.  
4. Extract both the full text and the interviewee-specific responses.  
5. Handle missing files or columns gracefully by logging and inserting `NaN` values.  
6. Save the processed data into `H11_data.csv`.

### Run
```bash
# Run data processing
python data_processing.py

# Run model training
python model_training.py
