import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AdamW, get_scheduler
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold
import os

# -----------------------------
# Seed everything for reproducibility
# -----------------------------
def seed_everything(seed=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Dataset class with chunking
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, overlap=128, cases=None):
        self.samples = []
        for idx, (text, label) in enumerate(zip(texts, labels)):
            case_id = cases[idx] if cases is not None else idx
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                return_attention_mask=False,
                return_tensors=None,
            )
            input_ids = encoding["input_ids"]
            start = 0
            while start < len(input_ids):
                end = start + max_len
                chunk = input_ids[start:end]
                if len(chunk) < max_len:
                    chunk += [tokenizer.pad_token_id] * (max_len - len(chunk))
                self.samples.append({
                    "input_ids": chunk,
                    "label": label,
                    "case_id": case_id,
                    "chunk_id": start // (max_len - overlap)
                })
                if end >= len(input_ids):
                    break
                start += max_len - overlap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "case_id": sample["case_id"],
            "chunk_id": sample["chunk_id"],
        }

# -----------------------------
# Custom classifier for base model
# -----------------------------
class CustomClassifier(torch.nn.Module):
    def __init__(self, base_model, hidden_size, num_labels=2):
        super().__init__()
        self.base = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled_output))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return type('Output', (), {'loss': loss, 'logits': logits})

# -----------------------------
# Metrics calculation
# -----------------------------
def compute_metrics(y_true, y_prob, y_pred):
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "AUC": auc,
        "Sensitivity": recall,
        "Specificity": specificity,
        "PPV": precision,
        "NPV": npv
    }

# -----------------------------
# Convert dataset back to DataFrame for majority voting
# -----------------------------
def dataset_to_dataframe(dataset, tokenizer):
    records = []
    for sample in dataset:
        input_ids = sample["input_ids"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        records.append({
            "case_id": sample["case_id"],
            "chunk_id": sample["chunk_id"],
            "label": sample["label"].item(),
            "chunk_text": text
        })
    return pd.DataFrame(records)

# -----------------------------
# Training one epoch
# -----------------------------
def train_epoch(model, data_loader, optimizer, scheduler, device, scaler):
    model.train()
    losses = []
    correct_predictions = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# -----------------------------
# Evaluate model on validation set
# -----------------------------
def eval_model(model, data_loader, device):
    model.eval()
    true_labels, pred_probs, cased = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            true_labels.extend(labels.cpu().numpy())
            pred_probs.extend(probs.cpu().numpy())
            cased.extend(batch["case_id"])
    return np.array(cased), np.array(true_labels), np.array(pred_probs)

# -----------------------------
# Main loop: load models, datasets, and train with early stopping
# -----------------------------
models = ['indiejoseph/bert-base-cantonese']
all_model_metrics = []
model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)
datatype=['combine','H11_data.csv','nonH11_data']

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for data_nam in datatype:
        data1 = pd.read_csv(f"{data_nam}.csv")
        data1['label'] = data1['suicidal'].map({'no suicidal': 0, 'passive': 1, 'active': 1})
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y = data1['label']
        X = data1['text']
        all_folds_results=[]
        
        folds = list(skf.split(X, y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx} ---")
            train_data = data1.iloc[train_idx].reset_index(drop=True)
            val_data = data1.iloc[val_idx].reset_index(drop=True)

            train_texts = train_data['text'].tolist()
            train_labels = train_data['label'].tolist()
            val_texts = val_data['text'].tolist()
            val_labels = val_data['label'].tolist()
            val_cases = val_data["case"].tolist()
            
            train_dataset = TextDataset(train_texts, train_labels, tokenizer)
            val_dataset = TextDataset(val_texts, val_labels, tokenizer, cases=val_cases)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Load model
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
            except:
                base = AutoModel.from_pretrained(model_name)
                model = CustomClassifier(base_model=base, hidden_size=base.config.hidden_size).to(device)
            
            optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*30)
            scaler = GradScaler()
            
            # -----------------------------
            # Training loop with early stopping
            # -----------------------------
            patience = 5
            counter = 0
            best_auc = 0.0
            
            for epoch in range(30):
                print(f"Epoch {epoch + 1}")
                train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
                print(f"Train loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                
                # Evaluate on validation set
                cased, true_labels, pred_probs = eval_model(model, val_loader, device)
                val_chunks = dataset_to_dataframe(val_dataset, tokenizer)
                fold_results = pd.DataFrame({
                    "cased": val_chunks["case_id"],
                    "true_label": val_chunks["label"],
                    "prob0": pred_probs[:, 0],
                    "prob1": pred_probs[:, 1],
                })
                
                # Majority voting per case
                voting_df = (
                    fold_results
                    .assign(pred=(fold_results["prob1"] > 0.5).astype(int))
                    .groupby("cased")
                    .agg({
                        "true_label": "first",
                        "pred": lambda x: x.value_counts().idxmax(),
                        "prob1": "mean"
                    })
                    .reset_index()
                    .rename(columns={"cased": "case", "true_label": "label", "prob1": "prob"})
                )
                
                y_true = voting_df["label"]
                y_pred = voting_df["pred"]
                y_prob = voting_df["prob"]
                
                metrics = compute_metrics(y_true, y_prob, y_pred)
                current_auc = metrics["AUC"]
                print(f"Validation AUC: {current_auc:.4f}")
                
                # Early stopping check
                if current_auc > best_auc:
                    best_auc = current_auc
                    counter = 0
                    # Save best model per fold
                    torch.save(model.state_dict(), f"{model_dir}/{model_name.split('/')[-1]}_fold{fold_idx}_best.pt")
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            
            # Compute p-value for Mann-Whitney U test
            pos_probs = y_prob[y_true == 1]
            neg_probs = y_prob[y_true == 0]
            if (len(pos_probs) > 0) and (len(neg_probs) > 0):
                u_stat, p_value = mannwhitneyu(pos_probs, neg_probs, alternative='two-sided')
            else:
                p_value = np.nan
            
            metrics["Model"] = model_name.split("/")[-1]
            metrics["Question"] = f'{data_nam}'
            metrics["pvalue"] = p_value
            all_model_metrics.append(metrics)
            all_folds_results.append(fold_results)

# -----------------------------
# Save metrics summary
# -----------------------------
metrics_df = pd.DataFrame(all_model_metrics)
metrics_df.to_csv("./Metrics_summary.csv", index=False)
print(metrics_df)
