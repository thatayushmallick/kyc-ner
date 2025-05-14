import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from transformers import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support

# ----------------------------------------------------------------------------
# 0. Parse command-line or env var for data root
# ----------------------------------------------------------------------------
if len(sys.argv) > 1:
    data_root = Path(sys.argv[1])
else:
    data_root = Path(os.getenv('KYC_DATA_ROOT', ''))

if not data_root or not data_root.exists():
    print("ERROR: No valid data_root provided.\n"
          "Usage (Windows):\n"
          "  python train_kyc_ner.py C:\\full\\path\\to\\archive\n"
          "Usage (Linux/Mac):\n"
          "  python train_kyc_ner.py /full/path/to/archive")
    sys.exit(1)
print(f"Using data_root = {data_root}")

# 1. Constants and label mappings
fields = ["O", "NAME", "DOB", "AADHAR", "ADDRESS"]
label2id = {lbl: idx for idx, lbl in enumerate(fields)}
id2label = {v: k for k, v in label2id.items()}

# 2. Utility: load OCR text from JSON annotations
def load_ocr_text(ann_path: Path) -> str:
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    tokens = [w.get('text', '') for w in ann.get('words', [])]
    return " ".join(tokens)

# 3. Heuristic labeling
def label_text(text: str) -> (List[str], List[str]):
    tokens = text.split()
    labels = ['O'] * len(tokens)
    for i, tok in enumerate(tokens):
        if re.match(r"Name[:]?", tok, re.I) and i+1 < len(tokens):
            labels[i+1] = 'NAME'
    dob_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b")
    for i, tok in enumerate(tokens):
        if dob_pattern.search(tok): labels[i] = 'DOB'
    aadhar_pat = re.compile(r"\b\d{4}-\d{4}-\d{4}\b")
    for i, tok in enumerate(tokens):
        if aadhar_pat.search(tok): labels[i] = 'AADHAR'
    if 'Address:' in tokens:
        start = tokens.index('Address:') + 1
        for j in range(start, min(start + 15, len(tokens))):
            labels[j] = 'ADDRESS'
    return tokens, labels

# 4. Dataset class
class KycDataset(Dataset):
    def __init__(self, texts: List[List[str]], tags: List[List[str]], tokenizer, max_len: int = 128):
        assert len(texts) == len(tags), "Texts and tags must match"
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        tokens, labels = self.texts[idx], self.tags[idx]
        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        word_ids = encoding.word_ids()
        label_ids = [(label2id[labels[wid]] if wid is not None else -100) for wid in word_ids]
        encoding.pop('offset_mapping')
        encoding['labels'] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}

# 5. Load and debug splits
def prepare_split(split_name: str):
    texts, tags = [], []
    print(f"\n--- Scanning {split_name} split ---")
    for doc_type in ['document', 'form', 'invoice', 'real_life']:
        ann_dir = data_root / doc_type / split_name / 'annotations'
        if not ann_dir.exists():
            print(f"Directory not found: {ann_dir}")
            continue
        files = list(ann_dir.glob('*.json'))
        print(f"Found {len(files)} files in {ann_dir}")
        for ann_file in files:
            txt = load_ocr_text(ann_file)
            toks, lbls = label_text(txt)
            texts.append(toks)
            tags.append(lbls)
    print(f"Total {split_name} examples: {len(texts)}")
    return texts, tags

train_texts, train_tags = prepare_split('train')
val_texts, val_tags = prepare_split('val')

if len(train_texts) == 0:
    print("ERROR: No training examples detected."
          " Please check the printed directory paths above.")
    sys.exit(1)

print(f"\nSummary: Training={len(train_texts)}, Validation={len(val_texts)}")

# 6. Tokenizer & Datasets
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_dataset = KycDataset(train_texts, train_tags, tokenizer)
val_dataset = KycDataset(val_texts, val_tags, tokenizer)

# 7. Model init
def create_model():
    return BertForTokenClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(fields), id2label=id2label, label2id=label2id)
model = create_model()

# 8. Metrics
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1).flatten()
    labels = p.label_ids.flatten()
    mask = labels != -100
    preds, lbls = preds[mask], labels[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(lbls, preds, average='weighted', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1}

# 9. Training args & Trainer
training_args = TrainingArguments(output_dir='./outputs', per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8, num_train_epochs=3,
                                  learning_rate=5e-5, weight_decay=0.01, logging_dir='./logs')
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, compute_metrics=compute_metrics)

# 10. Train & evaluate
print("\nStarting training...")
trainer.train()
print("\nTraining complete. Running final evaluation...")
metrics = trainer.evaluate(eval_dataset=val_dataset)
print("Validation metrics:", metrics)

# 11. Inference helper
def extract_entities(ocr_text: str) -> Dict[str, str]:
    tokens = ocr_text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt')
    outputs = model(**inputs).logits
    preds = torch.argmax(outputs, dim=2)[0].tolist()
    ents = {'Name':'', 'DOB':'', 'Aadhar':'', 'Address':''}
    for tok, pi in zip(tokens, preds):
        lbl = id2label[pi]
        if lbl == 'NAME': ents['Name'] += tok + ' '
        if lbl == 'DOB': ents['DOB'] += tok
        if lbl == 'AADHAR': ents['Aadhar'] += tok
        if lbl == 'ADDRESS': ents['Address'] += tok + ' '
    return {k: v.strip() for k, v in ents.items()}

# Usage example (Windows):
# python train_kyc_ner.py C:\\Users\\ayush\\Documents\\archive
