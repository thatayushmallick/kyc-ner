## KYC Entity Extraction

A lightweight pipeline to extract key entities (e.g., Name, DOB, Aadhar/PAN number, Address) from scanned KYC documents using OCR and BERT-based token classification.

**Dataset:** “OCR Dataset of Multi‑Type Documents”  
https://www.kaggle.com/datasets/senju14/ocr-dataset-of-multi-type-documents

**Objective:** From OCR‑text of KYC docs, extract Name, DOB, Aadhar/PAN, Address.

**Example Output:**

```json
{
  "Name": "Anita Desai",
  "DOB": "1990-07-12",
  "Aadhar": "1234-5678-9012",
  "Address": "45 MG Road, Mysuru"
}
```
