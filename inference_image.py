import sys, json
from pathlib import Path

from PIL import Image
import pytesseract
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# path to checkpoint folder:
MODEL_CHECKPOINT = Path('./outputs/checkpoint-813')

PRETRAINED_TOKENIZER = 'bert-base-uncased'

id2label = {0: 'O', 1: 'NAME', 2: 'DOB', 3: 'AADHAR', 4: 'ADDRESS'}

def ocr_image_to_text(image_path: Path) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# load model and tokenizer
print(f"Loading tokenizer (‘{PRETRAINED_TOKENIZER}’) and model from {MODEL_CHECKPOINT}…")
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_TOKENIZER)
model     = BertForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=id2label,
    label2id={v:k for k,v in id2label.items()}
)
model.eval()

def extract_entities_from_text(text: str) -> dict:
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()

    out = {'Name':'', 'DOB':'', 'Aadhar':'', 'Address':''}
    for tok, lab_id in zip(tokens, preds):
        lbl = id2label[lab_id]
        if lbl == 'NAME':    out['Name']     += tok + ' '
        if lbl == 'DOB':     out['DOB']      += tok
        if lbl == 'AADHAR':  out['Aadhar']   += tok
        if lbl == 'ADDRESS': out['Address']  += tok + ' '
    # striping trailing spaces
    return {k:v.strip() for k,v in out.items()}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_image.py /path/to/image.jpg")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"ERROR: file not found: {img_path}")
        sys.exit(1)

    print(f"\nOCR’ing image {img_path}…")
    raw_text = ocr_image_to_text(img_path)
    print("→ Raw OCR:\n", raw_text, "\n")

    print("Extracting entities …")
    entities = extract_entities_from_text(raw_text)
    print("\nResult:")
    print(json.dumps(entities, indent=2))
