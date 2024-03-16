import argparse
from datasets import load_dataset
from setfit import sample_dataset
import torch
from transformers import AutoModel, AutoTokenizer
import re
import unicodedata
from sklearn.linear_model import LogisticRegression
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")



def extract_CLS_token(batch):
    inputs = {k:v for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        cls_token = model(**inputs).last_hidden_state
    return {"hidden_state": cls_token[:,0].cpu().numpy()}

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--datasets", type=str, required=True)
    args.add_argument("--num_samples", type=int, default=8)
    config = args.parse_args()
    return config

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

def text_cleaning(input_string: str):
    lowercase = input_string.lower()
    remove_accented = (
        unicodedata.normalize("NFKD", lowercase)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    remove_extra_whitespace = re.sub(r"^\s*|\s\s*", " ", remove_accented).strip()
    return remove_extra_whitespace

def process_datasets(batch):
    return {
        "text": text_cleaning(batch["title"] + " " + batch["content"]),
        "label": batch["sentiment"],
    }
def main(config):
    ds = load_dataset("json", data_files=config.datasets)
    ds = ds.map(process_datasets)
    ds = ds.map(tokenize, batched= True)
    ds = ds["train"].train_test_split(test_size=0.2)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_ds = sample_dataset(ds["train"], num_samples=config.num_samples)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_ds =  train_ds.map(extract_CLS_token, batched=True)

    val_ds = ds["test"]
    val_ds = val_ds.map(extract_CLS_token, batched = True)

    X_train = np.array(train_ds["hidden_state"])
    X_valid = np.array(val_ds["hidden_state"])
    y_train = np.array(train_ds["label"])
    y_valid = np.array(val_ds["label"])

    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, y_train)
    val_acc = lr_clf.score(X_valid, y_valid)
    print(f"val_acc:{val_acc}")



if __name__ == "__main__":
    config = parse_args()
    main(config)
    

