import argparse
import json
import re
import unicodedata
import warnings
from typing import Optional

import yaml
from datasets import load_dataset
from pydantic import BaseModel
from setfit import (SetFitModel, SetFitModelCardData, Trainer,
                    TrainingArguments, sample_dataset)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class Config(BaseModel):
    """
    so i can use DotOperator to access the value
    """

    exp_name: str
    pretrained_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    language: str = "en"
    optimizer: str = "AdamW"
    lr: float = 2e-5
    epochs: int = 1
    num_pairs_contrastive: int = 10
    batch_size: int = 16
    second_phase_end_to_end: bool = False
    train_set: str
    val_ratio: float = 0.2
    test_set: Optional[str]
    max_seq_length: int = 256
    hf_push_to_hub: Optional[str]
    few_shot: Optional[int]


def text_cleaning(input_string: str):
    lowercase = input_string.lower()
    remove_accented = (
        unicodedata.normalize("NFKD", lowercase)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    remove_extra_whitespace = re.sub(r"^\s*|\s\s*", " ", remove_accented).strip()
    return remove_extra_whitespace


def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def process_datasets(batch):
    return {
        "text": text_cleaning(batch["title"] + " " + batch["content"]),
        "label": batch["sentiment"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file, "r") as file:
        config_dict = yaml.safe_load(file)
        config = Config(**config_dict)

    print(f"DEBUGPRINT[1]: train.py:32: config={config}")
    return config


def main(config: Config):
    datasets = load_dataset("json", data_files=config.train_set)
    clean_ds = datasets.map(
        process_datasets,
        remove_columns=[
            "id",
            "title",
            "content",
            "rating",
            "sentiment",
            "date",
            "name",
        ],
    )
    ds = clean_ds["train"].train_test_split(test_size=config.val_ratio)
    num_samples = min(
        len(ds["train"].filter(lambda x: x["label"] == 0)),
        len(ds["train"].filter(lambda x: x["label"] == 1)),
    )
    print(f"DEBUGPRINT[3]: train.py:96: num_samples={num_samples}")
    if config.few_shot and config.few_shot < num_samples:
        num_samples = config.few_shot
    train_ds = sample_dataset(ds["train"], num_samples=num_samples)
    val_ds = ds["test"]
    print(f"DEBUGPRINT[1]: train.py:85: train_ds={train_ds}")
    print(f"DEBUGPRINT[2]: train.py:87: val_ds={val_ds}")
    model = SetFitModel.from_pretrained(
        config.pretrained_model,
        labels=["negative", "positive"],
        model_card_data=SetFitModelCardData(
            language=config.language,
            license="apache-2.0",
            dataset_name=config.train_set,
        ),
        use_differentiable_head=True,
    )

    train_args = TrainingArguments(
        output_dir=f"exp/{config.exp_name}",
        batch_size=config.batch_size,
        num_epochs=config.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric="accuracy",
        # column_mapping={"sentence": "text", "label": "label"},
    )

    trainer.train()

    if config.hf_push_to_hub:
        trainer.push_to_hub(config.hf_push_to_hub)

    if config.test_set:
        test_ds = load_dataset("json", data_files=config.test_set)
        clean_test_ds = test_ds.map(
            process_datasets,
            remove_columns=[
                "id",
                "title",
                "content",
                "rating",
                "sentiment",
                "date",
                "name",
            ],
        )
        trainer.evaluate(clean_test_ds["train"])


if __name__ == "__main__":
    config = parse_args()
    main(config)
