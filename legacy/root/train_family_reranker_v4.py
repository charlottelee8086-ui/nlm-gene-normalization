import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

TRAIN_FILE = Path("family_pairwise_train_v4.jsonl")
OUT_DIR = Path("family_reranker_pubmedbert_v4")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_input(ex):
    text = (
        "Mention: {mention}\n"
        "Context: {context}\n"
        "Candidate Gene Symbol: {symbol}\n"
        "Candidate Description: {desc}\n"
        "Candidate Aliases: {aliases}\n"
        "Candidate Species TaxID: {taxid}\n"
        "Exact mention match: {exact}\n"
        "Candidate appears in context: {appears}\n"
        "Overlapping terms: {overlap}\n"
        "Candidate Gene ID: {gid}"
    ).format(
        mention=ex["mention"],
        context=ex["context"],
        symbol=ex.get("candidate_symbol", "UNKNOWN"),
        desc=ex.get("candidate_description", "UNKNOWN"),
        aliases=ex.get("candidate_aliases", "UNKNOWN"),
        taxid=ex.get("candidate_tax_id", "UNKNOWN"),
        exact=str(ex.get("exact_mention_match", False)),
        appears=str(ex.get("appears_in_context", False)),
        overlap=ex.get("overlap_terms", "NONE"),
        gid=ex["candidate_gene_id"],
    )

    return {
        "text": text,
        "label": int(ex["label"]),
    }



rows = [format_input(x) for x in load_jsonl(TRAIN_FILE)]

random.seed(13)
random.shuffle(rows)

split = int(len(rows) * 0.9)
train_rows = rows[:split]
dev_rows = rows[split:]

train_ds = Dataset.from_list(train_rows)
dev_ds = Dataset.from_list(dev_rows)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=384,
        padding="max_length",
    )


train_ds = train_ds.map(tokenize, batched=True)
dev_ds = dev_ds.map(tokenize, batched=True)


train_ds = train_ds.remove_columns(["text"])
dev_ds = dev_ds.remove_columns(["text"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

pos = sum(x["label"] for x in train_rows)
neg = len(train_rows) - pos


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1.0, neg / max(pos, 1)],
                dtype=torch.float,
                device=logits.device,
            )
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)


trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

print("saved:", OUT_DIR)

