import json
import numpy as np
import torch
from torch import nn
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def format_example(x):
    return {
        "input_text": (
            x["text"]
            + " [SEP] Mention: "
            + str(x["mention"])
            + " [SEP] Candidate Gene ID: "
            + str(x["candidate_gene_id"])
        ),
        "label": int(x["label"]),
    }


train_rows = [format_example(x) for x in load_jsonl("linker_train.jsonl")]
dev_rows = [format_example(x) for x in load_jsonl("linker_dev.jsonl")]

train_ds = Dataset.from_list(train_rows)
dev_ds = Dataset.from_list(dev_rows)


def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        truncation=True,
        max_length=256,
    )


train_tok = train_ds.map(tokenize, batched=True)
dev_tok = dev_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


collator = DataCollatorWithPadding(tokenizer)


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    p, r, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.5], device=logits.device)
        )

        loss = loss_fn(
            logits.view(-1, 2),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


args = TrainingArguments(
    output_dir="pubmedbert_ncbigene_linker",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=dev_tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.evaluate()

trainer.save_model("pubmedbert_ncbigene_linker_best")
tokenizer.save_pretrained("pubmedbert_ncbigene_linker_best")
