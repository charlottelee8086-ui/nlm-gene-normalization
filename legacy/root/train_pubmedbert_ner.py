import json
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
LABELS = ["O", "B-GENE", "I-GENE"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(example):
    text = example["text"]
    entities = example["entities"]

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    labels = []
    for start, end in encoding["offset_mapping"]:
        if start == end:
            labels.append(-100)
            continue

        tag = "O"
        for ent_start, ent_end in entities:
            if start >= ent_start and end <= ent_end:
                tag = "I-GENE"
                if start == ent_start:
                    tag = "B-GENE"
                break
        labels.append(label2id[tag])

    encoding["labels"] = labels
    encoding.pop("offset_mapping")
    return encoding

train_ds = Dataset.from_list(load_jsonl("ner_train.jsonl"))
test_ds = Dataset.from_list(load_jsonl("ner_test.jsonl"))

train_tok = train_ds.map(tokenize_and_align)
test_tok = test_ds.map(tokenize_and_align)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []

    for p_seq, l_seq in zip(preds, labels):
        pred_tags = []
        gold_tags = []
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            pred_tags.append(id2label[p])
            gold_tags.append(id2label[l])
        true_preds.append(pred_tags)
        true_labels.append(gold_tags)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

args = TrainingArguments(
    output_dir="pubmedbert_nlm_gene_ner",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
trainer.save_model("pubmedbert_nlm_gene_ner_best")
tokenizer.save_pretrained("pubmedbert_nlm_gene_ner_best")
