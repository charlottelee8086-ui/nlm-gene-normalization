import re
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "pubmedbert_ncbigene_linker_v2_best"

NER_PRED = "pubmedbert_ner_test_predictions.tsv"
TRAIN_ARROW = "nlm_gene-train.arrow"
TEST_ARROW = "nlm_gene-test.arrow"

OUT = "pubmedbert_neural_linked_tfidf.PubTator"

TOPK = 20


def norm_text(s):
    s = s.lower()
    s = s.replace("κ", "kappa")
    s = s.replace("β", "beta")
    s = s.replace("α", "alpha")
    s = s.replace("γ", "gamma")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_arrow(path):
    with pa.memory_map(path, "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def make_text(doc):
    passages = sorted(doc["passages"], key=lambda p: p["offsets"][0][0])
    text = ""
    for p in passages:
        start = p["offsets"][0][0]
        content = p["text"][0]
        if len(text) < start:
            text += " " * (start - len(text))
        text += content
    return text


def get_context(text, start, end, window=160):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return (
        text[left:start]
        + " [MENTION] "
        + text[start:end]
        + " [/MENTION] "
        + text[end:right]
    )


# Build synonym KB from train
train_rows = read_arrow(TRAIN_ARROW)

synonyms = []
synonym_gene_ids = []
gid2names = defaultdict(Counter)

seen = set()

for doc in train_rows:
    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        if not ent.get("normalized"):
            continue

        mention = ent["text"][0].strip()

        for norm in ent["normalized"]:
            if norm.get("db_name") != "NCBIGene":
                continue

            gid = norm["db_id"]
            pair = (mention, gid)

            if pair in seen:
                continue
            seen.add(pair)

            synonyms.append(mention)
            synonym_gene_ids.append(gid)
            gid2names[gid][mention] += 1

gid2bestname = {
    gid: names.most_common(1)[0][0]
    for gid, names in gid2names.items()
}

print("synonyms:", len(synonyms))
print("gene ids:", len(gid2bestname))

# Character ngram TF-IDF retrieval
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    lowercase=True,
    preprocessor=norm_text,
)

syn_matrix = vectorizer.fit_transform(synonyms)

# Test texts
test_rows = read_arrow(TEST_ARROW)
pmid2text = {doc["document_id"]: make_text(doc) for doc in test_rows}

# Neural reranker
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

linked = 0
unlinked = 0

with open(NER_PRED, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        pmid, start, end, mention, etype = parts[:5]
        if etype != "Gene":
            continue

        start = int(start)
        end = int(end)

        q = vectorizer.transform([mention])
        scores = (q @ syn_matrix.T).toarray()[0]

        if scores.max() <= 0:
            unlinked += 1
            continue

        top_idx = scores.argsort()[::-1][:TOPK]

        # unique candidate gene IDs
        candidate_ids = []
        for idx in top_idx:
            gid = synonym_gene_ids[idx]
            if gid not in candidate_ids:
                candidate_ids.append(gid)

        if not candidate_ids:
            unlinked += 1
            continue

        context = get_context(pmid2text[pmid], start, end)

        best_gid = None
        best_score = -1e9

        for gid in candidate_ids:
            candidate_name = gid2bestname.get(gid, "unknown gene")

            input_text = (
                context
                + " [SEP] Mention: "
                + mention
                + " [SEP] Candidate: "
                + candidate_name
                + " [SEP] Candidate Gene ID: "
                + str(gid)
            )

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            score = logits[0][1].item()

            if score > best_score:
                best_score = score
                best_gid = gid

        out.write(
            f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{best_gid}\t{best_score:.4f}\n"
        )

        linked += 1

print("linked:", linked)
print("unlinked:", unlinked)
print("saved:", OUT)
