from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = """You are doing biomedical gene normalization.

Task:
Choose exactly one Gene ID from the candidate list.

Mention:
MAPK

Context:
The phosphorylation of p38 MAPK was increased after treatment.

Candidates:
1432    MAPK14    human    mitogen-activated protein kinase 14
5594    MAPK1     human    mitogen-activated protein kinase 1
5595    MAPK3     human    mitogen-activated protein kinase 3

Answer format:
GeneID: <one candidate gene_id>
"""

messages = [
    {
        "role": "system",
        "content": "You are a biomedical gene normalization assistant. Choose only from the candidate Gene IDs.",
    },
    {
        "role": "user",
        "content": prompt,
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

generated = output[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
