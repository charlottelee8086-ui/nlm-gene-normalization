import json
import requests

with open("ner_test.jsonl", encoding="utf-8") as f:
    doc = json.loads(next(f))

url = "http://bern2.korea.ac.kr/plain"

r = requests.post(
    url,
    json={"text": doc["text"]},
    timeout=120,
)

print("status:", r.status_code)
data = r.json()
print(json.dumps(data, indent=2, ensure_ascii=False)[:5000])
