import json, random, re, os
random.seed(0)
norm = lambda t: " ".join(str(t).split()).lower()

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def question(x):
    u = x["messages"][0]["content"]
    if "Query: " in u:  return norm(u.split("Query: ", 1)[1])
    return norm(u.split("Question: ", 1)[1].split("\nSource:")[0])

def conv(x, source):
    return {"prompt": [x["messages"][0]], "completion": [x["messages"][1]], "source": source,
            "lang": str(x.get("lang", "")), "script": str(x.get("script", "native")), "kind": str(x.get("kind", "task1"))}

t1 = load("data/task1/train_task1.jsonl")
t2 = load("data/task2/train_task2.jsonl")
sf = load("data/task2/train_safety.jsonl")

# every question that appears in ANY test file
test_q = set()
for p in ("data/task1/test_task1.jsonl", "data/task2/test_task2.jsonl", "data/task2/test_traps.jsonl"):
    test_q |= {question(x) for x in load(p)}

items = [(x, "task1") for x in t1] + [(x, "task2") for x in t2] + [(x, "safety") for x in sf]
before = len(items)
items = [(x, s) for x, s in items if question(x) not in test_q]
print("Removed because the question also appears in a test file:", before - len(items))

random.shuffle(items)
n_val = max(1, int(0.03 * len(items)))
val, train = items[:n_val], items[n_val:]
safety_extra = [(x, s) for x, s in train if s == "safety"] * 2      # safety examples x3 in total
train = train + safety_extra
random.shuffle(train)

def dump(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for x, s in rows: f.write(json.dumps(conv(x, s), ensure_ascii=False) + "\n")
dump(train, "data/train_all.jsonl"); dump(val, "data/val_all.jsonl")

from collections import Counter
print("Train examples:", len(train), dict(Counter(s for _, s in train)))
print("Validation examples:", len(val), dict(Counter(s for _, s in val)))
print("Overlap train/val questions with test:", len({question(x) for x, _ in train + val} & test_q), "(must be 0)")