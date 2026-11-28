import re, json, collections
import pandas as pd

# formulation codes that follow a chemical name, e.g. "mancozeb 75% WP", "कार्टाफ हाइड्रोक्लोराइड 50% एसपी"
CODES = r"(?:wp|ec|sc|sl|sp|sg|dp|gr|fs|wg|zc|cs|ew|od|ds|एसपी|ईसी|डब्ल्यूपी|डब्लूपी|एससी|एसएल|डीपी|जीआर|ડબલ્યુપી|ઈસી|એસસી|એસએલ|એસપી)"
PAT = re.compile(r"((?:[^\s\d%|।.,]+\s+){1,2})\d+(?:\.\d+)?\s*%\s*" + CODES, re.I)

names = collections.Counter()
rows = 0
for split in ("train", "test"):
    for l in open(f"data/task2/{split}_task2.jsonl", encoding="utf-8"):
        x = json.loads(l)
        if x["kind"] != "answerable": continue
        src = x["messages"][0]["content"].split("Source: ", 1)[1]
        rows += 1
        for m in PAT.finditer(src):
            names[" ".join(m.group(1).lower().split())] += 1

print(f"Answerable examples scanned: {rows}")
print(f"Distinct chemical-like names: {len(names)}\n")
print("Top 50 (check each against the official banned / restricted list):")
for n, c in names.most_common(50):
    print(f"{c:5d}  {n}")
pd.DataFrame(names.most_common(), columns=["name", "count"]).to_csv("results/chemical_names_in_snippets.csv", index=False, encoding="utf-8-sig")
print("\nSaved results/chemical_names_in_snippets.csv")