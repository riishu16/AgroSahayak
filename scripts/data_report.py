import json, os
import pandas as pd

# Row counts logged from scripts/clean_kcc.py and scripts/make_labels.py runs
FUNNEL = [
    ("Raw KCC records (Bihar, Gujarat, Maharashtra; Jan 2026)", 46328),
    ("After removing rows with missing crop / query type / text / answer", 18502),
    ("After removing incomplete calls, 'Other' crop and 'NA' query type", 6883),
    ("After removing duplicates (normalized text + crop + query type)", 2908),
    ("After removing very short queries (< 3 words)", 2773),
    ("After mapping to 9 categories (rare / unmappable types dropped)", 2453),
    ("After removing near-duplicates of test pool (TF-IDF cosine >= 0.8)", 2062),
]

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

out = []
add = out.append

add("# AgroSahayak - Data Report\n")

add("## 1. Data sources\n")
add(md_table(pd.DataFrame([
    ["Kisan Call Centre transcripts (data.gov.in)", "Bihar, Gujarat, Maharashtra, Jan 2026", "GODL-India"],
    ["IndicTrans2 En-Indic Distilled 200M (translation)", "English -> Hindi, Gujarati, Marathi", "MIT"],
], columns=["Source", "Used for", "Licence"])))

add("\n## 2. Cleaning pipeline\n")
add(md_table(pd.DataFrame([[a, f"{b:,}"] for a, b in FUNNEL], columns=["Step", "Rows"])))

st_path = "data/task1/stats.json"
st = json.load(open(st_path)) if os.path.exists(st_path) else {}

add("\n## 3. Train / test separation\n")
rows = [["Near-duplicate removal (TF-IDF cosine >= 0.8) against test pool", "241 train rows removed"]]
if st:
    rows += [["Exact text overlap train vs test, all languages and scripts (before)", st["overlap_before"]],
             ["Train examples removed because of exact overlap", st["train_removed"]],
             ["Exact text overlap after removal", st["overlap_after"]]]
add(md_table(pd.DataFrame(rows, columns=["Check", "Result"])))

if os.path.exists("data/final/train_fixed.csv") and os.path.exists("data/final/test_final.csv"):
    tr = pd.read_csv("data/final/train_fixed.csv")
    te = pd.read_csv("data/final/test_final.csv")
    cats = sorted(set(tr["category"]) | set(te["category"]))
    d = pd.DataFrame({"Category": cats,
                      "Train rows": [int((tr["category"] == c).sum()) for c in cats],
                      "Test items (100)": [int((te["category"] == c).sum()) for c in cats]})
    add("\n## 4. Category distribution\n")
    add(md_table(d))
    add(f"\nTest items with crop = unknown (crop not named in the query): {int((te['crop'] == 'unknown').sum())}")

if os.path.exists("data/task1/train_task1.jsonl"):
    trn = load_jsonl("data/task1/train_task1.jsonl")
    tst = load_jsonl("data/task1/test_task1.jsonl")
    t = pd.DataFrame([{**x["gold"], "lang": x["lang"], "script": x["script"], "synthetic": x["synthetic"]} for x in trn])
    s = pd.DataFrame([{"lang": x["lang"], "script": x["script"]} for x in tst])
    add("\n## 5. Task 1 dataset (query -> JSON)\n")
    add(md_table(pd.DataFrame([
        ["Training examples", f"{len(trn):,}"],
        ["  of which synthetic (weather, post_harvest)", int(t["synthetic"].sum())],
        ["  with crop = unknown", f"{(t['crop'] == 'unknown').mean():.1%}"],
        ["  with a specific entity", f"{(t['specific_entity'] != 'none').mean():.1%}"],
        ["Test examples (100 items x 7 variants)", f"{len(tst):,}"],
    ], columns=["Item", "Value"])))
    g1 = t.groupby(["lang", "script"]).size().rename("Train").reset_index()
    g2 = s.groupby(["lang", "script"]).size().rename("Test").reset_index()
    add("\nExamples by language and script:\n")
    add(md_table(g1.merge(g2, on=["lang", "script"], how="outer").fillna(0).astype({"Train": int, "Test": int})
                 .rename(columns={"lang": "Language", "script": "Script"})))

add("\n## 6. Known limitations\n")
for x in [
    "KCC QueryText is an operator-written English summary, not the farmer's own words.",
    "Hindi, Gujarati and Marathi queries are machine translations (IndicTrans2); Gujarati and Marathi were not verified by a native speaker.",
    "Roman-script queries are rule-based transliterations with random spelling noise, not real farmer text.",
    "Training entity labels are keyword-rule based; test entity labels were reviewed manually.",
    "Test labels were drafted with an AI tool (Claude) and reviewed.",
    "Weather and post_harvest had almost no training data after de-duplication; synthetic examples were added and are flagged.",
    "Only one month (Jan 2026) of three states was used.",
]:
    add(f"- {x}")

text = "\n".join(out)
os.makedirs("reports", exist_ok=True)
open("reports/data_report.md", "w", encoding="utf-8").write(text)
print(text)
