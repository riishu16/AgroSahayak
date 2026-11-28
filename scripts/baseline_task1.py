import json, re, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

def load(p):
    rows = []
    for l in open(p, encoding="utf-8"):
        x = json.loads(l)
        x["query"] = x["messages"][0]["content"].split("Query: ", 1)[1]
        rows.append(x)
    return rows

train = load("data/task1/train_task1.jsonl")
test = load("data/task1/test_task1.jsonl")

# ---------- Baseline A: keyword rules (English only) ----------
CAT_RULES = [
    ("weather",      r"weather|rainfall|\brain\b|temperature|forecast"),
    ("market",       r"market|mandi|\bprice|\bmsp\b"),
    ("variety_seed", r"variet|seed rate|seed treatment|\bseeds?\b|hybrid"),
    ("irrigation",   r"irrigat|water management|\bdrip\b|sprinkler"),
    ("cultivation",  r"weed|herbicide|weedicide|sowing|land preparation|field preparation|spacing|transplant|harvest|cultivation practice"),
    ("insect_pest",  r"aphid|borer|caterpillar|thrips|whitefly|mite|\bpest|insect|jassid|hopper|weevil|termite|larva|worm|grub|nematode"),
    ("disease",      r"fung|wilt|blight|\brot\b|rust|mildew|virus|disease|blast|smut|\bspot\b"),
    ("nutrient",     r"fertili[sz]er|urea|\bdap\b|\bnpk\b|nutrient|deficien|zinc|potash|foliar|micronutrient"),
]
ENT = ["fall armyworm","armyworm","fruit fly","fruit borer","stem borer","shoot borer","pod borer","heliothis",
 "helicoverpa","white grub","termite","thrips","whitefly","mealybug","leaf miner","cutworm","weevil","jassid",
 "hopper","caterpillar","nematode","aphid","mite","panama wilt","early blight","late blight","leaf blight",
 "powdery mildew","downy mildew","mildew","blight","wilt","rust","blast","smut","fruit rot","root rot",
 "rhizome rot","rot","leaf curl","mosaic","sigatoka","anthracnose","damping off","gummosis","fungus","fungal",
 "nano urea","nano dap","urea","dap","npk","zinc sulphate","zinc","boron","sulphur","potash","calcium",
 "magnesium","humic acid","micronutrient","vermicompost","gypsum","weedicide","herbicide","insecticide",
 "fungicide","pesticide","weed","drought"]
ENT_NORM = {"aphids": "aphid", "fungal": "fungus", "armyworm": "fall armyworm"}
ENT_RE = re.compile(r"\b(" + "|".join(sorted(ENT, key=len, reverse=True)) + r")s?\b")
CROPS = sorted({x["gold"]["crop"] for x in train if x["gold"]["crop"] != "unknown"}, key=len, reverse=True)
CROP_ALIAS = {"chickpea": ["chickpea", "chick pea", "gram", "chana"], "pigeon pea": ["pigeon", "arhar", "tur"],
              "mung bean": ["green gram", "moong"], "pearl millets": ["bajra", "bajara"], "paddy (dhan)": ["paddy"],
              "rice": ["rice", "paddy"], "okra": ["okra", "bhindi", "bhendi"], "maize": ["maize", "makka"]}

def rules_predict(q):
    ql = q.lower()
    cat = next((c for c, rx in CAT_RULES if re.search(rx, ql)), "nutrient")
    crop = next((c for c in CROPS if any(a in ql for a in CROP_ALIAS.get(c, [c]))), "unknown")
    m = ENT_RE.search(ql)
    ent = ENT_NORM.get(m.group(1), m.group(1)) if m else "none"
    return {"crop": crop, "query_category": cat, "specific_entity": ent}

# ---------- Baseline B: retrieval (nearest training example, char n-grams) ----------
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
Xtr = vec.fit_transform([x["query"].lower() for x in train])
def retrieval_predict_all(rows):
    S = cosine_similarity(vec.transform([x["query"].lower() for x in rows]), Xtr)
    return [train[i]["gold"] for i in S.argmax(axis=1)]

# ---------- evaluation ----------
def evaluate(rows, preds, name):
    df = pd.DataFrame({
        "lang": [r["lang"] for r in rows], "script": [r["script"] for r in rows],
        "g_cat": [r["gold"]["query_category"] for r in rows], "p_cat": [p["query_category"] for p in preds],
        "g_crop": [r["gold"]["crop"] for r in rows], "p_crop": [p["crop"] for p in preds],
        "g_ent": [r["gold"]["specific_entity"] for r in rows], "p_ent": [p["specific_entity"] for p in preds]})
    def row(label, d):
        return {"baseline": name, "group": label, "n": len(d),
                "category_macro_F1": round(f1_score(d.g_cat, d.p_cat, average="macro", zero_division=0), 3),
                "crop_accuracy": round((d.g_crop == d.p_crop).mean(), 3),
                "entity_accuracy": round((d.g_ent == d.p_ent).mean(), 3)}
    out = [row("ALL", df), row("native script", df[df.script == "native"]), row("Roman script", df[df.script == "roman"])]
    for (lang, script), d in df.groupby(["lang", "script"]):
        out.append(row(f"{lang} / {script}", d))
    return out

res = evaluate(test, [rules_predict(x["query"]) for x in test], "Keyword rules")
res += evaluate(test, retrieval_predict_all(test), "Retrieval (nearest train example)")
R = pd.DataFrame(res)
os.makedirs("results", exist_ok=True)
R.to_csv("results/baseline_task1.csv", index=False)
print("Task 1 - baselines without any model (test: 100 items x 7 variants)\n")
print(R.to_string(index=False))