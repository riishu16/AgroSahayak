import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/clean/kcc_clean.csv")
print("Start:", len(df))

# 1. query types ko bade groups me milao
cat_map = {
    "Nutrient Deficiency/Excessiveness Management": "nutrient",
    "Nutrient Management": "nutrient",
    "Pathogenic Disease Management": "disease",
    "Disease": "disease",
    "Insect Management": "insect_pest",
    "Plant Protection": "insect_pest",
    "Varietal Selection": "variety_seed",
    "Seed Sowing And Treatment": "variety_seed",
    "Seeds and Planting Material": "variety_seed",
    "Irrigation Management": "irrigation",
    "Water Management": "irrigation",
    "Water Management, Micro Irrigation": "irrigation",
    "Field Preparation": "cultivation",
    "Cultural Practices": "cultivation",
    "Weed Management": "cultivation",
    "Sowing Time and Weather": "cultivation",
    "Cultivation Conditions": "cultivation",
    "Harvesting Management": "cultivation",
    "Weather": "weather",
    "Weather Problems": "weather",
    "Market Information": "market",
}

def map_cat(t):
    if t in cat_map:
        return cat_map[t]
    if t.startswith("Post Harvest"):
        return "post_harvest"
    return None  # baaki rare/ajeeb types

# 2. "Disease (Protozoan)" ko keywords se dobara label karo
def relabel(q):
    q = q.lower()
    if re.search(r"market|price|mandi|\brate\b", q): return "market"
    if re.search(r"weed|herbicide|weedicide", q): return "cultivation"
    if re.search(r"aphid|borer|caterpillar|thrips|whitefly|mite|pest|insect|jassid|hopper|weevil|termite|larva|worm", q): return "insect_pest"
    if re.search(r"fung|wilt|blight|\brot\b|rust|mildew|spot|virus|disease|blast|smut", q): return "disease"
    if re.search(r"fertili|urea|\bdap\b|deficiency|nutrient|zinc|potash|npk", q): return "nutrient"
    return None

def final_cat(row):
    if row["query_type"] == "Disease (Protozoan)":
        return relabel(row["query"])
    return map_cat(row["query_type"])

df["category"] = df.apply(final_cat, axis=1)
n_proto = (df["query_type"] == "Disease (Protozoan)").sum()
print("Disease (Protozoan) rows:", n_proto)
before = len(df)
df = df.dropna(subset=["category"]).reset_index(drop=True)
print("Dropped (label samajh nahi aaya / rare type):", before - len(df), "| left:", len(df))
print("\nCategory counts:\n", df["category"].value_counts())

# 3. test pool alag karo
def norm(t):
    t = re.sub(r"[^\w\s]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

df["qkey"] = df["query"].map(norm)
test = df.sample(150, random_state=42)
train = df.drop(test.index)

# 4. near-duplicate check: test se milte-julte sawaal train se hatao
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(df["qkey"])
sim = cosine_similarity(vec.transform(test["qkey"]), vec.transform(train["qkey"]))
leak = sim.max(axis=0) >= 0.8
print("\nTrain rows removed (test se bahut milte the):", leak.sum())
train = train[~leak]

os.makedirs("data/final", exist_ok=True)
train.drop(columns="qkey").to_csv("data/final/train_pool.csv", index=False)
test.drop(columns="qkey").to_csv("data/final/test_pool.csv", index=False)
print("Train:", len(train), "| Test pool:", len(test))