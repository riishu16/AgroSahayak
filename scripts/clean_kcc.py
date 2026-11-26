import pandas as pd
import re
import os

files = {
    "Bihar": "data/raw/kcc_bihar_2026_jan.csv",
    "Gujarat": "data/raw/kcc_gujarat_2026_jan.csv",
    "Maharashtra": "data/raw/kcc_maharashtra_2026_jan.csv",
}

# teeno files jodo (index reset zaroori hai, warna error aata hai)
df = pd.concat(
    [pd.read_csv(p).assign(state=s) for s, p in files.items()]
).reset_index(drop=True)
print("Start:", len(df))

df = df.rename(columns={"QueryText": "query", "KccAns": "answer",
                        "Crop": "crop", "QueryType": "query_type"})
df = df[["state", "crop", "query_type", "query", "answer"]]

# 1. khali labels / text hatao
df = df.dropna(subset=["crop", "query_type", "query", "answer"]).reset_index(drop=True)
print("After dropping empty:", len(df))

# 2. bekaar rows hatao
df = df[~df["query"].str.strip().str.upper().isin(["INCOMPLETE CALL", "OTHER STATE CALL"])]
df = df[~df["answer"].str.upper().str.contains("INCOMPLETE CALL|OTHER STATE CALL", regex=True)]
df = df[~df["crop"].isin(["Other", "Others"])]
df = df[df["query_type"] != "NA (Not Applicable)"]
df = df.reset_index(drop=True)
print("After removing junk:", len(df))

# 3. crop ke naam ek jaise karo
df["crop"] = df["crop"].str.strip().str.lower()
crop_map = {
    "bengalgram/chickpea": "chickpea",
    "bengal gram (gram/chick pea/kabuli/chana)": "chickpea",
    "maize (makka)": "maize",
    "pigeon pea (red gram/arhar/tur)": "pigeon pea",
    "pigeon pea/arhar": "pigeon pea",
    "lentil/masoor": "lentil",
    "sesame/sesamum/til": "sesame",
}
df["crop"] = df["crop"].replace(crop_map)

# 4. duplicates hatao (text normalize karke)
def norm(t):
    t = re.sub(r"[^\w\s]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

df["qkey"] = df["query"].map(norm)
before = len(df)
df = df.drop_duplicates(subset=["qkey", "crop", "query_type"]).reset_index(drop=True)
print("Duplicates removed:", before - len(df), "| left:", len(df))

# 5. bahut chhote sawaal hatao
df = df[df["qkey"].str.split().str.len() >= 3].reset_index(drop=True)
print("After dropping very short:", len(df))

print("\nPer state:\n", df["state"].value_counts())
print("\nTop crops:\n", df["crop"].value_counts().head(25))
print("\nQuery types:\n", df["query_type"].value_counts())

print("\n10 rows of 'Disease (Protozoan)':")
sub = df[df["query_type"] == "Disease (Protozoan)"]
for _, r in sub.sample(min(10, len(sub)), random_state=1).iterrows():
    print("-", r["crop"], "|", r["query"])

# save
os.makedirs("data/clean", exist_ok=True)
df.drop(columns="qkey").to_csv("data/clean/kcc_clean.csv", index=False)
print("\nSaved: data/clean/kcc_clean.csv")
df.drop(columns="qkey").to_csv("data/clean/kcc_clean.csv", index=False)