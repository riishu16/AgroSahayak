import pandas as pd

t = pd.read_csv("data/translated/translated_queries.csv")
print("Rows:", len(t))
print("Khali:\n", t[["hi", "gu", "mr"]].isna().sum())
print("Hindi me Devanagari nahi:", (~t["hi"].str.contains(r"[\u0900-\u097F]", na=False)).sum())
print("Gujarati me Gujarati script nahi:", (~t["gu"].str.contains(r"[\u0A80-\u0AFF]", na=False)).sum())
print("Marathi me Devanagari nahi:", (~t["mr"].str.contains(r"[\u0900-\u097F]", na=False)).sum())