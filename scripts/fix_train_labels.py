import pandas as pd
import re

df = pd.read_csv("data/final/train_pool.csv")
q = df["query"].str.lower()

rules = {
    "weather":      r"weather|rainfall|\brain\b|temperature|forecast",
    "market":       r"market|mandi|\bprice|\bmsp\b",
    "variety_seed": r"variet|seed rate|seed treatment|\bseeds?\b|hybrid",
    "irrigation":   r"irrigat|water management|\bdrip\b|sprinkler",
    "cultivation":  r"weed|herbicide|weedicide|sowing|land preparation|field preparation|spacing|transplant|harvest|cultivation practice",
    "insect_pest":  r"aphid|borer|caterpillar|thrips|whitefly|mite|\bpest|insect|jassid|hopper|weevil|termite|larva|worm|grub|nematode",
    "disease":      r"fung|wilt|blight|\brot\b|rust|mildew|virus|disease|blast|smut|\bspot\b",
    "nutrient":     r"fertili[sz]er|urea|\bdap\b|\bnpk\b|nutrient|deficien|zinc|potash|foliar|micronutrient",
}

hits = pd.DataFrame({k: q.str.contains(v, regex=True) for k, v in rules.items()})
n_hits = hits.sum(axis=1)
single = n_hits == 1
new_cat = hits.idxmax(axis=1)

old = df["category"].copy()
df.loc[single, "category"] = new_cat[single]
changed = df["category"] != old

print("Total rows:", len(df))
print("Ek hi category ke keywords mile:", int(single.sum()))
print("Label badla:", int(changed.sum()))
print("\nBadlav (pehle -> ab):")
print(pd.crosstab(old[changed], df.loc[changed, "category"]).to_string())
print("\n25 badli hui rows (khud padh ke check karo):")
print(pd.concat([df.loc[changed, "query"], old[changed].rename("pehle"),
                 df.loc[changed, "category"].rename("ab")], axis=1)
      .sample(min(25, int(changed.sum())), random_state=1).to_string())
print("\nNayi category counts:\n", df["category"].value_counts().to_string())

df.to_csv("data/final/train_fixed.csv", index=False)
print("\nSaved data/final/train_fixed.csv")