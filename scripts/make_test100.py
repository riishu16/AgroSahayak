import pandas as pd

t = pd.read_csv("data/final/test_pool.csv")
t["relabeled_by_regex"] = t["query_type"] == "Disease (Protozoan)"
print(t["category"].value_counts())

# chhoti categories sab rakho, baaki me se sample
small = t[t["category"].isin(["weather", "post_harvest", "market"])]
rest = t.drop(small.index).sample(100 - len(small), random_state=1)
test100 = pd.concat([small, rest]).sample(frac=1, random_state=1).reset_index(drop=True)

test100.insert(0, "id", range(1, len(test100) + 1))
test100 = test100[["id", "state", "query", "crop", "category", "relabeled_by_regex"]]
test100["fix_crop"] = ""
test100["fix_category"] = ""
test100["entity"] = ""
test100["notes"] = ""

test100.to_csv("data/final/test_to_label.csv", index=False, encoding="utf-8-sig")
print("Saved", len(test100), "rows")