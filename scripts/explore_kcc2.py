import pandas as pd

files = {
    "Bihar": "data/raw/kcc_bihar_2026_jan.csv",
    "Gujarat": "data/raw/kcc_gujarat_2026_jan.csv",
    "Maharashtra": "data/raw/kcc_maharashtra_2026_jan.csv",
}

for state, path in files.items():
    df = pd.read_csv(path)
    print("=" * 60)
    print(state, "rows:", len(df))

    bad = df["QueryText"].fillna("").str.strip().str.upper()
    print("INCOMPLETE CALL:", (bad == "INCOMPLETE CALL").sum())

    print("\nQueryType top 15:")
    print(df["QueryType"].value_counts().head(15))

    print("\nCrop top 10:")
    print(df["Crop"].value_counts().head(10))

    print("\nDuplicates (QueryText + KccAns same):",
          df.duplicated(subset=["QueryText", "KccAns"]).sum())

    labeled = df.dropna(subset=["Crop", "QueryType", "QueryText"])
    print("Rows with crop + type + text:", len(labeled))

    print("\n15 random QueryText:")
    for t in labeled["QueryText"].sample(15, random_state=1):
        print("-", t)