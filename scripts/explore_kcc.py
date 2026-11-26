import pandas as pd

files = {
    "Bihar": "data/raw/kcc_bihar_2026_jan.csv",
    "Gujarat": "data/raw/kcc_gujarat_2026_jan.csv",
    "Maharashtra": "data/raw/kcc_maharashtra_2026_jan.csv",
}

for state, path in files.items():
    print("=" * 60)
    print(state)
    df = pd.read_csv(path)
    print("Rows, Columns:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Exact duplicates:", df.duplicated().sum())
    print("Khali values:")
    print(df.isnull().sum())
    print(df.head(3).to_string())