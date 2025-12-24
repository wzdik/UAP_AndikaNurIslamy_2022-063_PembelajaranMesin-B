import pandas as pd
from sklearn.model_selection import train_test_split
from .config import LABEL2ID, SEED

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Kaggle AG News biasanya 3 kolom: Class Index, Title, Description
    if df.shape[1] >= 3:
        # kalau nama kolom tidak sesuai, paksa sesuai posisi
        if "Class Index" not in df.columns:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: "Class Index", cols[1]: "Title", cols[2]: "Description"})
    return df

def load_agnews(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = _standardize_columns(train_df)
    test_df = _standardize_columns(test_df)

    for df in (train_df, test_df):
        df["label"] = df["Class Index"].map(LABEL2ID)
        df["text"] = (df["Title"].astype(str) + " " + df["Description"].astype(str)).str.strip()

    return train_df[["text", "label"]], test_df[["text", "label"]]

def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    # sample per kelas secara proporsional
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    counts = df["label"].value_counts(normalize=True).to_dict()
    parts = []
    for label, frac in counts.items():
        k = max(1, int(round(frac * n)))
        part = df[df["label"] == label].sample(n=min(k, (df["label"] == label).sum()), random_state=seed)
        parts.append(part)

    out = pd.concat(parts).sample(frac=1, random_state=seed)

    # kalau totalnya meleset karena rounding, rapikan
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    elif len(out) < n:
        # tambahkan sisa dari data yang belum kepilih
        need = n - len(out)
        rest = df.drop(out.index, errors="ignore")
        add = rest.sample(n=min(need, len(rest)), random_state=seed)
        out = pd.concat([out, add]).sample(frac=1, random_state=seed)

    return out.reset_index(drop=True)

def make_train_val(df: pd.DataFrame, val_ratio: float, seed: int = SEED):
    tr, va = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=df["label"])
    return tr.reset_index(drop=True), va.reset_index(drop=True)
