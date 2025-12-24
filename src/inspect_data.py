import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import RAW_DIR, CLASS_NAMES, LABEL2ID
from src.data_utils import load_agnews

def _tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def top_keywords_per_class(train_df: pd.DataFrame, topk: int = 25):
    # TF-IDF per class: ambil kata yang paling "mewakili" kelas tsb
    vectorizer = TfidfVectorizer(
        preprocessor=_tokenize,
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3
    )
    X = vectorizer.fit_transform(train_df["text"].astype(str))
    vocab = np.array(vectorizer.get_feature_names_out())

    results = {}
    for c in range(4):
        idx = (train_df["label"].values == c)
        # rata-rata tfidf dalam kelas
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = np.argsort(mean_tfidf)[-topk:][::-1]
        results[CLASS_NAMES[c]] = vocab[top_idx].tolist()
    return results

def sample_texts_by_class(train_df: pd.DataFrame, n_each: int = 5, seed: int = 42):
    out = {}
    rng = np.random.default_rng(seed)
    for c in range(4):
        dfc = train_df[train_df["label"] == c]
        pick = dfc.sample(n=min(n_each, len(dfc)), random_state=seed)["text"].tolist()
        out[CLASS_NAMES[c]] = pick
    return out

def main():
    train_df, test_df = load_agnews(RAW_DIR/"train.csv", RAW_DIR/"test.csv")

    # contoh berita yang bisa kamu copy-paste ke streamlit
    samples = sample_texts_by_class(train_df, n_each=5)
    print("=== CONTOH TEKS PER KELAS (copy-paste untuk Streamlit) ===")
    for k, texts in samples.items():
        print(f"\n[{k}]")
        for i, t in enumerate(texts, 1):
            print(f"{i}. {t[:250]}")

    # kata/frasa yang paling representatif per kelas
    keywords = top_keywords_per_class(train_df, topk=20)
    print("\n\n=== TOP KEYWORDS PER KELAS (TF-IDF) ===")
    for k, kw in keywords.items():
        print(f"\n[{k}]")
        print(", ".join(kw))

if __name__ == "__main__":
    main()
