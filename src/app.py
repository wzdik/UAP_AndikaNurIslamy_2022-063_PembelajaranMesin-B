# src/app.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # folder UAP/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import json
import numpy as np
import streamlit as st
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODELS_DIR, FIG_DIR, METRICS_DIR, CLASS_NAMES, RAW_DIR, LSTM_MAX_LEN
from src.train_lstm import LSTMClassifier, clean_text
from src.data_utils import load_agnews


# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="AG News Classifier", page_icon="📰", layout="wide")

CSS = """
<style>
:root { --card: rgba(255,255,255,0.06); --border: rgba(255,255,255,0.10); }
body { background: radial-gradient(1200px 800px at 10% 10%, rgba(79,70,229,0.25), transparent 45%),
                   radial-gradient(900px 700px at 90% 20%, rgba(16,185,129,0.18), transparent 40%),
                   radial-gradient(1000px 700px at 30% 90%, rgba(236,72,153,0.14), transparent 45%); }
.block-container { padding-top: 1.2rem; }
.hero {
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  padding: 18px 18px;
  border-radius: 18px;
}
.hero h1 { margin: 0; font-size: 28px; }
.hero p { margin: 6px 0 0 0; opacity: 0.9; }
.card {
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 18px;
  padding: 14px 14px;
}
.badge {
  display: inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--border); background: rgba(255,255,255,0.05);
  font-size: 12px; opacity: 0.95;
}
.small { opacity: 0.85; font-size: 12px; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0; }
kbd { border: 1px solid rgba(255,255,255,0.18); padding: 2px 6px; border-radius: 6px; background: rgba(0,0,0,0.10); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    ez = np.exp(z)
    return ez / (ez.sum() + 1e-12)

def _basic_clean_for_tfidf(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# DATA INSPECT (contoh & keyword)
# =========================
@st.cache_data
def load_train_df():
    train_df, _ = load_agnews(RAW_DIR / "train.csv", RAW_DIR / "test.csv")
    return train_df

@st.cache_data
def build_keywords_topk(topk: int = 15):
    """
    Top keywords per kelas (TF-IDF).
    Ini bukan 'explain model', tapi cukup untuk bantu kamu tahu
    jenis kata/tema yang umum di kelas tsb.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as _np

    df = load_train_df()
    vec = TfidfVectorizer(
        preprocessor=_basic_clean_for_tfidf,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=30000,
    )
    X = vec.fit_transform(df["text"].astype(str))
    vocab = _np.array(vec.get_feature_names_out())

    out = {}
    for i, cname in enumerate(CLASS_NAMES):
        idx = (df["label"].values == i)
        mean = X[idx].mean(axis=0).A1
        top_idx = _np.argsort(mean)[-topk:][::-1]
        out[cname] = vocab[top_idx].tolist()
    return out


# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_lstm():
    vocab_path = MODELS_DIR / "lstm" / "vocab.joblib"
    model_path = MODELS_DIR / "lstm" / "model.pt"
    if not vocab_path.exists() or not model_path.exists():
        return None, None

    vocab = joblib.load(vocab_path)
    model = LSTMClassifier(vocab_size=len(vocab))

    # load state dict lebih aman (hilangkan warning future)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state)
    model.eval()
    return model, vocab

@st.cache_resource
def load_hf(which: str):
    final_dir = MODELS_DIR / which / "final"
    if not final_dir.exists():
        return None, None
    tok = AutoTokenizer.from_pretrained(final_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(final_dir)
    mdl.eval()
    return tok, mdl


# =========================
# PREDICT
# =========================
def predict_lstm(text: str):
    model, vocab = load_lstm()
    if model is None:
        return None, None

    max_len = int(LSTM_MAX_LEN) if int(LSTM_MAX_LEN) > 0 else 200
    tokens = clean_text(text).split()[:max_len]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))

    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x).cpu().numpy().reshape(-1)

    probs = softmax_np(logits)
    pred = int(np.argmax(probs))
    return pred, probs

def predict_hf(text: str, which: str, max_len: int = 128):
    tok, mdl = load_hf(which)
    if tok is None:
        return None, None

    enc = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = mdl(**enc).logits.cpu().numpy().reshape(-1)

    probs = softmax_np(logits)
    pred = int(np.argmax(probs))
    return pred, probs


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
  <span class="badge">UAP • Text Classification</span>
  <span class="badge">AG News</span>
  <span class="badge">3 Models: LSTM vs DistilBERT vs BERT</span>
  <h1>📰 AG News Topic Classifier</h1>
  <p>Masukkan teks berita (judul + deskripsi). Pilih model, lalu lihat prediksi + probabilitas + evaluasi.</p>
</div>
""", unsafe_allow_html=True)

st.write("")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    model_choice = st.radio(
        "Pilih model",
        ["LSTM (Non-Pretrained)", "DistilBERT (Pretrained)", "BERT (Pretrained)"]
    )

    st.markdown("---")
    st.markdown("### 📌 Contoh teks cepat")
    if st.button("Contoh: Sports"):
        st.session_state["sample"] = "Team wins championship after dramatic final match and fans celebrate across the city."
    if st.button("Contoh: Business"):
        st.session_state["sample"] = "Company shares surge after strong quarterly earnings and new product expansion plan."
    if st.button("Contoh: Sci/Tech"):
        st.session_state["sample"] = "Researchers develop new AI model improving medical diagnosis with higher accuracy."
    if st.button("Contoh: World"):
        st.session_state["sample"] = "Leaders meet to discuss new peace agreement and humanitarian aid for conflict region."

    st.markdown("---")
    st.markdown("### 🎲 Ambil contoh dari dataset (AG News)")
    st.caption("Ini cara paling gampang biar kamu tahu input yang sesuai dataset.")
    try:
        train_df = load_train_df()
        pick_class = st.selectbox("Pilih kelas", CLASS_NAMES, index=0)
        if st.button("Ambil 1 contoh random dari train"):
            c = CLASS_NAMES.index(pick_class)
            row = train_df[train_df["label"] == c].sample(1).iloc[0]
            st.session_state["sample"] = str(row["text"])
    except Exception as e:
        st.warning("Dataset belum terbaca. Pastikan data/raw/train.csv & test.csv ada.")

    st.markdown("---")
    st.markdown("### 🔎 Top keywords per kelas")
    st.caption("Bukan wajib, tapi membantu memahami 'kata-kata khas' tiap topik.")
    try:
        kw = build_keywords_topk(topk=12)
        show_class = st.selectbox("Lihat keyword kelas", CLASS_NAMES, index=0, key="kw_class")
        st.write(", ".join(kw[show_class]))
    except Exception:
        st.info("Keyword belum tersedia (butuh scikit-learn). Install: pip install scikit-learn")


default_text = st.session_state.get("sample", "")


# =========================
# MAIN LAYOUT
# =========================
c1, c2 = st.columns([1.2, 1.0], gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ✍️ Input Teks")

    st.caption("Gunakan 1–2 kalimat (judul + deskripsi). Kata tunggal seperti 'plane' biasanya ambigu.")
    text = st.text_area(
        "Teks berita",
        value=default_text,
        height=180,
        placeholder="Tempel judul + deskripsi berita di sini..."
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run = st.button("🔮 Prediksi", use_container_width=True)
    with col_btn2:
        clear = st.button("🧹 Bersihkan", use_container_width=True)
        if clear:
            st.session_state["sample"] = ""
            st.rerun()

    st.markdown(
        '<div class="small">Tips: semakin jelas konteks (judul+deskripsi), prediksi makin stabil.</div>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧾 Output")

    if run:
        if not text.strip():
            st.warning("Isi teks dulu.")
        else:
            if model_choice.startswith("LSTM"):
                pred, probs = predict_lstm(text)
                key = "lstm"
            elif model_choice.startswith("DistilBERT"):
                pred, probs = predict_hf(text, "distilbert", max_len=128)
                key = "distilbert"
            else:
                pred, probs = predict_hf(text, "bert", max_len=128)
                key = "bert"

            if pred is None:
                st.error("Model belum ada. Jalankan training dulu (train_lstm / train_transformer).")
            else:
                st.success(f"Prediksi: **{CLASS_NAMES[pred]}**")

                st.write("Probabilitas per kelas:")
                st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(4)})

                # ringkasan evaluasi
                summ = load_json(METRICS_DIR / f"{key}_summary.json")
                st.markdown("**Ringkas Evaluasi (Test):**")
                if summ:
                    # kompatibel untuk dua format: test_accuracy atau test_acc
                    acc = summ.get("test_accuracy", summ.get("test_acc", None))
                    st.write(f"- Accuracy: **{acc}**")
                    st.write(f"- Train samples: **{summ.get('train_samples','-')}**")
                else:
                    st.write("- (summary belum ada)")

                # interpretasi singkat (biar kamu paham cara baca)
                top2 = np.argsort(probs)[-2:][::-1]
                st.markdown("**Interpretasi cepat:**")
                st.write(
                    f"- Model paling yakin ke **{CLASS_NAMES[top2[0]]}** ({probs[top2[0]]:.2f}). "
                    f"Alternatif terdekat: **{CLASS_NAMES[top2[1]]}** ({probs[top2[1]]:.2f})."
                )

    else:
        st.info("Klik **Prediksi** setelah mengisi teks.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# REPORTS SECTION
# =========================
st.write("")
tab1, tab2, tab3 = st.tabs(["📊 Metrik", "🧩 Confusion Matrix", "📈 Kurva Training"])

with tab1:
    st.markdown("### 📊 Classification Report (JSON)")
    st.caption("Ini yang kamu pakai untuk membandingkan 3 model (LSTM vs DistilBERT vs BERT).")
    colA, colB, colC = st.columns(3)

    for col, key, title in [
        (colA, "lstm", "LSTM"),
        (colB, "distilbert", "DistilBERT"),
        (colC, "bert", "BERT"),
    ]:
        with col:
            rep = load_json(METRICS_DIR / f"{key}_report.json")
            st.markdown(f"**{title}**")
            if rep:
                acc = rep.get("accuracy", None)
                if isinstance(acc, (float, int)):
                    st.write(f"Accuracy: **{acc:.4f}**")
                else:
                    st.write(f"Accuracy: **{acc}**")

                st.write("Macro F1:", rep.get("macro avg", {}).get("f1-score", "-"))
                st.write("Weighted F1:", rep.get("weighted avg", {}).get("f1-score", "-"))
            else:
                st.warning("Belum ada report. Jalankan training dulu.")

with tab2:
    st.markdown("### 🧩 Confusion Matrix")
    st.caption("Ini untuk menjelaskan kelas mana yang sering tertukar.")
    colA, colB, colC = st.columns(3)

    for col, key, title in [
        (colA, "lstm", "LSTM"),
        (colB, "distilbert", "DistilBERT"),
        (colC, "bert", "BERT"),
    ]:
        with col:
            img = FIG_DIR / f"{key}_cm.png"
            st.markdown(f"**{title}**")
            if img.exists():
                st.image(str(img), use_container_width=True)
            else:
                st.warning("Belum ada gambar CM.")

with tab3:
    st.markdown("### 📈 Loss & Accuracy Curves")
    st.caption("Kurva untuk menunjukkan training stabil / overfitting.")
    colA, colB, colC = st.columns(3)

    for col, key, title in [
        (colA, "lstm", "LSTM"),
        (colB, "distilbert", "DistilBERT"),
        (colC, "bert", "BERT"),
    ]:
        with col:
            img = FIG_DIR / f"{key}_loss_acc.png"
            st.markdown(f"**{title}**")
            if img.exists():
                st.image(str(img), use_container_width=True)
            else:
                st.warning("Belum ada kurva.")
