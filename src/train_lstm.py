import re
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import (
    RAW_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR, CLASS_NAMES,
    TRAIN_SAMPLES, MIN_TRAIN_SAMPLES, MAX_TRAIN_SAMPLES,
    VAL_RATIO, SEED, LSTM_EPOCHS, LSTM_BATCH, LSTM_MAX_LEN
)
from .data_utils import load_agnews, stratified_sample, make_train_val
from .eval_utils import save_report, save_confmat, save_curves

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_vocab(texts, min_freq=2):
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(clean_text(t).split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, f in c.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

class TextDS(Dataset):
    def __init__(self, texts, labels, vocab, max_len=LSTM_MAX_LEN):
        self.texts = [clean_text(t) for t in texts]
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()[:self.max_len]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=128, n_classes=4, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, n_classes)

    def forward(self, x):
        e = self.emb(x)
        out, (h, c) = self.lstm(e)
        hcat = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.drop(hcat))

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    ys, ps = [], []
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
        pred = logits.argmax(1)
        ys.extend(yb.cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return float(np.mean(losses)), correct/total, ys, ps

def main():
    n = int(TRAIN_SAMPLES)
    if n < MIN_TRAIN_SAMPLES or n > MAX_TRAIN_SAMPLES:
        raise ValueError(f"TRAIN_SAMPLES harus antara {MIN_TRAIN_SAMPLES}..{MAX_TRAIN_SAMPLES}")

    train_df, test_df = load_agnews(RAW_DIR/"train.csv", RAW_DIR/"test.csv")
    train_df = stratified_sample(train_df, n=n, seed=SEED)
    tr_df, va_df = make_train_val(train_df, val_ratio=VAL_RATIO, seed=SEED)

    vocab = build_vocab(tr_df["text"].tolist(), min_freq=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMClassifier(vocab_size=len(vocab)).to(device)

    tr_ds = TextDS(tr_df["text"].tolist(), tr_df["label"].tolist(), vocab)
    va_ds = TextDS(va_df["text"].tolist(), va_df["label"].tolist(), vocab)
    te_ds = TextDS(test_df["text"].tolist(), test_df["label"].tolist(), vocab)

    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=LSTM_BATCH, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    out_dir = MODELS_DIR / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    vocab_path = out_dir / "vocab.joblib"

    for ep in range(1, LSTM_EPOCHS + 1):
        model.train()
        losses = []
        correct, total = 0, 0

        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        tr_loss = float(np.mean(losses))
        tr_acc = correct / total
        va_loss, va_acc, _, _ = eval_loader(model, va_ld, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), model_path)
            joblib.dump(vocab, vocab_path)

        print(f"[LSTM] Epoch {ep}/{LSTM_EPOCHS} | train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

    # Load best
    model.load_state_dict(torch.load(model_path, map_location=device))
    te_loss, te_acc, y_true, y_pred = eval_loader(model, te_ld, device)

    save_curves(history, FIG_DIR / "lstm_loss_acc.png")
    save_confmat(y_true, y_pred, CLASS_NAMES, FIG_DIR / "lstm_cm.png")
    save_report(y_true, y_pred, CLASS_NAMES, METRICS_DIR / "lstm_report.json")

    (METRICS_DIR / "lstm_summary.json").write_text(
        json.dumps({"test_acc": te_acc, "train_samples": n}, indent=2), encoding="utf-8"
    )

    print(f"[LSTM] TEST ACC = {te_acc:.4f}")

if __name__ == "__main__":
    main()
