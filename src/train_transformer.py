# src/train_transformer.py

import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .config import (
    RAW_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR, CLASS_NAMES,
    TRAIN_SAMPLES, MIN_TRAIN_SAMPLES, MAX_TRAIN_SAMPLES,
    VAL_RATIO, SEED, HF_MAX_LEN, HF_EPOCHS, HF_TRAIN_BS, HF_EVAL_BS, HF_LR
)
from .data_utils import load_agnews, stratified_sample, make_train_val
from .eval_utils import save_report, save_confmat, save_curves


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def train_one(model_name: str, out_name: str):
    # ---------- data ----------
    n = int(TRAIN_SAMPLES)
    if n < MIN_TRAIN_SAMPLES or n > MAX_TRAIN_SAMPLES:
        raise ValueError(f"TRAIN_SAMPLES harus antara {MIN_TRAIN_SAMPLES}..{MAX_TRAIN_SAMPLES}")

    train_df, test_df = load_agnews(RAW_DIR / "train.csv", RAW_DIR / "test.csv")
    train_df = stratified_sample(train_df, n=n, seed=SEED)
    tr_df, va_df = make_train_val(train_df, val_ratio=VAL_RATIO, seed=SEED)

    ds_train = Dataset.from_pandas(tr_df)
    ds_val = Dataset.from_pandas(va_df)
    ds_test = Dataset.from_pandas(test_df)

    # ---------- tokenizer ----------
    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=HF_MAX_LEN,
        )

    ds_train = ds_train.map(tok_fn, batched=True)
    ds_val = ds_val.map(tok_fn, batched=True)
    ds_test = ds_test.map(tok_fn, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    ds_train.set_format(type="torch", columns=cols)
    ds_val.set_format(type="torch", columns=cols)
    ds_test.set_format(type="torch", columns=cols)

    # ---------- model ----------
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    out_dir = MODELS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- training args (compat fix) ----------
    common_args = dict(
        output_dir=str(out_dir / "checkpoints"),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        num_train_epochs=HF_EPOCHS,
        per_device_train_batch_size=HF_TRAIN_BS,
        per_device_eval_batch_size=HF_EVAL_BS,
        learning_rate=HF_LR,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=SEED,

        # stabil untuk laptop (opsional tapi aman)
        fp16=False,              # ubah True kalau GPU NVIDIA support
        dataloader_num_workers=0 # windows friendly
    )

    # Beberapa versi transformers memakai `evaluation_strategy`, yang lain `eval_strategy`
    try:
        args = TrainingArguments(**common_args, evaluation_strategy="epoch")
    except TypeError:
        args = TrainingArguments(**common_args, eval_strategy="epoch")

    # ---------- trainer ----------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---------- save final ----------
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    # ---------- test eval ----------
    pred = trainer.predict(ds_test)
    y_true = pred.label_ids.tolist()
    y_pred = np.argmax(pred.predictions, axis=1).tolist()

    # ---------- curves ----------
    log = trainer.state.log_history
    eval_loss = [x["eval_loss"] for x in log if "eval_loss" in x]
    eval_acc = [x.get("eval_accuracy", None) for x in log if "eval_accuracy" in x]
    train_loss = [x["loss"] for x in log if "loss" in x and "epoch" in x and "eval_loss" not in x]

    L = len(eval_loss)
    if len(train_loss) < L and len(train_loss) > 0:
        train_loss = train_loss + [train_loss[-1]] * (L - len(train_loss))
    train_loss = train_loss[:L] if L > 0 else train_loss

    hist = {
        "train_loss": train_loss if train_loss else [np.nan] * L,
        "val_loss": eval_loss,
        "train_acc": [np.nan] * L,
        "val_acc": eval_acc,
    }

    save_curves(hist, FIG_DIR / f"{out_name}_loss_acc.png")
    save_confmat(y_true, y_pred, CLASS_NAMES, FIG_DIR / f"{out_name}_cm.png")
    rep = save_report(y_true, y_pred, CLASS_NAMES, METRICS_DIR / f"{out_name}_report.json")

    summary = {
        "model": model_name,
        "train_samples": n,
        "test_accuracy": float(rep.get("accuracy", 0.0)),
    }
    (METRICS_DIR / f"{out_name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[{out_name}] DONE | test_acc={summary['test_accuracy']:.4f} | saved: {final_dir}")


def main():
    # Model Pretrained 1
    train_one("distilbert-base-uncased", "distilbert")

    # Model Pretrained 2
    train_one("bert-base-uncased", "bert")


if __name__ == "__main__":
    main()
