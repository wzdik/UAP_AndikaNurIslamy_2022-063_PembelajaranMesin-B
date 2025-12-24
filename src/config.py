from pathlib import Path

# Root project = folder yang berisi requirements.txt
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for p in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIG_DIR, METRICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# AG News label umumnya: 1..4
LABEL2ID = {1: 0, 2: 1, 3: 2, 4: 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Ketentuan dataset untuk training (kamu minta 5000..10000)
MIN_TRAIN_SAMPLES = 5000
MAX_TRAIN_SAMPLES = 10000
TRAIN_SAMPLES = 10000  # default: 10k, nanti bisa kamu ubah

VAL_RATIO = 0.1
SEED = 42

# Training settings (lokal-friendly)
LSTM_EPOCHS = 5
LSTM_BATCH = 64
LSTM_MAX_LEN = 200

HF_MAX_LEN = 128
HF_EPOCHS = 2
HF_TRAIN_BS = 16
HF_EVAL_BS = 32
HF_LR = 2e-5
