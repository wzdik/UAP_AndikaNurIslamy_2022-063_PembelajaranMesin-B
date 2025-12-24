<div align="center">

# 🚀 AG News Topic Classification

### *Pertarungan Neural Network vs Transformer*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

*Ujian Akhir Praktikum – Pembelajaran Mesin*

[Fitur](#-fitur-utama) • [Instalasi](#️-cara-menjalankan-secara-lokal) • [Hasil](#-hasil-eksperimen) • [Demo](#-implementasi-website-streamlit)

</div>

---

## 🎯 Deskripsi Proyek

Proyek ini merupakan **Ujian Akhir Praktikum (UAP) Mata Kuliah Pembelajaran Mesin** yang berfokus pada pembangunan, evaluasi, dan implementasi **sistem klasifikasi teks berita** menggunakan dataset **AG News**.

Pada proyek ini dilakukan **perbandingan performa tiga model pembelajaran mesin**, yaitu:

<div align="center">

| 🔷 Model | 📝 Deskripsi | ⚡ Keunggulan |
|---------|-------------|--------------|
| **LSTM** | Neural Network Non-Pretrained | Baseline model, dilatih dari nol |
| **DistilBERT** | Pretrained Transformer Ringan | 40% lebih cepat, efisien |
| **BERT** | Pretrained Transformer Penuh | Akurasi tertinggi, pemahaman konteks terbaik |

</div>

Selain evaluasi kuantitatif, seluruh model diintegrasikan ke dalam **aplikasi web berbasis Streamlit** untuk keperluan demonstrasi dan analisis hasil prediksi secara interaktif.

### 💡 Apa yang Membuat Proyek Ini Menarik?

- ⚔️ **Head-to-Head Battle**: Perbandingan langsung LSTM vs DistilBERT vs BERT
- 🎨 **Visualisasi Menawan**: Confusion matrix dan grafik training yang informatif
- 🌐 **Production-Ready**: Aplikasi web yang siap digunakan untuk demo
- 📊 **Analisis Mendalam**: Evaluasi komprehensif dengan berbagai metrik

---

## 📂 Dataset

Dataset yang digunakan adalah **AG News Dataset**, yang terdiri dari berita berbahasa Inggris dan terbagi ke dalam **4 kelas utama**:

<div align="center">

| 🌍 **World** | ⚽ **Sports** | 💼 **Business** | 🔬 **Sci/Tech** |
|:------------:|:------------:|:---------------:|:---------------:|
| Berita Dunia & Politik | Berita Olahraga | Ekonomi & Bisnis | Sains & Teknologi |

</div>

**📥 Link Dataset:** [Klik di sini untuk download](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download)

Setiap data merupakan gabungan dari **judul dan deskripsi berita**, sehingga cocok untuk tugas klasifikasi teks berbasis konteks.

### 🔄 Tahapan Preprocessing

```
📄 Teks Mentah
    ↓
🔤 Case Folding (lowercase)
    ↓
🧹 Penghapusan Karakter Non-Alfanumerik
    ↓
✂️ Tokenisasi Teks
    ↓
📏 Padding & Truncation
    ↓
🏷️ Encoding Label Kelas
    ↓
✅ Data Siap Digunakan!
```

---

## 🧠 Model yang Digunakan

### 1️⃣ LSTM (The Baseline Champion)

**Arsitektur:**
```
Embedding Layer → LSTM Layers → Dense Layer → Softmax
```

**Karakteristik:**
- ✅ Model neural network yang dilatih dari nol
- ✅ Menggunakan embedding sederhana dan arsitektur LSTM
- ✅ Digunakan sebagai **baseline model** untuk perbandingan
- ⚠️ Membutuhkan epoch lebih banyak untuk konvergensi
- ⚠️ Cenderung overfitting pada data kompleks

---

### 2️⃣ DistilBERT (The Speed Demon)

**Arsitektur:**
```
Pretrained DistilBERT → Classification Head → Fine-tuning
```

**Karakteristik:**
- ✅ Model transformer hasil distilasi dari BERT
- ✅ **40% lebih ringan dan cepat** dibanding BERT
- ✅ Mempertahankan **97% performa** BERT
- ✅ Menghasilkan peningkatan performa signifikan dibanding LSTM
- 🎯 **Sweet spot** antara kecepatan dan akurasi

---

### 3️⃣ BERT (The Powerhouse)

**Arsitektur:**
```
Pretrained BERT-base → Classification Head → Fine-tuning
```

**Karakteristik:**
- ✅ Model transformer penuh dengan 12 layer
- ✅ Mampu menangkap **konteks semantik lebih kompleks**
- ✅ Bidirectional attention mechanism
- ✅ Memberikan **performa terbaik** dalam eksperimen ini
- 🏆 State-of-the-art untuk klasifikasi teks

---

## 📊 Hasil Eksperimen

### 🏆 Leaderboard Performa Model

Evaluasi dilakukan menggunakan **data uji (test set)** dengan metrik:
- ✓ Accuracy
- ✓ Precision
- ✓ Recall
- ✓ F1-score
- ✓ Confusion Matrix
- ✓ Grafik Loss & Accuracy

<div align="center">

| 🥇 Ranking | Model | Akurasi | 📈 Peningkatan | Hasil Analisis |
|:----------:|-------|:-------:|:--------------:|----------------|
| 🥉 **3rd** | **LSTM (Non-Pretrained)** | **79.84%** | *Baseline* | Performa cukup baik sebagai baseline, namun masih sering tertukar pada kelas Business dan World karena keterbatasan pemahaman konteks. |
| 🥈 **2nd** | **DistilBERT (Pretrained)** | **91.09%** | **+11.25%** | Performa meningkat signifikan! Kesalahan klasifikasi berkurang dan model lebih stabil dalam memahami konteks berita. |
| 🥇 **1st** | **BERT (Pretrained)** | **92.33%** | **+12.49%** | 🎯 **JUARA!** Memberikan hasil terbaik dengan akurasi tertinggi dan distribusi prediksi paling seimbang di seluruh kelas. |

</div>

### 🎯 Key Insights

> 💡 **Transfer Learning adalah Game Changer!**  
> Model pretrained (DistilBERT & BERT) memberikan peningkatan akurasi **11-13%** dibanding LSTM yang dilatih dari nol.

> ⚡ **Efisiensi itu Penting!**  
> DistilBERT menawarkan trade-off sempurna: **91% akurasi** dengan waktu training **40% lebih cepat** dari BERT.

> 🧠 **Konteks adalah Kunci!**  
> BERT unggul dalam memahami nuansa konteks, terutama pada berita dengan topik yang saling tumpang tindih.

---

## 🧩 Confusion Matrix

Confusion Matrix digunakan untuk menganalisis kesalahan klasifikasi antar kelas dan memahami pola error setiap model.

<table>
<tr>
<td width="33%" align="center">

### 🔷 LSTM
![Confusion Matrix LSTM](reports/figures/lstm_cm.png)
*Masih tertukar antara Business & Sci/Tech*

</td>
<td width="33%" align="center">

### 🔶 DistilBERT
![Confusion Matrix DistilBERT](reports/figures/distilbert_cm.png)
*Kesalahan jauh berkurang!*

</td>
<td width="33%" align="center">

### 🔷 BERT
![Confusion Matrix BERT](reports/figures/bert_cm.png)
*Prediksi paling konsisten! 🏆*

</td>
</tr>
</table>

**📊 Analisis:**
- **LSTM**: Masih sering tertukar antara *Business* dan *Sci/Tech* karena keterbatasan pemahaman konteks
- **DistilBERT**: Kesalahan klasifikasi berkurang drastis dengan transfer learning
- **BERT**: Prediksi paling akurat dan konsisten di semua kategori

---

## 📈 Kurva Training (Loss & Accuracy)

Grafik training menunjukkan proses pembelajaran setiap model dan mengidentifikasi potensi overfitting atau underfitting.

<table>
<tr>
<td width="33%" align="center">

### 🔷 LSTM
![Loss & Accuracy LSTM](reports/figures/lstm_loss_acc.png)
*Konvergensi lambat, tanda overfitting*

</td>
<td width="33%" align="center">

### 🔶 DistilBERT
![Loss & Accuracy DistilBERT](reports/figures/distilbert_loss_acc.png)
*Konvergensi cepat & stabil! ⚡*

</td>
<td width="33%" align="center">

### 🔷 BERT
![Loss & Accuracy BERT](reports/figures/bert_loss_acc.png)
*Smooth learning curve! 🎯*

</td>
</tr>
</table>

**📊 Observasi:**
- **LSTM**: Membutuhkan epoch lebih banyak dan cenderung overfitting
- **DistilBERT & BERT**: Konvergensi lebih cepat dengan learning curve yang smooth
- **Model Pretrained**: Lebih stabil selama proses training berkat transfer learning

---

## 🌐 Implementasi Website (Streamlit)

Sistem website sederhana dibangun menggunakan **Streamlit** dengan antarmuka yang intuitif dan user-friendly.

### ✨ Fitur Unggulan

<div align="center">

| 🎯 Fitur | 📝 Deskripsi |
|---------|-------------|
| 📝 **Input Interaktif** | Input teks berita (judul + deskripsi) dengan text area |
| 🤖 **Multi-Model** | Pemilihan model (LSTM / DistilBERT / BERT) via dropdown |
| 🎯 **Prediksi Real-time** | Output prediksi kelas beserta probabilitas secara instant |
| 📊 **Visualisasi Lengkap** | Tampilan metrik evaluasi, confusion matrix, dan grafik training |
| 🎨 **UI Modern** | Interface yang clean, responsive, dan mudah digunakan |

</div>

### 🖼️ Preview Aplikasi

```
╔════════════════════════════════════════════════╗
║  🎯 AG News Classifier - Streamlit App        ║
╠════════════════════════════════════════════════╣
║                                                ║
║  📝 Masukkan Teks Berita:                     ║
║  ┌──────────────────────────────────────────┐ ║
║  │ Apple unveils revolutionary new iPhone   │ ║
║  │ with advanced AI capabilities...         │ ║
║  └──────────────────────────────────────────┘ ║
║                                                ║
║  🤖 Pilih Model: [BERT ▼]                    ║
║                                                ║
║  [🚀 Klasifikasi Sekarang]                    ║
║                                                ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║                                                ║
║  ✅ Hasil Prediksi:                           ║
║  📰 Kategori: Sci/Tech                        ║
║  📊 Confidence: 98.5%                         ║
║                                                ║
╚════════════════════════════════════════════════╝
```

### 🔗 Live Demo

👉 **Streamlit App**: Coming Soon, udah jadi kok

*Aplikasi akan segera di-deploy untuk demonstrasi publik!*

---

## ▶️ Cara Menjalankan Secara Lokal

### 📋 Prerequisites

```bash
✓ Python 3.8 atau lebih tinggi
✓ pip (Python package manager)
✓ CUDA (opsional, untuk akselerasi GPU)
```

### 🚀 Langkah Instalasi

```bash
# 1️⃣ Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# 2️⃣ Buat virtual environment (recommended)
python -m venv venv

# Aktivasi virtual environment:
# Untuk Linux/Mac:
source venv/bin/activate
# Untuk Windows:
venv\Scripts\activate

# 3️⃣ Install semua dependency
pip install -r requirements.txt

# 4️⃣ Download dataset
# Letakkan dataset di folder data/raw/

# 5️⃣ (Opsional) Train model dari awal
python src/train_lstm.py
python src/train_transformer.py --model distilbert
python src/train_transformer.py --model bert

# 6️⃣ Jalankan aplikasi Streamlit
streamlit run src/app.py
```

### 📦 Dependencies Utama

```
torch>=2.0.0              # Deep Learning framework
transformers>=4.30.0      # Hugging Face Transformers
streamlit>=1.28.0         # Web app framework
pandas>=1.5.0             # Data manipulation
numpy>=1.24.0             # Numerical computing
scikit-learn>=1.3.0       # ML utilities
matplotlib>=3.7.0         # Plotting
seaborn>=0.12.0           # Statistical visualization
```

---

## 📁 Struktur Repository

```
UAP/
│
├── 📂 src/                          # Kode sumber utama
│   ├── app.py                       # Aplikasi Streamlit
│   ├── train_lstm.py                # Script training LSTM
│   ├── train_transformer.py         # Script training Transformer
│   └── data_utils.py                # Utility preprocessing data
│
├── 📂 data/
│   └── raw/                         # Dataset mentah AG News
│
├── 📂 models/                       # Model yang sudah dilatih
│   ├── lstm/                        # Model LSTM tersimpan
│   ├── distilbert/                  # Model DistilBERT tersimpan
│   └── bert/                        # Model BERT tersimpan
│
├── 📂 reports/
│   ├── metrics/                     # Metrik evaluasi (JSON/CSV)
│   └── figures/                     # Visualisasi (confusion matrix, grafik)
│
├── 📂 notebooks/                    # Jupyter notebooks untuk eksplorasi
│
├── 📄 requirements.txt              # Daftar dependencies Python
├── 📄 README.md                     # Dokumentasi proyek (file ini!)
└── 📄 LICENSE                       # Lisensi proyek
```

---

## 🔬 Metodologi

### 🎯 Training Strategy

<div align="center">

| Parameter | LSTM | DistilBERT | BERT |
|-----------|:----:|:----------:|:----:|
| **Epochs** | 20 | 5 | 5 |
| **Batch Size** | 32 | 16 | 16 |
| **Learning Rate** | 0.001 | 2e-5 | 2e-5 |
| **Max Sequence Length** | 200 | 128 | 128 |
| **Embedding Dimension** | 100 | 768 | 768 |
| **Optimizer** | Adam | AdamW | AdamW |

</div>

### 📊 Evaluasi

- **Data Split**: 80% training, 20% testing
- **Loss Function**: CrossEntropyLoss
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Validation**: Confusion Matrix & Classification Report

---

## 🎓 Kesimpulan

### 📌 Rangkuman Hasil

Berdasarkan hasil eksperimen yang telah dilakukan:

1. **🏆 Model Pretrained Menang Telak**: DistilBERT dan BERT secara konsisten mengungguli model non-pretrained dengan margin yang signifikan (**+11-13% akurasi**)

2. **⚡ DistilBERT = Sweet Spot**: Menawarkan keseimbangan sempurna antara performa dan efisiensi - **91% akurasi** dengan **40% lebih cepat**

3. **🎯 BERT untuk Akurasi Maksimal**: BERT memberikan performa terbaik dengan **92.33% akurasi** dan prediksi paling konsisten

4. **🌐 Implementasi Praktis**: Streamlit mempermudah proses evaluasi dan demonstrasi model secara interaktif

### 💡 Pembelajaran Penting

> **"Transfer Learning mengubah permainan dalam NLP"**
> 
> Proyek ini membuktikan bahwa pemanfaatan pretrained transformer sangat efektif untuk tugas klasifikasi teks dibandingkan model neural network konvensional yang dilatih dari nol.

### 🚀 Pengembangan Selanjutnya

- [ ] Implementasi ensemble learning dari ketiga model
- [ ] Penambahan fitur multi-bahasa (Indonesia, dll)
- [ ] Integrasi dengan real-time news scraping
- [ ] Deployment ke cloud platform (Heroku/AWS)
- [ ] Optimasi model untuk mobile deployment
- [ ] API REST dengan FastAPI

---

## 🙏 Acknowledgments

Terima kasih kepada:

- 📚 **AG News Dataset** dari Kaggle
- 🤗 **Hugging Face** untuk library Transformers
- 🔥 **PyTorch Team** untuk framework deep learning
- 🎨 **Streamlit** untuk framework web app
- 👨‍🏫 **Dosen Pembimbing** Mata Kuliah Pembelajaran Mesin
- 🎓 **Universitas Muhammadiyah Malang**

---

## 👤 Identitas

<div align="center">

### **Andika Nur Islamy**

🎓 **Program Studi**: Informatika  
🏫 **Universitas**: Universitas Muhammadiyah Malang  
🆔 **NIM**: 202210370311063

---

📧 Email: [andika.nurislamy@gmail.com](andika.nurislamy@gmail.com)  
🐙 GitHub: [github.com/wzdik](https://github.com/wzdik)

---

### ⭐ **Jika proyek ini bermanfaat, jangan lupa beri Star!** ⭐

</div>

---

<div align="center">

**by Andika Nur Islamy**

*"Dalam era informasi, kemampuan mengklasifikasi adalah kemampuan memahami."*

---

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=username.ag-news-classification)
[![GitHub Stars](https://img.shields.io/github/stars/username/repo?style=social)](https://github.com/wzdik/UAP_AndikaNurIslamy_2022-063_PembelajaranMesin-B.git)
[![GitHub Forks](https://img.shields.io/github/forks/username/repo?style=social)](https://github.com/username/repo/network/members)

**© 2025 Andika Nur Islamy | UAP Pembelajaran Mesin B**

</div>