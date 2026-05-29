# Dokumentasi Arsitektur CNN - Tugas Diskusi

## 1. Deskripsi Arsitektur CNN

Arsitektur CNN yang dirancang untuk klasifikasi gambar hewan terdiri dari 3 blok konvolusi, dengan struktur progresif yang meningkat secara bertahap.

```
INPUT (128×128×3)
    ↓
BLOCK 1: Conv2D(32) → Conv2D(32) → MaxPooling → BatchNorm → Dropout
    ↓
BLOCK 2: Conv2D(64) → Conv2D(64) → MaxPooling → BatchNorm → Dropout
    ↓
BLOCK 3: Conv2D(128) → Conv2D(128) → MaxPooling → BatchNorm → Dropout
    ↓
Flatten
    ↓
Dense(256) → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.3)
    ↓
Output (num_classes) → Softmax
```

---

## 2. Komponen Arsitektur CNN yang Digunakan

✅ **Convolution Layer**  
✅ **Activation Function (ReLU)**  
✅ **Pooling Layer (MaxPooling)**  
✅ **Flatten Layer**  
✅ **Dense Layer**  
✅ **Output Layer**  

---

## 3. Penjelasan Pilihan Arsitektur

### A. Jumlah Filter pada Convolution Layer

| Block | Filter | Alasan |
|-------|--------|--------|
| **BLOCK 1** | 32 filter | Mendeteksi fitur dasar (edge, tekstur sederhana) pada input gambar 128×128. Jumlah 32 sudah cukup efisien untuk layer pertama. |
| **BLOCK 2** | 64 filter | Mendeteksi fitur yang lebih kompleks (bentuk geometris). Peningkatan menjadi 64 memungkinkan model mempelajari kombinasi fitur dari layer sebelumnya. |
| **BLOCK 3** | 128 filter | Mendeteksi fitur tingkat tinggi (karakteristik spesifik hewan). Peningkatan menjadi 128 memaksimalkan representasi fitur sebelum Flatten. |

**Alasan peningkatan progresif:** Setiap layer berkurang ukuran spatial (karena pooling), tetapi meningkat kedalaman channel. Ini efisien secara komputasi dan membantu model belajar hierarki fitur.

---

### B. Ukuran Kernel

**Kernel yang digunakan:** `(3, 3)`

**Alasan:**
- **Kernel 3×3 adalah standar industri** untuk CNN modern (AlexNet, VGG, ResNet menggunakan 3×3)
- **Efisiensi komputasi:** Kernel 3×3 lebih cepat dibanding kernel besar seperti 5×5 atau 7×7
- **Receptive field yang tepat:** Untuk gambar 128×128, kernel 3×3 dapat menangkap edge dan tekstur lokal dengan baik
- **Padding 'same':** Mempertahankan dimensi spatial setelah konvolusi, memastikan informasi tepi tidak hilang

---

### C. Jenis Pooling

**Pooling yang digunakan:** `MaxPooling2D(2, 2)`

**Alasan:**
- **Max Pooling:** Mengambil nilai maksimum dalam window 2×2, mempertahankan fitur paling dominan
- **Window 2×2:** Mengurangi dimensi spatial sebesar 50% (128 → 64 → 32 → 16), mengurangi beban komputasi
- **Invariansi translasi:** Max pooling membuat model lebih robust terhadap pergeseran kecil dalam gambar
- **Pemilihan 2×2:** Standard untuk CNN, tidak terlalu agresif (seperti 4×4) sehingga informasi spasial tetap terjaga

---

### D. Fungsi Aktivasi

**Aktivasi yang digunakan:**
- **Hidden Layers:** `ReLU (Rectified Linear Unit)`
- **Output Layer:** `Softmax`

**Alasan ReLU:**
- **ReLU(x) = max(0, x)** mengatasi vanishing gradient problem pada deep networks
- **Komputasi efisien:** Hanya thresholding sederhana dibanding sigmoid/tanh
- **Non-linearity:** Memungkinkan model belajar fungsi kompleks
- **Sparse activation:** Hanya neuron aktif yang berkontribusi, meningkatkan efisiensi

**Alasan Softmax:**
- **Multi-class classification:** Softmax mengonversi output ke probability distribution (total = 1)
- **Interpretasi probabilitas:** Output dapat diinterpretasi sebagai confidence score untuk setiap kelas
- **Kompatibel dengan sparse_categorical_crossentropy loss**

---

### E. Jumlah Neuron pada Output Layer

**Jumlah neuron:** `num_classes` (8 kelas: cane, cavallo, elefante, farfalla, gallina, gatto, mucca)

**Alasan:**
- **Satu neuron per kelas** adalah standar untuk multi-class classification
- **8 output neurons** karena ada 8 kategori hewan dalam dataset
- **Softmax aktivasi** mengubah 8 output menjadi probabilitas yang jumlahnya = 1

---

## 4. Komponen Tambahan

### Batch Normalization
- **Tujuan:** Normalisasi input setiap layer agar training lebih stabil
- **Manfaat:** Mengurangi internal covariate shift, mempercepat konvergensi

### Dropout
**Tingkat dropout yang digunakan:**
| Layer | Dropout Rate | Alasan |
|-------|--------------|--------|
| Conv Block 1-2 | 0.25 | Regularisasi ringan di awal network |
| Conv Block 3 | 0.40 | Regularisasi lebih kuat di layer dalam |
| Dense 1 | 0.50 | Regularisasi agresif untuk mengurangi overfitting pada fully-connected layer |
| Dense 2 | 0.30 | Regularisasi sedang sebelum output |

**Tujuan:** Mencegah overfitting dengan menonaktifkan neuron secara random selama training

---

## 5. Ringkasan Keputusan Desain

| Aspek | Pilihan | Alasan Utama |
|-------|---------|-------------|
| **Jumlah Block** | 3 block | Keseimbangan antara kedalaman network dan efisiensi komputasi |
| **Filter progresif** | 32 → 64 → 128 | Hierarki fitur dari sederhana ke kompleks |
| **Kernel size** | 3×3 | Standard, efisien, receptive field tepat |
| **Pooling** | Max, 2×2 | Mengurangi dimensi sambil mempertahankan fitur penting |
| **Hidden activation** | ReLU | Mengatasi vanishing gradient, komputasi efisien |
| **Output activation** | Softmax | Multi-class probability distribution |
| **Regularisasi** | Batch Norm + Dropout | Meningkatkan generalisasi |

---

## 6. Performa Model

- **Test Accuracy:** Diperoleh dari evaluasi pada validation/test dataset
- **Confusion Matrix:** Menunjukkan distribusi prediksi untuk setiap kelas
- **Training History:** Grafik akurasi dan loss menunjukkan model converging dengan baik

---

## Kesimpulan

Arsitektur CNN yang dirancang ini merupakan **CNN sederhana namun efektif** untuk klasifikasi gambar hewan 128×128 piksel dengan 8 kelas. Setiap keputusan desain dibuat berdasarkan pertimbangan trade-off antara **akurasi**, **efisiensi komputasi**, dan **generalisasi model**.
