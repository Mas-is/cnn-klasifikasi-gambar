# CNN Klasifikasi Gambar

Project ini adalah model klasifikasi gambar hewan menggunakan TensorFlow dan Keras.

## Isi project

- `train_cnn.py` : skrip untuk melatih model CNN dari folder `dataset/`
- `cnn_model.h5` : model hasil pelatihan
- `predict.py` : skrip untuk memprediksi kelas gambar tunggal
- `dataset/` : kumpulan gambar terorganisir berdasarkan kelas
- `training_history.png` : grafik akurasi dan loss
- `confusion_matrix.png` : heatmap confusion matrix
- `requirements.txt` : daftar paket Python yang dibutuhkan

## Cara menjalankan

1. Siapkan environment Python

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Jalankan pelatihan model

```bash
python train_cnn.py
```

- Model akan disimpan ke `cnn_model.h5`
- Grafik training akan disimpan ke `training_history.png`
- Confusion matrix akan disimpan ke `confusion_matrix.png`

3. Jalankan prediksi gambar

```bash
python predict.py dataset/gatto/1.jpeg
```

Jika Anda ingin menggunakan model yang sudah ada, cukup jalankan `predict.py` setelah `cnn_model.h5` tersedia.

## Catatan

- Dataset harus berada di dalam folder `dataset/` dengan struktur subfolder per kelas.
- Model menggunakan input gambar ukuran `128x128`.
