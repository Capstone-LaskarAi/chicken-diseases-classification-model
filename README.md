# 📌 Project Title: Chicken Disease Classification Based on Fecal Images

## 🗭 Domain Proyek

### Latar Belakang

Penyakit unggas seperti Coccidiosis, Newcastle Disease, dan Salmonella menjadi tantangan besar bagi peternak skala kecil dan menengah. Penyakit ini dapat menyebar dengan cepat dan menimbulkan kerugian ekonomi signifikan jika tidak terdeteksi sejak dini. Sayangnya, deteksi penyakit masih sangat bergantung pada kehadiran tenaga ahli dan pemeriksaan manual—yang belum tentu tersedia di daerah terpencil.

Salah satu pendekatan inovatif adalah memanfaatkan teknologi **computer vision** untuk menganalisis **gambar kotoran ayam**, yang memiliki ciri visual khas sesuai jenis penyakitnya. Dataset yang digunakan dikumpulkan di Tanzania melalui aplikasi Open Data Kit (ODK) dari September 2020 hingga Februari 2021. Gambar difoto menggunakan kamera ponsel dan telah diresize ke ukuran 224x224 piksel.

### Urgensi Permasalahan

- Menyediakan solusi deteksi dini penyakit unggas berbasis AI yang murah dan mudah digunakan.
- Meningkatkan efisiensi dan produktivitas peternakan rakyat.
- Mengurangi risiko kerugian ekonomi akibat wabah penyakit.

---

## 🌟 Business Understanding

### Problem Statement

Bagaimana cara membangun model machine learning yang mampu mengklasifikasikan jenis penyakit unggas berdasarkan gambar kotoran ayam?

### Goals

- Mengembangkan model klasifikasi berbasis citra.
- Meningkatkan akurasi dan efisiensi klasifikasi penyakit.
- Mengevaluasi kinerja beberapa algoritma dan memilih model terbaik.

### Solution Statement

Solusi yang dikembangkan berupa model supervised learning berbasis deep learning (CNN), menggunakan **arsitektur MobileNetV2** yang ringan namun akurat. Transfer learning dimanfaatkan dari pretrained ImageNet untuk mempercepat dan meningkatkan performa model.

---

## 📊 Data Understanding

### Sumber dan Karakteristik Dataset

- Dataset: [Chicken Disease Dataset](https://www.kaggle.com/api/v1/datasets/download/efoeetienneblavo/chicken-disease-dataset) (Kaggle)
- Jumlah gambar: 6.508 (train), 778 (val), 781 (test)
- Resolusi: 224x224 piksel
- Format: RGB (.jpg)
- Label:

  - Coccidiosis
  - Healthy
  - New Castle Disease
  - Salmonella

### Distribusi Kelas

- Coccidiosis: 1992
- Healthy: 1926
- New Castle Disease: 507
- Salmonella: 2111

Distribusi label tidak seimbang sehingga perlu penanganan khusus seperti augmentasi dan teknik evaluasi yang sesuai.

### Visualisasi Contoh Data

Visualisasi gambar dan distribusi label dilakukan untuk memastikan kualitas dan representasi data:

![Distribusi Kelas](https://github.com/user-attachments/assets/46eddaa8-c43d-42f8-ae48-daa205f3b9c4)

Label dikodekan sebagai berikut:

```python
{'Coccidiosis': 0, 'Healthy': 1, 'New Castle Disease': 2, 'Salmonella': 3}
```

---

## ⚙️ Data Preparation

### Teknik yang Digunakan

1. **Resize** gambar ke 224x224 piksel.
2. **Normalisasi** piksel ke skala \[0, 1].
3. **Augmentasi Data** untuk meningkatkan generalisasi:

   - Rotasi acak
   - Zoom acak
   - Flipping horizontal

4. **Stratified Split** untuk pembagian data train/val/test.
5. **Label Encoding** untuk label ke bentuk numerik.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)
```

---

## 🤖 Modeling

### Arsitektur Model: MobileNetV2

Model menggunakan arsitektur **MobileNetV2**, yang efisien dan cocok untuk perangkat mobile. Transfer learning digunakan dari bobot **ImageNet**. Layer klasifikasi ditambahkan di bagian atas dengan 4 output (softmax):

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True
```

Kemudian ditambahkan:

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')
])
```

Model dikompilasi dengan:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Proses Pelatihan

Model dilatih selama 10 epoch:

```bash
Epoch 1/10 - accuracy: 0.7475 - val_accuracy: 0.8368
...
Epoch 10/10 - accuracy: 0.9710 - val_accuracy: 0.9524
```

---

## 📈 Evaluation

Model menunjukkan performa sangat baik:

- **Akurasi validasi akhir**: 95.24%
- **Akurasi uji akhir**: 95.13%
- **Tidak overfitting** signifikan
- Inferensi cepat dan efisien untuk lingkungan produksi ringan

---

## 📅 Conclusion & Insights

- Arsitektur MobileNetV2 berhasil mengklasifikasikan gambar kotoran ayam dengan **akurasi tinggi**.
- Pendekatan transfer learning efektif untuk mengatasi keterbatasan data.
- Augmentasi data sangat berperan dalam meningkatkan generalisasi model.

### Insight Bisnis

- Solusi ini berpotensi diintegrasikan ke **aplikasi mobile** untuk diagnosis mandiri.
- Meningkatkan **akses kesehatan hewan** di daerah terpencil.
- Mencegah potensi **kerugian ekonomi** akibat wabah penyakit unggas.

### Rekomendasi Lanjutan

- 🔍 Integrasi **Grad-CAM** untuk interpretabilitas model.
- 🚀 **Deployment** ke aplikasi mobile (TF Lite) atau web (TF.js).
- 🌐 Penambahan kelas penyakit lainnya (misal: Avian Influenza).
- ⌛ Monitoring **model drift** untuk pembelajaran berkelanjutan.

✅ Dengan akurasi tinggi dan efisiensi tinggi, model ini sangat layak untuk dilanjutkan ke tahap implementasi lapangan.
