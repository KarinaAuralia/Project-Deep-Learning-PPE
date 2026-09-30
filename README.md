# PPE Object Detection — Kasus 31

## Monitoring Kepatuhan Penggunaan Personal Protective Equipment (PPE) Kru Lapangan Berbasis YOLOv8n dengan Optimasi Kuantisasi dan Pruning untuk Inferensi Edge AI di Browser

Proyek tugas besar mata kuliah **Deep Learning C-083** untuk Kasus 31 pada domain **HSSE (Health, Safety, Security, and Environment)** industri migas.

Sistem ini menggunakan **YOLOv8n** untuk mendeteksi objek Personal Protective Equipment (PPE/APD) pada citra. Model mendeteksi lima kelas objek:

- Helmet
- Mask
- Safety Vest
- Boots
- Glove

> **Catatan ruang lingkup:** dataset yang digunakan tidak memiliki kelas `Person`. Karena itu, sistem melakukan deteksi keberadaan objek PPE pada citra dan belum dapat memverifikasi kepatuhan PPE setiap individu.

## Pilar Industri yang Disasar

Proyek ini menyasar pilar **HSSE — Keselamatan, Kesehatan Kerja, Keamanan Fasilitas, dan Lingkungan**, khususnya aspek **keselamatan tenaga kerja lapangan**. Sistem berfungsi sebagai prototipe pendukung monitoring keberadaan objek PPE pada citra area kerja seperti rig floor, dermaga, dan kilang.

Kontribusi yang ditargetkan adalah membantu safety officer menyaring citra, meninjau objek PPE yang terdeteksi, dan mendokumentasikan hasil inspeksi. Sistem tidak menggantikan keputusan safety officer dan belum menyatakan kepatuhan personal karena dataset tidak memiliki kelas `Person`.

## Live Demo

**Vercel:** https://ppe-kelompok4.vercel.app/

## Tujuan Proyek

Proyek ini bertujuan untuk:

1. Membangun model deteksi objek PPE menggunakan YOLOv8n.
2. Mengevaluasi model menggunakan precision, recall, F1-score, mAP@0.5, dan mAP@0.5:0.95.
3. Membandingkan model baseline dengan model ONNX INT8 dan model hasil pruning.
4. Mengukur ukuran model dan latency inferensi.
5. Mengintegrasikan model ONNX ke aplikasi web dengan fitur upload citra dan Try Sample Data.
6. Mendukung proses monitoring keselamatan kerja pada pilar HSSE.

## Fitur Aplikasi Web

- Landing page profil kelompok.
- Profil lima anggota tim, NPM, peran, dan tautan LinkedIn.
- Upload citra JPG/PNG.
- Tiga sample image bawaan:
  - Sample 01 — Rig Floor
  - Sample 02 — Dermaga
  - Sample 03 — Kilang
- Pengaturan confidence threshold.
- Bounding box hasil deteksi.
- Nama kelas objek dan confidence score.
- Tabel perbandingan performa model.
- Evaluasi performa per kelas.

## Alur Penelitian

Penelitian terdiri atas delapan tahap utama:

1. **Dataset PPE:** dataset Kaggle melalui `kagglehub`, berisi 11.777 citra dan lima kelas.
2. **Audit data:** pemeriksaan `data.yaml`, jumlah citra, anotasi, dan distribusi instance.
3. **Split bawaan:** train 8.243, validation 2.357, dan test 1.177 citra.
4. **Preprocessing dan augmentasi:** input 640 × 640 piksel dengan mosaic, horizontal flip, HSV jitter, scale, translasi, random erasing, dan RandAugment pada data training.
5. **Training YOLOv8n:** transfer learning dari COCO selama 50 epoch menggunakan GPU Tesla T4.
6. **Evaluasi:** precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95, dan confusion matrix pada test set.
7. **Optimasi Edge AI:** ekspor ONNX FP32/FP16, post-training quantization INT8, pruning unstructured 30%, dan benchmark.
8. **Web app:** integrasi aplikasi Vercel dengan tombol Try Sample Data dan inferensi ONNX.

> Seluruh eksperimen dijalankan di Google Colab menggunakan Ultralytics 8.4.166 dan PyTorch 2.11.

## Arsitektur Sistem

```text
Input image / sample image
          |
          v
Preprocessing dan letterbox ke 640 x 640 RGB
          |
          v
ONNX Runtime — best_model.onnx (FP32)
          |
          v
Post-processing, confidence filtering, dan NMS
          |
          v
Bounding box + class + confidence score
          |
          v
Visualisasi hasil pada Flask web application
```

### Catatan implementasi Edge AI

Model yang digunakan aplikasi adalah **ONNX FP32 berukuran 11,73 MB**. Aplikasi web menjalankan inferensi melalui Flask dan ONNX Runtime pada deployment Vercel. Format ONNX dipilih sebagai model utama karena kompatibilitasnya paling luas dan bobotnya sama dengan model baseline yang telah dievaluasi.

Latency khusus pada browser belum diukur. Oleh karena itu, hasil benchmark notebook tidak boleh dianggap sebagai latency browser langsung.

## Dataset

Dataset yang digunakan:

- **Nama:** PPE Dataset — Filtered PPE Detection Dataset (5 Classes)
- **Sumber:** Kaggle
- **URL:** https://www.kaggle.com/datasets/waquarahmed1/ppe-dataset
- **Format:** YOLO Object Detection
- **Lisensi:** CC BY-SA 4.0
- **Split:** train, validation, dan test bawaan dataset

### Distribusi dataset

| Split | Jumlah citra | Persentase |
|---|---:|---:|
| Train | 8.243 | 69,99% |
| Validation | 2.357 | 20,01% |
| Test | 1.177 | 9,99% |
| **Total** | **11.777** | **100%** |

### Kelas objek

| Kelas | Keterangan |
|---|---|
| Helmet | Helm keselamatan |
| Mask | Masker/pelindung wajah |
| Safety Vest | Rompi keselamatan |
| Boots | Sepatu keselamatan |
| Glove | Sarung tangan |

Dataset Kaggle merupakan adaptasi dari dataset Roboflow. Kelas `Person` dan `glasses` telah dihapus dari dataset adaptasi tersebut.

Sumber Roboflow asal: https://universe.roboflow.com/glovesdetection/my-first-project-przjz-dj1mn/dataset/1

## Hasil Evaluasi

Evaluasi dilakukan pada 1.177 citra test dengan 3.751 instance bounding box.

### Model baseline

| Metrik | Nilai |
|---|---:|
| Precision | 0,894 |
| Recall | 0,797 |
| F1-score | 0,843 |
| mAP@0.5 | 0,860 |
| mAP@0.5:0.95 | 0,576 |
| Ukuran model PyTorch | 5,96 MB |

### Perbandingan optimasi

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Ukuran |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n FP32 baseline | 0,894 | 0,797 | 0,843 | 0,860 | 0,576 | 5,96 MB |
| YOLOv8n INT8 ONNX | 0,895 | 0,775 | 0,831 | 0,792 | 0,531 | 3,27 MB |
| YOLOv8n pruned 30% tanpa fine-tuning | 0,885 | 0,775 | 0,826 | 0,847 | 0,561 | 5,99 MB |

### Latency benchmark

Benchmark dilakukan dengan lima warm-up dan 20 pengulangan pada satu citra uji.

| Model | Perangkat/runtime | Median latency | P95 latency |
|---|---|---:|---:|
| FP32 PyTorch | GPU Tesla T4 | 11,82 ms | 13,38 ms |
| Pruned PyTorch* | GPU Tesla T4 | 11,11 ms | 12,18 ms |
| FP32 ONNX | CPU Intel Xeon 2,00 GHz | 101,01 ms | 114,43 ms |
| FP16 ONNX | CPU Intel Xeon 2,00 GHz | 96,86 ms | 135,23 ms |
| INT8 ONNX | CPU Intel Xeon 2,00 GHz | 171,00 ms | 257,68 ms |

Latency PyTorch dan ONNX tidak dibandingkan secara langsung karena menggunakan perangkat/runtime yang berbeda. Model INT8 merupakan model terkecil, tetapi paling lambat pada CPU benchmark. Model utama aplikasi adalah ONNX FP32; INT8 disediakan sebagai opsi ringan dengan konsekuensi penurunan mAP dan recall.

`*` Benchmark pruned memakai model hasil fine-tuning; arsitekturnya sama sehingga ukuran dan latency setara dengan baseline.

## Dampak Operasional dan Risiko

Solusi ini menyasar **Pilar 1 HSSE** dengan target dukungan terhadap zero accident dan berkontribusi pada efisiensi biaya komputasi melalui Edge AI. Skenario penggunaan adalah penapisan otomatis cuplikan CCTV di rig floor, dermaga, dan kilang. Model membantu menandai frame yang perlu ditinjau, sedangkan petugas HSE tetap melakukan verifikasi.

Interpretasi metrik pada ambang confidence 0,25:

- Precision 0,894 berarti sebagian deteksi dapat berupa false positive.
- Recall 0,797 berarti sebagian objek PPE yang benar-benar ada masih terlewat.
- Karena dataset tidak memiliki kelas `Person`, sistem belum dapat menyimpulkan kepatuhan per pekerja.
- Risiko alarm palsu dan pelanggaran yang terlewat harus dikendalikan dengan verifikasi petugas HSE.

Estimasi penghematan rupiah belum dihitung karena memerlukan data jumlah kamera, biaya server/cloud GPU, dan jam kerja pemantau dari perusahaan.

## Struktur Repository

```text
.
├── README.md
└── Website/
    ├── app.py
    ├── requirements.txt
    ├── best_model.onnx
    ├── best.onnx
    ├── best_ppe.onnx
    ├── best.pt
    ├── best_ppe.pt
    ├── convert.ipynb
    ├── debug_output.py
    ├── ppe.db
    ├── templates/
    │   └── index.html
    └── static/
        ├── logo.svg
        ├── samples/
        ├── team/
        ├── uploads/
        └── results/
```

Model yang digunakan oleh aplikasi berdasarkan kode `Website/app.py` adalah:

```python
MODEL_PATH = os.path.join(BASE_DIR, "best_model.onnx")
```

## Cara Menjalankan Secara Lokal

### 1. Clone repository

```bash
git clone https://github.com/KarinaAuralia/Project-Deep-Learning-PPE.git
cd Project-Deep-Learning-PPE/Website
```

### 2. Buat virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies utama:

- Flask
- Pillow
- NumPy
- ONNX Runtime

### 4. Jalankan aplikasi

```bash
python app.py
```

Buka alamat berikut pada browser:

```text
http://localhost:5000
```

## Deployment Vercel

Aplikasi dapat dideploy sebagai aplikasi Flask Python pada Vercel dengan memastikan:

1. File entry point `app.py` tersedia.
2. `requirements.txt` berada di direktori deployment.
3. Model `best_model.onnx` tersedia dan berada di bawah batas ukuran file repository/deployment.
4. Folder `templates/` dan `static/` ikut ter-deploy.
5. Folder sementara pada environment server digunakan untuk file upload dan hasil inference.
6. Variabel environment `VERCEL=1` digunakan oleh aplikasi untuk mengarahkan file sementara ke direktori temporary.

File upload dan hasil analisis pada environment serverless bersifat sementara. Database lokal tidak sebaiknya digunakan sebagai penyimpanan permanen untuk production.

## Anggota Kelompok dan Contribution Statement

| Nama (NPM) | Peran | Tanggung jawab utama | Kontribusi | Profil |
|---|---|---|---:|---|
| Difta Alzena Sakhi (23083010061) | Ketua; Model Architect & Quantization | Koordinasi tim, arsitektur YOLOv8n, konfigurasi training, quantization INT8, ekspor ONNX, pruning, dan analisis model. | 25% | [LinkedIn](https://www.linkedin.com/in/difta-alzena-sakhi-09a2b1315/) |
| Azizah Zalfa Assyadida (23083010064) | Data Pipeline & Augmentation | Dataset, audit `data.yaml`, anotasi, distribusi kelas, split, preprocessing, dan augmentasi. | 20% | [LinkedIn](https://www.linkedin.com/in/azalfassyadida/) |
| Ni Luh Ayu Nariswari Dewi (23083010068) | Edge Runtime & Inference Optimization | Ekspor ONNX, INT8, pruning 30%, pengujian ONNX Runtime, ukuran model, dan latency. | 20% | [LinkedIn](https://www.linkedin.com/in/ayu-nariswari/) |
| Karina Auralia (23083010072) | Frontend & Integrasi Web | Antarmuka, upload, Try Sample Data, Flask, ONNX Runtime, visualisasi, deployment, dan pengujian Vercel. | 20% | [LinkedIn](https://www.linkedin.com/in/karinaauralia/) |
| Tiara Audrey Anugerah Hadin (23083010079) | Technical Writer & Validasi HSE | Studi literatur, laporan ilmiah, analisis dampak HSSE, daftar pustaka, validasi keselamatan, dan proofreading. | 15% | [LinkedIn](https://www.linkedin.com/in/tiara-audrey/) |

**Total kontribusi: 100%.**

## Artefak Proyek

- **Repositori GitHub:** https://github.com/KarinaAuralia/Project-Deep-Learning-PPE/tree/main
- **Aplikasi web:** https://ppe-kelompok4.vercel.app/

## Lisensi dan Atribusi

Kode proyek dapat digunakan untuk keperluan akademik dengan tetap mencantumkan atribusi kelompok. Dataset PPE mengikuti lisensi **CC BY-SA 4.0** dari halaman Kaggle dan sumber adaptasinya.

Sebelum mendistribusikan ulang dataset atau model, periksa kembali ketentuan lisensi dataset asal Roboflow dan dataset adaptasi Kaggle.

## Citation

Jika proyek ini digunakan sebagai referensi akademik, cantumkan informasi berikut:

```text
Kelompok 4. Monitoring Objek Personal Protective Equipment Kru Lapangan
Berbasis YOLOv8n dengan Optimasi Kuantisasi dan Pruning untuk Inferensi Edge AI.
Tugas Besar Deep Learning C-083, Program Studi Sains Data,
Universitas Pembangunan Nasional “Veteran” Jawa Timur, 2026.
```
