# MineToday 2026 — Rekomendasi Modul Pembelajaran

## Permasalahan

Kompetisi ini meminta sistem untuk mengurutkan **17 modul pembelajaran Intelligo** berdasarkan relevansinya bagi setiap pengguna. Profil asesmen dan riwayat percakapan pengguna perlu diterjemahkan menjadi skor untuk setiap modul, lalu digunakan untuk memilih lima modul paling relevan. Metrik evaluasinya adalah **NDCG@5** dengan gain relevansi linear: modul yang tepat perlu ditempatkan sedekat mungkin dengan urutan teratas.

## Dataset

Keenam berkas input resmi tersedia di [`data/raw/`](data/raw/):

| Berkas | Isi |
| --- | --- |
| `train_relevance.csv` | 4.000 pengguna latih; `user_id` dan skor relevansi `M_001`–`M_017`. |
| `test.csv` | 1.000 `user_id` yang perlu diprediksi. |
| `chat_history.csv` | 14.573 catatan percakapan pengguna. |
| `user_assessments.csv` | Hasil asesmen 5.000 pengguna, termasuk jawaban yang perlu dibersihkan. |
| `modules_catalog.csv` | Nama, deskripsi, silabus, dan tingkat prasyarat 17 modul. |
| `sample_submission.csv` | Struktur dan urutan pengguna yang wajib diikuti saat menyerahkan prediksi. |

## Pendekatan yang diimplementasikan

[`notebooks/psi-1_Notebook.ipynb`](notebooks/psi-1_Notebook.ipynb) adalah pipeline solusi utama. Notebook membersihkan asesmen, membentuk fitur tabular serta afinitas teks antara pengguna dan modul, lalu membagi data ke dalam lima *fold* yang terstratifikasi menurut modul paling relevan. Enam keluarga model dilatih dengan pembagian yang sama: **LightGBM ListMLE**, **XGBoost pairwise**, **CatBoost**, **17 regresor LightGBM per modul**, **LightGBM dua tahap**, dan **Ridge**.

Prediksi *out-of-fold* keenam komponen menjadi masukan meta-model LightGBM ListMLE. Setelah penggabungan, notebook menerapkan penyesuaian skor pengguna yang deterministik dan memeriksa kolom, urutan `user_id`, serta nilai kosong sebelum menghasilkan [`submissions/submission_det6.csv`](submissions/submission_det6.csv). Keterangan ketentuan final ada di [`docs/GUDEBOOK_FINAL.md`](docs/GUDEBOOK_FINAL.md).

**Catatan:** notebook membutuhkan pustaka pembelajaran mesin terkait. Menjalankan ulang seluruh sel dapat menulis ulang berkas submisi; berkas `dataset/` dari pipeline lama sengaja tidak disertakan karena katalog modulnya berbeda dari `data/raw/`.
