# Laporan Proyek Machine Learning - Moch. Yusuf Haidar Ali Ramdhani

## Domain Proyek

Dalam industri telekomunikasi dan layanan digital, menjaga loyalitas pelanggan merupakan tantangan utama. Salah satu indikator penting yang mencerminkan loyalitas tersebut adalah churn, yaitu kondisi ketika pelanggan berhenti menggunakan layanan suatu perusahaan. Tingkat churn yang tinggi berdampak langsung pada penurunan pendapatan, meningkatnya biaya akuisisi pelanggan baru, serta menurunnya citra perusahaan di mata investor dan publik.

Wardani et al. (2018) menyatakan bahwa churn pelanggan dapat menyebabkan kerugian besar bagi perusahaan retail, karena biaya promosi dan akuisisi pelanggan baru tidak selalu menghasilkan loyalitas jangka panjang. Hal ini menunjukkan betapa pentingnya mempertahankan pelanggan yang ada dibandingkan terus-menerus mencari pelanggan baru. Oleh karena itu, kemampuan untuk memprediksi pelanggan yang berpotensi churn menjadi sangat krusial agar perusahaan dapat mengambil tindakan preventif seperti memberikan penawaran khusus, memperbaiki layanan, atau melakukan pendekatan personal.

Seiring berkembangnya teknologi dan data analitik, machine learning kini menjadi salah satu solusi paling menjanjikan dalam menganalisis perilaku pelanggan. Dengan memanfaatkan data historis pelanggan—seperti lama berlangganan, jenis kontrak, metode pembayaran, status penggunaan layanan digital, dan total biaya bulanan—model prediktif dapat dibangun untuk mengidentifikasi pelanggan yang berisiko tinggi melakukan churn. Model ini membantu perusahaan untuk mengambil keputusan berbasis data dalam hal retensi dan strategi pemasaran.

Dalam proyek ini, digunakan beberapa algoritma supervised learning seperti Random Forest, Gradient Boosting, dan K-Nearest Neighbors (KNN) untuk membangun sistem prediksi churn berdasarkan dataset pelanggan dari sektor telekomunikasi. Dataset ini mencerminkan kondisi nyata pelanggan dan mencakup fitur-fitur penting seperti ‘Contract’, ‘PaymentMethod’, dan ‘MonthlyCharges’ yang terbukti relevan dalam analisis churn. Model yang dihasilkan tidak hanya dievaluasi berdasarkan akurasi, tetapi juga menggunakan metrik seperti precision, recall, dan F1-score untuk memastikan performa yang optimal, terutama terhadap kelas minoritas (pelanggan yang churn).

**Referensi**
Wardani, N. W., Dantes, G. R., & Indrawan, G. (2018). Prediksi Customer Churn dengan Algoritma Decision Tree C4.5 Berdasarkan Segmentasi Pelanggan untuk Mempertahankan Pelanggan pada Perusahaan Retail. Jurnal RESISTOR (Rekayasa Sistem Komputer), 1(1), 16–24. https://doi.org/10.31598/jurnalresistor.v1i1.219

## Business Understanding

Untuk mempertahankan keberlanjutan bisnis, perusahaan telekomunikasi harus mampu memahami dan merespons perilaku pelanggan secara proaktif. Salah satu tantangan utama adalah churn pelanggan, yaitu ketika pelanggan memilih berhenti menggunakan layanan yang ditawarkan. Masalah ini tidak hanya menyebabkan kehilangan pendapatan, tetapi juga menambah beban biaya untuk mengakuisisi pelanggan baru. Oleh karena itu, penting untuk memahami akar masalah churn serta mengembangkan solusi berbasis data.

### Problem Statements

- Bagaimana cara mengidentifikasi pelanggan yang berpotensi melakukan churn sebelum mereka benar-benar berhenti menggunakan layanan?
- Faktor-faktor apa saja yang paling berpengaruh terhadap keputusan pelanggan untuk melakukan churn?
- Seberapa akurat model machine learning dapat digunakan untuk memprediksi churn berdasarkan data historis pelanggan?

### Goals

- Membangun model prediktif yang mampu mengidentifikasi pelanggan berisiko churn secara akurat.
- Menentukan fitur-fitur pelanggan yang paling berkontribusi terhadap churn.
- Mengevaluasi kinerja berbagai algoritma machine learning dalam memprediksi churn dan memilih model terbaik.

### Solution statements
- Menerapkan dan membandingkan beberapa algoritma machine learning, seperti:

  - Random Forest: untuk menangani data kategorikal dan numerik secara seimbang serta menghasilkan interpretasi fitur yang jelas.

  - Gradient Boosting: untuk meningkatkan akurasi prediksi melalui pendekatan ensambel yang mengoptimalkan kesalahan model secara bertahap.

  - K-Nearest Neighbors (KNN): sebagai baseline model yang sederhana namun cukup efektif dalam mengenali pola tetangga terdekat.

- Melakukan hyperparameter tuning pada model terbaik guna meningkatkan performa dan menghindari overfitting, misalnya dengan GridSearchCV atau         RandomizedSearchCV.

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari Telco Customer Churn yang tersedia di Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data). Dataset ini berisi informasi pelanggan dari sebuah perusahaan telekomunikasi fiktif, termasuk detail layanan yang digunakan, data demografis, dan status churn.

Dataset terdiri dari 7043 baris dan 21 kolom, dengan setiap baris mewakili satu pelanggan.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- customerID : ID unik pelanggan.

- gender : Jenis kelamin pelanggan (Male atau Female).

- SeniorCitizen : Menunjukkan apakah pelanggan adalah warga senior (1) atau bukan (0).

- Partner : Apakah pelanggan memiliki pasangan (Yes/No).

- Dependents : Apakah pelanggan memiliki tanggungan (Yes/No).

- tenure : Lama berlangganan dalam bulan.

- PhoneService : Apakah pelanggan menggunakan layanan telepon (Yes/No).

- MultipleLines : Apakah pelanggan memiliki beberapa saluran telepon (No, Yes, No phone service).

- InternetService : Jenis layanan internet yang digunakan (DSL, Fiber optic, No).

- OnlineSecurity : Apakah pelanggan memiliki layanan keamanan online (Yes/No/No internet service).

- OnlineBackup : Apakah pelanggan memiliki layanan backup online (Yes/No/No internet service).

- DeviceProtection : Apakah pelanggan memiliki proteksi perangkat (Yes/No/No internet service).

- TechSupport : Apakah pelanggan memiliki dukungan teknis (Yes/No/No internet service).

- StreamingTV : Apakah pelanggan menggunakan layanan streaming TV (Yes/No/No internet service).

- StreamingMovies : Apakah pelanggan menggunakan layanan streaming film (Yes/No/No internet service).

- Contract : Jenis kontrak yang dipilih pelanggan (Month-to-month, One year, Two year).

- PaperlessBilling : Apakah pelanggan menggunakan tagihan tanpa kertas (Yes/No).

- PaymentMethod : Metode pembayaran yang digunakan pelanggan (Electronic check, Mailed check, Bank transfer, Credit card).

- MonthlyCharges : Jumlah biaya bulanan yang dibayar pelanggan.

- TotalCharges : Total biaya yang dibayarkan oleh pelanggan selama masa langganan.

- Churn : Target variabel, apakah pelanggan berhenti berlangganan (Yes) atau tidak (No).

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Sebelum membangun model prediksi churn, dilakukan beberapa tahapan persiapan data agar data siap digunakan oleh algoritma machine learning. Tahapan tersebut adalah sebagai berikut:

**Penanganan Data Kategori (Encoding)**

Dataset berisi fitur kategorikal yang perlu diubah menjadi representasi numerik agar dapat diproses oleh model. Tahapan encoding ini penting untuk memastikan model dapat memahami dan memanfaatkan informasi kategori secara efektif.

- Untuk fitur biner seperti Partner, Dependents, PhoneService, dan PaperlessBilling dilakukan Label Encoding, yang mengubah nilai "Yes"/"No" menjadi 1 dan 0.

- Untuk fitur kategorikal non-biner seperti gender, Contract, PaymentMethod, dan beberapa fitur layanan lainnya, digunakan One-Hot Encoding. Teknik ini mengubah setiap kategori menjadi kolom biner terpisah agar tidak memberikan bobot urutan pada model.

**Pemisahan Data (Train-Test Split)**

Dataset kemudian dibagi menjadi data pelatihan (train) dan data pengujian (test) dengan perbandingan 80:20. Pemisahan menggunakan stratifikasi berdasarkan label churn agar proporsi churn dan non-churn pada kedua subset tetap representatif dan seimbang. Tujuan split ini adalah untuk melatih model pada data train dan menguji performanya pada data test yang belum pernah dilihat model, sehingga evaluasi menjadi lebih valid.

**Standarisasi Fitur Numerik**

Fitur numerik seperti tenure, MonthlyCharges, dan TotalCharges memiliki skala nilai yang berbeda-beda dan rentang yang luas. Untuk menghindari fitur dengan skala besar mendominasi pembelajaran, dilakukan StandardScaler yang mentransformasikan fitur menjadi distribusi dengan mean 0 dan standar deviasi 1. Standarisasi dilakukan hanya pada data train (fit) dan hasil transformasi diaplikasikan juga ke data test untuk menghindari data leakage. Tahapan standarisasi ini membantu mempercepat konvergensi model dan meningkatkan akurasi, terutama pada algoritma yang sensitif terhadap skala fitur seperti KNN dan Gradient Boosting.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

