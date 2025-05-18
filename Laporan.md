# Laporan Proyek Machine Learning - Moch. Yusuf Haidar Ali Ramdhani
# Klasifikasi Customer Churn

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

- Melakukan hyperparameter tuning pada model terbaik guna meningkatkan performa dan menghindari overfitting, misalnya dengan GridSearchCV.

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari Telco Customer Churn yang tersedia di Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data). Dataset ini berisi informasi pelanggan dari sebuah perusahaan telekomunikasi fiktif, termasuk detail layanan yang digunakan, data demografis, dan status churn.

Dataset terdiri dari 7043 baris dan 21 kolom, dengan setiap baris mewakili satu pelanggan.

### Variabel-variabel pada Telco Customer Churn dataset adalah sebagai berikut:
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

### Exploratory Data Analysis

**Informasi Dataset**

![info dataset](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/info.png)

**Kolom Null**

![data null](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/null.png)

Nilai null hanya terdapat pada kolom 'TotalCharges' sebanyak 11 baris. Dikarenakan konversi dari tipe object ke tipe float.

**Informasi Kolom Numerikal**

![describe numerical](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/numerik.png)

1. tenure (lama pelanggan berlangganan, dalam bulan):

Mean: 32.42 bulan → rata-rata pelanggan bertahan sekitar 2,7 tahun.

Min–Max: dari 1 hingga 72 bulan → ada pelanggan yang baru 1 bulan, ada juga yang sudah 6 tahun (maksimal).

Distribusi:

- 25% pelanggan hanya bertahan ≤ 9 bulan → indikasi churn dini.

- 50% (median) bertahan sampai 29 bulan.

- 25% pelanggan tertinggi bertahan lebih dari 55 bulan → pelanggan loyal.

2. MonthlyCharges (tagihan bulanan):

Mean: $64.80, dengan rentang $18.25 – $118.75.

25% pelanggan membayar ≤ $35.59 → mungkin hanya ambil paket dasar.

75% pelanggan membayar hingga hampir $90 → berarti banyak juga yang ambil layanan tambahan.

3. TotalCharges (total biaya selama berlangganan):

Mean: $2,283.30 tapi sangat spread out (std = $2,266.77).

Ada pelanggan yang hanya bayar total $18.8 (mungkin baru bergabung), dan yang lain lebih dari $8,684 (pelanggan lama & mahal).

**Visualisasi Data Kategorikal**

![kategorikal](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/kategorikal1.png)

- Gender: Terdapat 3.549 data berjenis kelamin laki-laki dan 3.483 data berjenis kelamin perempuan.

- Partner: Jumlah data dengan status tidak memiliki pasangan adalah 3.639, sedangkan yang memiliki pasangan sebanyak 3.393.

- Dependents: Data pelanggan tanpa tanggungan berjumlah 4.933, sementara yang memiliki tanggungan berjumlah 2.099.

- PhoneService: Sebanyak 6.352 pelanggan menggunakan layanan telepon, dan 680 pelanggan tidak menggunakan layanan tersebut.

![kategorikal](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/kategorikal2.png)

- MultipleLines: Data terbagi menjadi tiga kategori, yaitu 3.385 pelanggan tanpa layanan multiple lines, 2.967 pelanggan yang menggunakan layanan multiple lines, serta 680 pelanggan yang tidak memiliki layanan telepon.

- InternetService: Pelanggan terbagi menjadi tiga kelompok berdasarkan jenis layanan internet yang digunakan, yaitu 3.096 pelanggan menggunakan Fiber optic, 2.416 pelanggan menggunakan DSL, dan 1.520 pelanggan tidak menggunakan layanan internet.

- OnlineSecurity: Sebanyak 3.497 pelanggan tidak menggunakan layanan keamanan online, 2.015 pelanggan menggunakan layanan tersebut, dan 1.520 pelanggan tidak menggunakan layanan internet sehingga tidak memakai layanan keamanan online.

- OnlineBackup: Jumlah pelanggan tanpa layanan backup online adalah 3.087, dengan 2.425 pelanggan menggunakan layanan tersebut, dan 1.520 pelanggan tidak memiliki layanan internet.

![kategorikal](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/kategorikal3.png)

- DeviceProtection: Terdapat 3.094 pelanggan tanpa layanan proteksi perangkat, 2.418 pelanggan yang menggunakan layanan tersebut, dan 1.520 pelanggan tanpa layanan internet.

- TechSupport: Data menunjukkan 3.472 pelanggan tanpa layanan dukungan teknis, 2.040 pelanggan yang menggunakan layanan, dan 1.520 pelanggan tanpa layanan internet.

- StreamingTV: Sebanyak 2.809 pelanggan tidak menggunakan layanan streaming TV, 2.703 pelanggan menggunakan layanan tersebut, dan 1.520 pelanggan tanpa layanan internet.

- StreamingMovies: Terdapat 2.781 pelanggan tanpa layanan streaming film, 2.731 pelanggan yang menggunakan layanan tersebut, serta 1.520 pelanggan tanpa layanan internet.

![kategorikal](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/kategorikal4.png)

- Contract: Pelanggan terbagi dalam tiga jenis kontrak, yaitu 3.875 pelanggan dengan kontrak bulanan, 1.685 pelanggan dengan kontrak dua tahun, dan 1.472 pelanggan dengan kontrak satu tahun.

- PaperlessBilling: Data menunjukkan 4.168 pelanggan menggunakan layanan tagihan tanpa kertas, dan 2.864 pelanggan menggunakan tagihan konvensional.

- PaymentMethod: Terdapat empat metode pembayaran yang digunakan pelanggan, yaitu 2.365 dengan cek elektronik, 1.604 dengan cek pos, 1.542 dengan transfer bank otomatis, dan 1.521 dengan kartu kredit otomatis.

- Churn: Jumlah pelanggan yang tidak berhenti berlangganan sebanyak 5.163, sedangkan yang berhenti berlangganan sejumlah 1.869.

**Visualisasi Data Numerikal**

![numerikal](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/numerikal.png)

- TotalCharges dan Tenure menunjukkan hubungan yang kuat dengan nilai korelasi sebesar 0.83. Hal ini mengindikasikan bahwa total tagihan pelanggan cenderung meningkat seiring dengan lamanya pelanggan menggunakan layanan (tenure).

- MonthlyCharges dan TotalCharges memiliki korelasi sedang sebesar 0.65, yang menunjukkan bahwa tagihan bulanan juga berkontribusi terhadap total tagihan, namun pengaruhnya tidak sebesar hubungan antara tenure dan total charges.

- Tenure dan MonthlyCharges menunjukkan korelasi yang paling lemah dengan nilai 0.25, menandakan bahwa lama berlangganan pelanggan tidak terlalu berpengaruh terhadap besarnya tagihan bulanan.

**Matriks Korelasi**

![numerikal](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/matriks_korelasi.png)

Kolom TotalCharges dengan kolom tenure memiliki hubungan yang kuat yang bernilai 0.83

Kolom MonthlyCharges dengan TotalCharges memiliki sedikit hubungan yang bernilai 0.65

Sedangkan kolom tenure dan MonthlyCharges menunjukkan hubungan paling lemah yang brenilai 0.25

## Data Preparation
Sebelum membangun model prediksi churn, dilakukan beberapa tahapan persiapan data agar data siap digunakan oleh algoritma machine learning. Tahapan tersebut adalah sebagai berikut:

**Menangani Missing Value**

Ditemukan nilai kosong sebanyak 11 entri pada kolom TotalCharges. Nilai-nilai ini kemungkinan besar muncul dari pelanggan yang baru bergabung dan belum memiliki tagihan total. Untuk menjaga kualitas data, baris dengan nilai kosong tersebut dihapus dari dataset karena jumlahnya sangat kecil dan tidak akan berdampak signifikan terhadap distribusi data.

**Menangani Data Duplikat**

Pemeriksaan terhadap seluruh dataset menunjukkan bahwa tidak terdapat duplikasi baris, sehingga tidak diperlukan penghapusan data duplikat.

**Drop Kolom customerID**

Kolom customerID dihapus karena bersifat unik untuk setiap pelanggan dan tidak memiliki kontribusi informasi terhadap proses prediksi. Kolom ini hanya berfungsi sebagai identifier dan tidak relevan untuk model pembelajaran mesin.

**Menangani Outlier**

Analisis terhadap kolom numerik seperti MonthlyCharges, TotalCharges, dan tenure menunjukkan bahwa tidak terdapat outlier yang mencolok. Penyebaran data berada dalam rentang yang wajar, sehingga tidak diperlukan penyesuaian atau transformasi lebih lanjut.

**Penanganan Data Kategori (Encoding)**

Dataset berisi fitur kategorikal yang perlu diubah menjadi representasi numerik agar dapat diproses oleh model. Tahapan encoding ini penting untuk memastikan model dapat memahami dan memanfaatkan informasi kategori secara efektif.

- Untuk fitur biner seperti Partner, Dependents, PhoneService, dan PaperlessBilling dilakukan Label Encoding, yang mengubah nilai "Yes"/"No" menjadi 1 dan 0.

- Untuk fitur kategorikal non-biner seperti gender, Contract, PaymentMethod, dan beberapa fitur layanan lainnya, digunakan One-Hot Encoding. Teknik ini mengubah setiap kategori menjadi kolom biner terpisah agar tidak memberikan bobot urutan pada model.

**Pemisahan Data (Train-Test Split)**

Dataset kemudian dibagi menjadi data pelatihan (train) dan data pengujian (test) dengan perbandingan 80:20. Pemisahan menggunakan stratifikasi berdasarkan label churn agar proporsi churn dan non-churn pada kedua subset tetap representatif dan seimbang. Tujuan split ini adalah untuk melatih model pada data train dan menguji performanya pada data test yang belum pernah dilihat model, sehingga evaluasi menjadi lebih valid.

**Standarisasi Fitur Numerik**

Fitur numerik seperti tenure, MonthlyCharges, dan TotalCharges memiliki skala nilai yang berbeda-beda dan rentang yang luas. Untuk menghindari fitur dengan skala besar mendominasi pembelajaran, dilakukan StandardScaler yang mentransformasikan fitur menjadi distribusi dengan mean 0 dan standar deviasi 1. Standarisasi dilakukan hanya pada data train (fit) dan hasil transformasi diaplikasikan juga ke data test untuk menghindari data leakage. Tahapan standarisasi ini membantu mempercepat konvergensi model dan meningkatkan akurasi, terutama pada algoritma yang sensitif terhadap skala fitur seperti KNN dan Gradient Boosting.

## Modeling
Dalam proyek ini, digunakan tiga algoritma machine learning untuk memprediksi churn pelanggan, yaitu K-Nearest Neighbors (KNN), Random Forest, dan Gradient Boosting. Ketiga model ini dipilih karena memiliki karakteristik dan kelebihan masing-masing dalam menangani data klasifikasi seperti churn prediction.

**1. K-Nearest Neighbors (KNN)**

- Deskripsi: KNN merupakan algoritma berbasis instance-based learning yang melakukan klasifikasi berdasarkan mayoritas label dari k tetangga terdekat dalam ruang fitur.

- Kelebihan: Mudah dipahami dan diimplementasikan, efektif untuk dataset dengan distribusi yang jelas.

- Kekurangan: Sensitif terhadap fitur yang tidak diskalakan dan data outlier, serta cenderung lambat pada dataset besar karena harus menghitung jarak ke semua titik data.

- Parameter utama: Jumlah tetangga (n_neighbors=5 secara default).

**2. Random Forest**

- Deskripsi: Random Forest adalah algoritma ensemble berbasis decision tree. Model ini membentuk banyak pohon keputusan secara acak dan mengambil voting mayoritas untuk menentukan prediksi akhir.

- Kelebihan: Tahan terhadap overfitting, dapat menangani fitur numerik dan kategorikal, serta memberikan fitur penting (feature importance).

- Kekurangan: Model cenderung lebih kompleks dan membutuhkan waktu komputasi lebih lama dibandingkan KNN.

- Parameter utama: n_estimators=100, random_state=42.

**3. Gradient Boosting**

- Deskripsi: Gradient Boosting membangun model prediktif secara bertahap. Setiap model baru fokus pada memperbaiki kesalahan dari model sebelumnya.
  
- Kelebihan: Memiliki performa yang tinggi pada banyak jenis data, mampu menangani interaksi fitur yang kompleks.

- Kekurangan: Rentan terhadap overfitting jika tidak dilakukan tuning parameter dengan tepat, membutuhkan waktu pelatihan yang lebih lama.

- Parameter utama: n_estimators=100, learning_rate=0.1, random_state=42.

**Proses Pelatihan dan Evaluasi**

- Semua model dilatih menggunakan data training yang sudah dipersiapkan (encoded dan distandarisasi).

- Evaluasi dilakukan dengan mengukur akurasi, precision, recall, dan F1-score pada data test.

- Cross-validation dengan Stratified K-Fold (5 fold) juga dilakukan untuk mendapatkan estimasi performa yang lebih stabil dan menghindari bias.

**Pemilihan Model Terbaik**

Berdasarkan evaluasi cross-validation 5-fold, model Gradient Boosting menunjukkan performa tertinggi dibanding model lainnya, dengan rata-rata akurasi 80.07% dan F1-score 57.68%. Khususnya pada recall dan F1-score yang penting untuk kasus churn karena ingin meminimalkan false negative (pelanggan yang berisiko churn tidak terdeteksi) Oleh karena itu, model ini dipilih sebagai kandidat final.

**Improvement Model**

Untuk model Gradient Boosting, telah dilakukan hyperparameter tuning menggunakan teknik Grid Search untuk mencari kombinasi terbaik dari parameter seperti n_estimators, learning_rate, dan max_depth. Proses ini menggunakan 5-fold cross-validation dengan metrik evaluasi F1-score sebagai dasar pemilihan model terbaik. Meskipun tuning ini membantu mengeksplorasi performa model secara lebih luas, hasil akhir pada data uji sedikit lebih rendah dibandingkan model Gradient Boosting dengan parameter default. Hal ini dapat disebabkan oleh:

- Model default secara kebetulan sudah cukup optimal untuk distribusi data yang ada.

- Kombinasi parameter hasil tuning lebih cocok untuk distribusi pada training set, namun sedikit kurang generalisasi pada test set.

Dengan demikian, meskipun proses tuning tetap bernilai dalam konteks eksplorasi dan validasi, model Gradient Boosting default tetap dipilih sebagai model final karena memberikan hasil yang sedikit lebih unggul secara keseluruhan.

## Evaluation
### Metrik Evaluasi yang Digunakan
Dalam proyek prediksi churn pelanggan ini, beberapa metrik evaluasi utama yang digunakan adalah:

**Accuracy**

![Accuracy](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Accuracy.png)

Mengukur proporsi prediksi yang benar dari keseluruhan data. Namun, akurasi dapat menyesatkan jika data tidak seimbang karena model dapat memberikan prediksi mayoritas yang tinggi.

**Precision**

![Precision](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Precision.png)

Mengukur seberapa tepat prediksi positif model, yaitu dari semua yang diprediksi churn, berapa yang benar-benar churn. Precision penting untuk menghindari terlalu banyak false alarm, yaitu pelanggan yang diprediksi churn padahal tidak.

**Recall (Sensitivity)**

![Recall](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Recall.jpg)

Mengukur kemampuan model untuk menangkap semua kasus positif sebenarnya, yaitu dari semua pelanggan yang benar-benar churn, berapa yang terdeteksi oleh model. Recall sangat penting dalam konteks churn prediction karena perusahaan ingin meminimalkan pelanggan yang hilang tanpa terdeteksi.

**F1-Score**

![F1-Score](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/F1-Score.png)

Harmonik rata-rata dari precision dan recall, memberikan keseimbangan antara keduanya. F1 Score berguna sebagai metrik tunggal untuk mengukur performa pada dataset yang tidak seimbang.

**Cross-Validation: Stratified K-Fold**

Selain evaluasi pada data uji, kami juga menggunakan teknik Stratified K-Fold Cross Validation (CV) dengan 5 lipatan untuk mengevaluasi performa model secara lebih stabil dan menghindari overfitting. Stratified berarti distribusi kelas dijaga tetap seimbang di setiap lipatan (fold), penting karena data churn cenderung tidak seimbang. Cross-validation membagi data latih menjadi 5 bagian: 4 untuk training, 1 untuk validasi, dan dilakukan secara bergantian sebanyak 5 kali. Hasil akhir adalah rata-rata dari setiap metrik di seluruh fold. Metode ini memberikan estimasi performa model yang lebih andal dan mendekati performa sebenarnya di data dunia nyata.

### Penjelasan Hasil Evaluasi

![Evaluasi](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Evaluasi.png)

Berdasarkan hasil evaluasi pada data uji, performa tiga algoritma pembelajaran mesin—K-Nearest Neighbors (KNN), Random Forest, dan Gradient Boosting—dibandingkan menggunakan metrik accuracy, precision, recall, dan F1 score.

Model KNN mencapai akurasi 0.7669 dengan precision 0.5650, recall 0.5338, dan F1 score 0.5489. Meskipun sederhana, KNN memiliki keterbatasan dalam menangkap pola yang kompleks serta sensitif terhadap skala data dan distribusi kelas.

Model Random Forest menunjukkan peningkatan performa dengan akurasi 0.7927, precision 0.6438, recall 0.4916, dan F1 score 0.5569. Model ini lebih baik dalam membedakan pelanggan churn dibanding KNN, meskipun recall-nya masih rendah, sehingga masih banyak churner yang tidak terdeteksi.

Model terbaik adalah Gradient Boosting dengan akurasi tertinggi 0.8007, precision 0.6621, recall 0.5117, dan F1 score 0.5768. F1 score yang lebih tinggi menandakan keseimbangan terbaik antara kemampuan model mengidentifikasi churner (recall) tanpa terlalu banyak prediksi positif yang salah (precision). Oleh karena itu, Gradient Boosting dipilih sebagai model akhir karena performa paling optimal untuk kasus churn yang membutuhkan keseimbangan antara deteksi dan ketepatan.

![Improved](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Improved.png)

Selain model utama tersebut, upaya improvement melalui hyperparameter tuning dan fitur tambahan juga dilakukan. Namun, hasil evaluasi menunjukkan penurunan akurasi menjadi 0.7932 pada model improved, dengan precision turun menjadi 0.6317, recall naik ke 0.5321, dan F1 score sedikit meningkat menjadi 0.5776. Cross-validated accuracy model improved adalah 0.7995, hampir setara dengan akurasi model default.

Perbandingan ini menunjukkan adanya trade-off: precision menurun sementara recall meningkat, sehingga F1 score relatif stabil. Penurunan akurasi kemungkinan disebabkan oleh parameter yang dioptimalkan untuk meningkatkan deteksi churner (kelas minoritas), yang berdampak pada kemampuan generalisasi. Cross-validation mengindikasikan model improved masih konsisten dan tidak mengalami overfitting.

Secara keseluruhan, meskipun akurasi sedikit menurun, peningkatan recall dan F1 score adalah hal positif dalam konteks churn, di mana kemampuan mendeteksi pelanggan yang akan churn sangat penting. Pemilihan antara model default atau improved dapat disesuaikan dengan prioritas bisnis, apakah mengutamakan ketepatan prediksi positif (precision) atau cakupan deteksi churn (recall).

### Keterkaitan dengan Business Understanding

Evaluasi performa model dalam proyek ini memberikan kontribusi signifikan terhadap pemahaman bisnis, khususnya dalam menjawab kebutuhan utama perusahaan: mengurangi kehilangan pelanggan secara proaktif. Model Gradient Boosting yang dipilih terbukti mampu mendeteksi pelanggan yang berisiko churn dengan akurasi dan keseimbangan metrik yang baik (precision dan recall), yang berarti perusahaan dapat mengidentifikasi dan menargetkan pelanggan yang berpotensi churn dengan intervensi yang tepat waktu dan efisien.

### Keterkaitan dengan Problem Statements

**Identifikasi pelanggan sebelum churn:**

Model Gradient Boosting dengan recall 0.5117 dan versi improved dengan 0.5321 menunjukkan bahwa lebih dari 50% pelanggan churn berhasil terdeteksi sebelum mereka pergi. Meskipun ini menunjukkan kemajuan, angka tersebut belum sepenuhnya ideal untuk kebutuhan bisnis yang ingin meminimalkan kehilangan pelanggan. Namun, mendeteksi setengah dari churner tetap memberi nilai praktis dan menjadi langkah awal yang penting dalam menjawab problem statement pertama.

**Menemukan faktor yang memengaruhi churn:**

![Feature Importance](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Feature_Importance.png)

Berdasarkan hasil analisis feature importance dari model Gradient Boosting, fitur yang paling berpengaruh terhadap churn adalah tenure, diikuti oleh InternetService_Fiber optic, dan PaymentMethod_Electronic check. Hal ini menunjukkan bahwa lama berlangganan, jenis layanan internet, dan metode pembayaran merupakan faktor utama yang menentukan apakah seorang pelanggan akan churn atau tidak.

**Mengukur akurasi prediksi churn dengan machine learning:**

Dengan akurasi model di atas 79%, dan bahkan mencapai 80.07% pada cross validation Gradient Boosting, maka machine learning terbukti efektif dalam memprediksi churn berdasarkan data historis pelanggan.

### Evaluasi Terhadap Goals

**Membangun model prediktif churn yang akurat:**

![Churn](https://raw.githubusercontent.com/LapplandSa/Proyek-Predictive-Analytics/main/images/Churn.png)

Tercapai, ditunjukkan oleh model Gradient Boosting yang menghasilkan akurasi sebesar 0.8007 dan F1 score sebesar 0.5768, serta model improved yang tetap menunjukkan performa kompetitif. Gambar di atas menunjukkan hasil pengujian terhadap sejumlah pelanggan, beserta prediksi churn dan kesesuaiannya dengan label asli.

**Menentukan fitur yang paling berkontribusi:**

Tercapai dengan memanfaatkan model Gradient Boosting, yang tidak hanya memberikan performa prediksi terbaik, tetapi juga mampu mengidentifikasi fitur-fitur yang paling berpengaruh terhadap churn yakni tenure, InternetService_Fiber optic, dan PaymentMethod_Electronic check.

**Mengevaluasi dan memilih model terbaik:**

Tercapai dengan melalui perbandingan tiga algoritma utama dan satu model hasil improvement, kemudian dipilih model Gradient Boosting yang telah di cross validation sebagai final model dengan performa paling optimal secara keseluruhan.

### Dampak dari Solution Statements

**Penerapan algoritma beragam (KNN, RF, GB):**

Menunjukkan perbedaan performa dan kompleksitas yang memberikan insight terhadap efektivitas pendekatan sederhana (KNN) hingga kompleks (GB).

**Hyperparameter tuning:**

Memberikan hasil yang kompetitif dan meningkatkan recall, meskipun terjadi trade-off pada akurasi. Ini menegaskan bahwa tuning dapat disesuaikan dengan prioritas bisnis (misal: fokus deteksi churn sebanyak mungkin).

**Cross-validation:**

Cross-validation memberikan keyakinan bahwa performa model tidak hanya cocok di data uji, tetapi juga stabil dan dapat digeneralisasi, yang penting untuk memastikan keputusan bisnis berbasis model tetap dapat diandalkan saat diterapkan ke data pelanggan baru.
