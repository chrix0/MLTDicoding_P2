# Laporan Proyek Machine Learning - Chris Tianto Pratama

## Project Overview

Aktivitas menonton film di era digital sudah menjadi hal yang mudah dijumpai dalam kehidupan sehari-hari dan dapat dinikmati oleh siapa pun. Berbagai film dapat dengan mudah ditemukan dan ditonton melalui aplikasi *streaming* film atau *Subscription Video on Demand* (SVOD). Selain itu, adanya perangkat seperti *smartphone*, *tablet*, atau laptop dapat memudahkan pengguna, sehingga mereka dapat menonton film di mana dan kapan saja selama memiliki koneksi *internet* [3].

Aplikasi SVOD merupakan aplikasi penyedia layanan streaming film berbayar, di mana pengguna harus membayar setiap hari / minggu / bulan berdasarkan tarif yang ditentukan oleh penyedia layanan SVOD untuk dapat menikmati layanan atau fitur dari aplikasi tersebut. Aplikasi SVOD pada umumnya menyediakan banyak pilihan video atau film yang dapat ditonton oleh pengguna [4].

Salah satu aspek yang perlu diperhatikan oleh penyedia layanan streaming film adalah mengenai bagaimana pengguna menemukan film yang sesuai dengan preferensinya di antara kumpulan pilihan film yang berjumlah sangat banyak. Jumlah film atau video yang terlalu banyak dapat menyebabkan permasalahan *information overload*, di mana pengguna merasa kesulitan dalam memproses jumlah informasi yang sangat banyak. Tanpa pemrosesan informasi yang tepat, pembuatan keputusan menjadi sangat sulit untuk dilakukan [1], [2].

Permasalahan tersebut penting untuk diselesaikan karena hal tersebut dapat mempengaruhi *User Experience* (UX) dalam menggunakan aplikasi *streaming* film. Pencarian film secara berulang kali untuk menemukan film yang sesuai dengan preferensi tentu akan merepotkan pengguna serta membutuhkan waktu yang lama. UX yang buruk akan berpengaruh terhadap *Customer Loyalty*, yang menentukan apakah pengguna tersebut akan kembali membeli untuk menggunakan fitur-fitur dalam aplikasi SVOD [5]. Salah satu solusi yang dapat digunakan untuk menyelesaikan permasalahan tersebut adalah menyediakan fitur yang dapat memberikan rekomendasi berdasarkan kategori atau genre dari film yang disukai dan memberikan rekomendasi film yang belum pernah ditonton berdasarkan penilaian (*rating*) oleh pengguna lain dengan memanfaatkan *machine learning* dengan akurasi yang tinggi. Dengan solusi tersebut, pengguna dapat lebih mudah membuat keputusan saat memilih film yang ingin ditonton.

## Business Understanding

### Problem Statements

- Bagaimana cara memberikan rekomendasi film yang dipersonalisasi berdasarkan genre preferensi pengguna?
- Bagaimana cara memberikan rekomendasi film yang belum pernah ditonton berdasarkan rating yang pernah dibuat oleh pengguna lain?
- Algoritma apa yang memberikan memberikan rekomendasi film yang belum pernah ditonton berdasarkan rating yang pernah dibuat oleh pengguna lain dengan akurasi terbaik?

### Goals

- Membuat model machine learning yang dapat menghasilkan sejumlah rekomendasi film yang dipersonalisasi berdasarkan genre preferensi pengguna.
- Membuat model machine learning yang dapat menghasilkan sejumlah rekomendasi film yang belum pernah ditonton berdasarkan rating yang pernah dibuat oleh pengguna lain.
- Mengetahui algoritma yang dapat memberikan rekomendasi film yang belum pernah ditonton berdasarkan rating yang pernah dibuat oleh pengguna lain dengan akurasi tinggi.

### Solution Approaches

- Untuk menghasilkan rekomendasi film yang dipersonalisasi berdasarkan genre preferensi pengguna, model yang digunakan adalah model sistem rekomendasi Content Based Filtering dengan kombinasi algoritma TF-IDF dan Cosine Similiarity.
- Untuk menghasilkan rekomendasi film yang belum pernah ditonton berdasarkan rating yang pernah dibuat oleh pengguna lain, model yang digunakan adalah model sistem rekomendasi Collaborative Filtering.
- Untuk mengetahui algoritma model sistem rekomendasi Collaborative Filtering terbaik, perbandingan atau evaluasi terhadap Singular Value Decomposition (SVD) dan K-Nearest Neighbor (KNN) akan dilakukan dengan 10-Fold Cross Validation dengan metrik evaluasi RMSE (Root Mean Square Error) dan MAE (Mean Absolute Error).


## Data Understanding

Berikut informasi dari dataset yang digunakan.

Tabel 1. Informasi dataset
|   |   |
|---|---|
|__Nama dataset__| ml-latest-small (MovieLens Latest Datasets - Small)|
|__Deskripsi dataset__| This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018. |
|__Jumlah jenis data__| 4 |
|__Link download__| https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |

Dataset ini berisi empat jenis data, yaitu:
- movies 
- ratings
- links 
- tags

Berikut informasi dari keempat jenis data tersebut.

Tabel 2. Informasi jenis data movies
|   |   |
|---|---|
|__Nama jenis data__| movies ||
|__Jumlah sampel__| 9742 |
|__Jumlah variabel__| 3 |

Tabel 3. Informasi jenis data ratings
|   |   |
|---|---|
|__Nama jenis data__| ratings |
|__Jumlah sampel__| 100836 |
|__Jumlah variabel__| 4 |

Tabel 4. Informasi jenis data links
|   |   |
|---|---|
|__Nama jenis data__| links |
|__Jumlah sampel__| 3683 |
|__Jumlah variabel__| 3 |

Tabel 5. Informasi jenis data tags
|   |   |
|---|---|
|__Nama jenis data__| tags |
|__Jumlah sampel__| 9742 |
|__Jumlah variabel__| 4 |

Dalam proyek ini, jenis data yang digunakan adalah movies dan ratings.

### Exploratory Data Analysis

#### 1. Deksripsi Variabel

Berikut variabel / fitur yang digunakan dalam movies dan ratings.

Tabel 6. Deskripsi variabel pada jenis data movies
| Nama fitur | Deskripsi | Tipe data |
|---|---|---|
|movieId| ID dari film berdasarkan website [TMDB](https://www.themoviedb.org) |int64|
|title| Judul film |object|
|genres| Genre film, yang dipisah dengan tanda pipe ('\|') |object|

Tabel 7. Deskripsi variabel pada jenis data ratings
| Nama fitur | Deskripsi | Tipe data |
|---|---|---|
|userId| ID dari pengguna website MovieLens yang memberikan rating |int64|
|movieId| ID dari film yang diberi rating |int64|
|rating| Nilai yang diberikan dalam skala 0.5 - 5, dengan peningkatan nilai sebesar 0.5 |float64|
|timestamp| Waktu pemberian rating dalam satuan detik |int64|

Fitur timestamp dalam ratings tidak digunakan dalam pemberian rekomendasi, sehingga fitur tersebut dihapus dalam dataset ratings.

#### 2. Pemeriksaan Missing Value

Dalam tahap ini dilakukan pemeriksaan data dengan missing value pada movies dan ratings. Kedua dataset tersebut tidak memiliki data dengan missing value, sehingga tidak dilakukan penghapusan data.

#### 3. Univariate Analysis
Univariate analysis merupakan analisis dari setiap atau masing-masing variabel penelitian yang memberikan informasi atau ringkasan data yang dikumpulkan sehingga diperoleh informasi yang jelas. Informasi ini dapat disajikan dalam bentuk tabel, grafik atau ukuran statistik lainnya [6].

##### 3.1. Movies

###### Fitur movieId

Fitur movieId menggunakan angka untuk menyatakan ID dari film yang tersedia. Terdapat 9742 movie ID unik dalam dataset movies. Jumlah movieID unik sama dengan jumlah data pada dataset movies (9742). Hal ini menunjukkan bahwa setiap data film memiliki movieId yang berbeda.

###### Fitur title

Terdapat 9737 judul film unik dalam dataset movies. Jumlah title unik lebih sedikit daripada jumlah data pada dataset movies (9742). Hal ini menunjukkan bahwa terdapat 5 judul duplikat yang perlu dihapus.

###### Fitur genre

Terdapat 19 genre unik dalam dataset movies, yaitu Drama, Comedy, Thriller, Action, Romance, Adventure, Crime, Sci-Fi, Horror, Fantasy, Children, Animation, Mystery, Documentary, War, Musical, Western, IMAX, Film-Noir, dan genre kosong (no genres listed). Tabel 8 dan Gambar 1 menunjukkan jumlah data pada dataset movies berdasarkan genre.

Tabel 8. Jumlah data berdasarkan genre pada dataset movies
|index|Genre|Jumlah data|persentase|
|---|---|---|---|
|0|Drama|4361|19\.75|
|1|Comedy|3756|17\.01|
|2|Thriller|1894|8\.58|
|3|Action|1828|8\.28|
|4|Romance|1596|7\.23|
|5|Adventure|1263|5\.72|
|6|Crime|1199|5\.43|
|7|Sci-Fi|980|4\.44|
|8|Horror|978|4\.43|
|9|Fantasy|779|3\.53|
|10|Children|664|3\.01|
|11|Animation|611|2\.77|
|12|Mystery|573|2\.59|
|13|Documentary|440|1\.99|
|14|War|382|1\.73|
|15|Musical|334|1\.51|
|16|Western|167|0\.76|
|17|IMAX|158|0\.72|
|18|Film-Noir|87|0\.39|
|19|\(no genres listed\)|34|0\.15|

![Grafik julah data berdasarkan genre pada dataset movies](https://i.ibb.co/zRk4Gxv/jlhdata-genre.png)

Gambar 1. Grafik jumlah data berdasarkan genre pada dataset movies

##### 3.2. Ratings

###### Fitur userId

Fitur userId menggunakan angka untuk menyatakan ID pengguna. Terdapat 610 userId unik unik pada dataset ratings. Hal ini menunjukkan bahwa 610 pengguna secara keseluruhan telah membuat 100836 rating. Tabel 9 dan Gambar 2 menunjukkan 10 pengguna yang memiliki jumlah rating terbanyak.

Tabel 9. 10 Pengguna dengan jumlah rating terbanyak
|userId|Jumlah rating|
|---|---|
|414|2698|
|599|2478|
|474|2108|
|448|1864|
|274|1346|
|610|1302|
|68|1260|
|380|1218|
|606|1115|
|288|1055|

![10 Pengguna dengan jumlah rating terbanyak](https://i.ibb.co/S7vFKjt/jlhdata-user-Id-rating.png)

Gambar 2. Grafik 10 Pengguna dengan jumlah rating terbanyak

Tabel 9 dan Gambar 2 menunjukkan bahwa pengguna yang memberikan rating terbanyak adalah pengguna dengan userId 414, yang telah membuat 2698 rating. Pengguna dengan jumlah rating yang paling sedikit dalam dataset ini adalah 20.

###### Fitur movieId

Terdapat 9724 movieId unik pada dataset rating. Jumlah film yang dirating tidak sama dengan jumlah film keseluruhan pada dataset movies (9742). Hal ini menunjukkan bahwa terdapat 18 film yang belum diberi rating.

###### Fitur rating

Terdapat 10 kemungkinan nilai dalam pemberian rating, yaitu 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, dan 0.5. Hal ini menunjukkan bahwa skala rating yang digunakan adalah 0.5 - 5.0. Tabel 10 dan Gambar 3 menunjukkan jumlah data pada dataset ratings berdasarkan rating yang diberikan.

Tabel 10. Jumlah data berdasarkan rating yang diberikan pengguna
|rating|jumlah data|
|---|---|
|0\.5|1370|
|1\.0|2811|
|1\.5|1791|
|2\.0|7551|
|2\.5|5550|
|3\.0|20047|
|3\.5|13136|
|4\.0|26818|
|4\.5|8551|
|5\.0|13211|

![Jumlah data berdasarkan rating](https://i.ibb.co/NZB9L1T/jlhdata-rating.png)

Gambar 3. Grafik jumlah data berdasarkan rating yang diberikan pengguna

Tabel 10 dan Grafik 3 menunjukkan jumlah data berdasarkan rating dari yang terbanyak hingga yang paling sedikit. Jumlah data terbanyak adalah data yang memberikan rating 4.0, sedangkan jumlah data paling sedikit adalah data yang memberikan rating 0.5.

## Data Preparation

Dalam kasus ini, data preparation terdiri atas 5 tahapan, yaitu:
1. Penanganan judul film duplikat pada dataset movies
2. Penanganan data film yang memiliki tidak memiliki genre pada dataset movies
3. Penggabungan dataset movies dengan ratings
4. Penanganan data film yang belum memiliki rating pada dataset gabungan
5. Pembagian data training dan data testing pada dataset ratings

### 1. Penanganan judul film duplikat pada dataset movies

Pada tahap ini, data film dengan judul duplikat akan dihapus. Tahap ini bertujuan untuk mencegah pemberian dua rekomendasi film yang sama oleh model Content Based Filtering dan Collaborative Filtering. Tabel 11 dan Tabel 12 menunjukkan data film duplikat yang terdapat pada dataset movies.

Tabel 11. Data duplikat pada dataset movies.
|index|movieId|title|genres|
|---|---|---|---|
|650|838|Emma \(1996\)|Comedy,Drama,Romance|
|2141|2851|Saturn 3 \(1980\)|Adventure,Sci-Fi,Thriller|
|4169|6003|Confessions of a Dangerous Mind \(2002\)|Comedy,Crime,Drama,Thriller|
|5601|26958|Emma \(1996\)|Romance|
|5854|32600|Eros \(2004\)|Drama|
|5931|34048|War of the Worlds \(2005\)|Action,Adventure,Sci-Fi,Thriller|
|6932|64997|War of the Worlds \(2005\)|Action,Sci-Fi|
|9106|144606|Confessions of a Dangerous Mind \(2002\)|Comedy,Crime,Drama,Romance,Thriller|
|9135|147002|Eros \(2004\)|Drama,Romance|
|9468|168358|Saturn 3 \(1980\)|Sci-Fi,Thriller|

Tabel 12. Jumlah data film duplikat berdasarkan judul film
|index|title|jumlah data|
|---|---|---|
|0|Emma \(1996\)|2|
|1|War of the Worlds \(2005\)|2|
|2|Confessions of a Dangerous Mind \(2002\)|2|
|3|Eros \(2004\)|2|
|4|Saturn 3 \(1980\)|2|

Salah satu data film duplikat dihapus, sehingga setiap film memiliki satu data saja. Setelah penhapusan data duplikat, data movies berkurang dari 9742 menjadi 9737.

### 2. Penanganan data film yang memiliki tidak memiliki genre pada dataset movies

Pada tahap ini, semua data film yang memiliki genre kosong (*no genres listed*) akan dihapus. Tabel 8 menunjukkan bahwa terdapat 34 data dengan genre kosong yang perlu dihapus dalam dataset movies. Tahap ini bertujuan untuk mencegah pemberian rekomendasi film tanpa genre kepada pengguna oleh model Content Based Filtering, karena ada kemungkinan bahwa film yang direkomendasikan tidak sesuai dengan preferensi pengguna. Tabel 13 menunjukkan data movies yang memiliki genre kosong.

Tabel 13. Daftar data movies yang bergenre kosong.
|index|movieId|title|genres|
|---|---|---|---|
|8517|114335|La cravate \(1957\)|\(no genres listed\)|
|8684|122888|Ben-hur \(2016\)|\(no genres listed\)|
|8687|122896|Pirates of the Caribbean: Dead Men Tell No Tales \(2017\)|\(no genres listed\)|
|8782|129250|Superfast\! \(2015\)|\(no genres listed\)|
|8836|132084|Let It Be Me \(1995\)|\(no genres listed\)|
|8902|134861|Trevor Noah: African American \(2013\)|\(no genres listed\)|
|9033|141131|Guardians \(2016\)|\(no genres listed\)|
|9053|141866|Green Room \(2015\)|\(no genres listed\)|
|9070|142456|The Brand New Testament \(2015\)|\(no genres listed\)|
|9091|143410|Hyena Road|\(no genres listed\)|
|9138|147250|The Adventures of Sherlock Holmes and Doctor Watson|\(no genres listed\)|
|9178|149330|A Cosmic Christmas \(1977\)|\(no genres listed\)|
|9217|152037|Grease Live \(2016\)|\(no genres listed\)|
|9248|155589|Noin 7 veljestä \(1968\)|\(no genres listed\)|
|9259|156605|Paterson|\(no genres listed\)|
|9307|159161|Ali Wong: Baby Cobra \(2016\)|\(no genres listed\)|
|9316|159779|A Midsummer Night's Dream \(2016\)|\(no genres listed\)|
|9348|161008|The Forbidden Dance \(1990\)|\(no genres listed\)|
|9413|165489|Ethel & Ernest \(2016\)|\(no genres listed\)|
|9426|166024|Whiplash \(2013\)|\(no genres listed\)|
|9448|167570|The OA|\(no genres listed\)|
|9478|169034|Lemonade \(2016\)|\(no genres listed\)|
|9514|171495|Cosmos|\(no genres listed\)|
|9515|171631|Maria Bamford: Old Baby|\(no genres listed\)|
|9518|171749|Death Note: Desu nôto \(2006–2007\)|\(no genres listed\)|
|9525|171891|Generation Iron 2|\(no genres listed\)|
|9534|172497|T2 3-D: Battle Across Time \(1996\)|\(no genres listed\)|
|9541|172591|The Godfather Trilogy: 1972-1990 \(1992\)|\(no genres listed\)|
|9562|173535|The Adventures of Sherlock Holmes and Doctor Watson: The Hunt for the Tiger \(1980\)|\(no genres listed\)|
|9573|174403|The Putin Interviews \(2017\)|\(no genres listed\)|
|9611|176601|Black Mirror|\(no genres listed\)|
|9661|181413|Too Funny to Fail: The Life and Death of The Dana Carvey Show \(2017\)|\(no genres listed\)|
|9663|181719|Serving in Silence: The Margarethe Cammermeyer Story \(1995\)|\(no genres listed\)|
|9669|182727|A Christmas Story Live\! \(2017\)|\(no genres listed\)|

Setelah data-data tersebut dihapus, jumlah data movies telah berkurang dari 9737 menjadi 9703.

###  3. Penggabungan dataset movies dengan ratings

Pada tahap ini, data movies dan ratings akan digabungkan berdasarkan movieId. Tahap ini bertujuan untuk menunjukkan data film yang belum memiliki rating dan mempersiapkan data sebelum melakukan pelatihan model Content Based Filtering.

Setelah dilakukan penggabungan data movies dan ratings, dataframe akan dihasilkan seperti yang ditampilkan pada Tabel 14.

Tabel 14. Hasil penggabungan dataset movies dan ratings
|index|movieId|title|genres|userId|rating|
|---|---|---|---|---|---|
|0|1|Toy Story \(1995\)|Adventure,Animation,Children,Comedy,Fantasy|1\.0|4\.0|
|1|1|Toy Story \(1995\)|Adventure,Animation,Children,Comedy,Fantasy|5\.0|4\.0|
|2|1|Toy Story \(1995\)|Adventure,Animation,Children,Comedy,Fantasy|7\.0|4\.5|
|3|1|Toy Story \(1995\)|Adventure,Animation,Children,Comedy,Fantasy|15\.0|2\.5|
|4|1|Toy Story \(1995\)|Adventure,Animation,Children,Comedy,Fantasy|17\.0|4\.5|
...|...|...|...|...|...|

Tabel 14 menunjukkan rating yang diberikan oleh berbagai pengguna terhadap film yang tersedia.

### 4. Penanganan data film yang belum memiliki rating pada dataset gabungan

Pada tahap ini, data yang belum memiliki rating pada dataset gabungan movies dan ratings akan dihapus. Tahap ini bertujuan untuk mencegah pemberian rekomendasi film yang belum memiliki rating oleh model Collaborative Filtering.

Untuk memudahkan pencarian data film yang belum memiliki rating, semua rating oleh pengguna akan dirata-ratakan dan dikelompokkan berdasarkan judul film. Tabel 15 menunjukkan hasil dari proses tersebut.

Tabel 15. Hasil perhitungan rata-rata terhadap fitur rating berdasarkan judul film
|title|movieId|rating|
|---|---|---|
|Karlson Returns \(1970\)|172585|5\.0|
|Enter the Void \(2009\)|78836|5\.0|
|English Vinglish \(2012\)|99636|5\.0|
|...|...|...|
|Road Home, The \(Wo de fu qin mu qin\) \(1999\)|6668|NaN|
|Roaring Twenties, The \(1939\)|25855|NaN|
|Scrooge \(1970\)|6849|NaN|

Tabel 15 menunjukkan bahwa terdapat beberapa data dengan missing value (NaN) pada fitur rating. Data-data yang memiliki nilai rating rata-rata NaN dapat dilihat pada Tabel 16.

Tabel 16. Data-data yang memiliki nilai rating rata-rata NaN
|title|movieId|rating|
|---|---|---|
|Browning Version, The \(1951\)|34482|NaN|
|Call Northside 777 \(1948\)|32371|NaN|
|Chalet Girl \(2011\)|85565|NaN|
|Chosen, The \(1981\)|5721|NaN|
|Color of Paradise, The \(Rang-e khoda\) \(1999\)|3456|NaN|
|For All Mankind \(1989\)|3338|NaN|
|I Know Where I'm Going\! \(1945\)|4194|NaN|
|In the Realms of the Unreal \(2004\)|30892|NaN|
|Innocents, The \(1961\)|1076|NaN|
|Mutiny on the Bounty \(1962\)|26085|NaN|
|Niagara \(1953\)|2939|NaN|
|Parallax View, The \(1974\)|7792|NaN|
|Proof \(1991\)|7020|NaN|
|Road Home, The \(Wo de fu qin mu qin\) \(1999\)|6668|NaN|
|Roaring Twenties, The \(1939\)|25855|NaN|
|Scrooge \(1970\)|6849|NaN|
|This Gun for Hire \(1942\)|8765|NaN|
|Twentieth Century \(1934\)|32160|NaN|

Tabel 16 menunjukkan bahwa terdapat 18 data film yang belum memiliki rating oleh pengguna. Data-data tersebut dihapus dalam dataset gabungan movies dan ratings, sehingga jumlah data film berkurang dari 9703 menjadi 9685.

### 5. Pembagian data training dan data testing pada dataset ratings

Pada tahap ini, dataset ratings dibagi menjadi dua bagian, yaitu data training (`trainset`) dan data testing (`testset`). Pembagian dataset dilakukan dengan metode yang berasal dari library [Surprise](https://surpriselib.com/), yaitu:
- `build_full_trainset`: Menggunakan keseluruhan dataset untuk dijadikan trainset tanpa membaginya menjadi kumpulan folds.
- `build_anti_testset`: Mengambil rating-rating yang tidak terdapat pada trainset untuk dijadikan sebagai data testing.

Tahap ini bertujuan untuk mempersiapkan data ratings yang akan digunakan dalam pelatihan dan pengujian model Collaborative Filtering dengan metode [Cross Validation yang disediakan oleh library Surprise](https://surprise.readthedocs.io/en/stable/model_selection.html#module-surprise.model_selection.split).

## Modeling and Result

Pada tahap ini, model machine learning akan dikembangkan dengan algoritma yang telah ditentukan. Model machine learning yang akan dibuat adalah model sistem rekomendasi Content Based Filtering dan Collaborative Filtering. Berikut pembahasan mengenai kedua jenis sistem rekomendasi tersebut dan algoritma-algoritma yang akan digunakan dalam proyek ini.

### Pembahasan Jenis Sistem Rekomendasi dan Algoritma yang Digunakan

#### 1. Content Based Filtering

Content Based Filtering merupakan teknik dalam sistem rekomendasi yang menggunakan nilai fitur dalam data atau item sebagai dasar dalam pemberian rekomendasi. Metode ini akan mengekstrak informasi yang terdapat pada item kemudian membandingkannya dengan informasi item yang pernah dilihat atau disukai oleh user [7].

Kelebihan Content Based Filtering:
- Model Content Based Filtering tidak memerlukan data pengguna lain, karena rekomendasi jenis ini diberikan berdasarkan data yang berasal oleh pengguna itu sendiri.
- Model Content Based Filtering dapat digunakan untuk mengetahui preferensi atau minat spesifik pengguna, dan dapat merekomendasikan item khusus yang mungkin sangat diminati oleh beberapa pengguna lain.

Kekurangan Content Based Filtering:
- Pengembangan model Content Based Filtering memerlukan pengetahuan mengenai domain atau bidang terkait. 
- Model Content Based Filtering hanya dapat membuat rekomendasi berdasarkan minat pengguna itu sendiri. Dengan kata lain, model memiliki kemampuan terbatas untuk memperluas minat pengguna tersebut.

Dalam proyek ini, metode yang digunakan dalam pemberian rekomendasi dengan Content Based Filtering adalah kombinasi *Term Frequency – Inverse Document Frequency* (TF-IDF) dengan *Cosine Similarity*. TF-IDF berfungsi untuk memberikan pembobotan terhadap data, dan *Cosine Similiarity* digunakan untuk membandingkan kemiripan suatu data dengan data lainnya berdasarkan hasil pembobotan dari TF-IDF.

##### 1.1 Term Frequency – Inverse Document Frequency (TF-IDF)

Algoritma *Term Frequency – Inverse Document Frequency* (TF-IDF) merupakan algoritma yang berasal dari bidang information retrieval, yang biasanya digunakan dalam perbandingan dokumen. Algoritma ini digunakan untuk menentukan bobot dari suatu kata (t) pada suatu dokumen (d) [8].

Term Frequency (TF) merupakan bobot dari suatu kata (t) dalam dokumen (d), yang ditentukan dengan melihat jumlah kemunculan kata dalam suatu dokumen. Untuk mengurangi efek dari kata yang frekuensi kemunculannya terlalu tinggi,  *Inverse Document Frequency* (IDF) dapat digunakan untuk mengurangi bobot dari kata dengan frekuensi kolektif (frekuensi total kemunculan kata di semua dokumen) yang tinggi. Oleh karena itu, semakin banyak frekuensi kemunculan kata, maka nilai bobot menjadi semakin rendah [8].

Kelebihan TF-IDF:
- Komputasi bersifat mudah.
- Dapat dengan mudah mengekstrak kata-kata yang paling deskriptif dalam suatu dokumen

Kekurangan TF-IDF:
- TF-IDF menghitung kesamaan dokumen secara langsung di *word-count space*, yang mungkin lambat jika jumlah kata unik sangat banyak.
- TF-IDF mengasumsikan bahwa jumlah kata yang berbeda memberikan pembukti kesamaan yang independen.
- TF-IDF tidak menggunakan kesamaan semantik antara kata-kata.

##### 1.2 Cosine Similarity

*Cosine Similarity* merupakan metode pengukuran kemiripan (*similarity measure*) yang banyak digunakan dalam kasus *pattern recognition* dan *text classification*. Cosine Similarity bekerja dengan mengukur kemiripan dua buah vektor dalam sebuah *product space* dengan mengukur nilai cosine dari sudut antara kedua vektor. Semakin besar hasil fungsi similarity, maka kedua objek yang dievaluasi semakin mirip, demikian pula sebaliknya [8]. Berikut formula yang digunakan dalam *Cosine Similarity*.

![Formula Cosine Similarity](https://i.ibb.co/QXY9jbT/cosines.png)

Keterangan:
- x :  Vektor x yang dibentuk dengan dokumen x
- y : Vektor y yang dibentuk dengan dokumen y
- [ x ] : Panjang vektor x
- [ y ] : Panjang vektor y

Kelebihan *Cosine Similarity*:
- Kompleksitas komputasi yang rendah
- Tidak terpengaruh pada panjang pendeknya suatu dokumen 
- Memiliki tingkat akurasi yang tinggi

Kekurangan *Cosine Similarity*:
- Tidak cocok digunakan untuk membandingkan data nominal.

#### 2. Collaborative Filtering

Collaborative Filtering merupakan teknik dalam sistem rekomendasi yang menggunakan opini atau rating dari pengguna lain untuk memprediksi suatu item yang mungkin merupakan preferensi dari pengguna tersebut [7]. Collaborative Filtering dapat dibagi menjadi dua bagian, yaitu *memory-based* dan *model-based*. Teknik *memory-based* dapat dibagi menjadi dua jenis, yaitu: *user-based* dan *item-based*. Teknik *model-based* menggunakan metode seperti *Matrix factorization*, *Neural network*, dll untuk melatih model.

Kelebihan Collaborative Filtering:
-  Pengembangan model Collaborative Filtering tidak memerlukan pengetahuan mengenai domain atau bidang terkait
-  Model Collaborative Filtering dapat membantu pengguna dalam menentukan minat baru sesuai dengan penilaian atau opini pengguna lain.

Kekurangan Collaborative Filtering:
- Tidak dapat menangani item baru (*cold start*)
- Sulit untuk menyertakan fitur sampingan untuk kueri/item selain fitur penilaian.

Kualitas rekomendasi dari sistem rekomendasi Collaborative Filtering bergantung pada opini pengguna lain terhadap suatu item. Opini atau rating pengguna lain dapat dianggap sebagai *neighbor*. Untuk meningkatkan kualitas rekomendasi sistem rekomendasi tersebut, upaya yang dapat dilakukan adalah dengan melakukan reduksi neighbor, yaitu memotong neighbor hingga ditemukan beberapa pengguna yang memiliki kesamaan (similiarity) tertinggi [7].

Dalam proyek ini, teknik *collaborative filtering* yang akan digunakan adalah *model-based*. Terdapat dua metode Collaborative Filtering yang akan dibandingkan performanya, yaitu teknik matrix factorization dengan Singular Value Decomposition (SVD) dan algoritma K-nearest neighbors (KNN)

##### 2.1 Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) merupakan teknik *matrix factorization* yang mendekomposisi matriks A yang berukuran apa saja (m x n) untuk mempermudah pengolahan data dan pemberian rekomendasi [9]. Berikut formula yang digunakan dalam SVD:

SVD(A) = U x S x Vᵗ

Keterangan:
A = Matriks dengan ukuran m x n
U = Matriks orthogonal dengan ukuran m x m
S = Matriks singular, yaitu matriks yang determinannya bernilai nol.
Vᵗ = Matriks orthogonal dengan ukuran n x n yang telah ditranspos

Kelebihan SVD:
- Memiliki kompleksitas yang rendah dan efisien.
- Dasarnya hierarkis, diurutkan berdasarkan relevansi.
- Memiliki performa yang baik terhadap sebagian besar dataset.

Kekurangan SVD:
- Jika data sangat non-linear, teknik ini mungkin tidak akan berfungsi dengan baik.
- Hasil yang diberikan kurang sesuai untuk dijadikan visualisasi data.
- Sulit untuk ditafsirkan.
- Sangat terfokus pada varians, terkadang tidak ada hubungan langsung antara varians dan kekuatan prediksi sehingga dapat membuang informasi yang berguna.

##### 2.2 K-nearest neighbors

Algoritma K-Nearest Neighbor (KNN) merupakan algoritma yang bekerja dengan mengambil sejumlah data terdekat (tetangganya) dengan jumlah k sebagai acuan untuk menentukan atau memprediksi suatu data. Algoritma KNN dapat digunakan untuk kasus-kasus klasifikasi dan regresi. Dalam Collaborative Filtering, algoritma ini dapat memberikan rekomendasi berdasarkan similarity atau kemiripan dari suatu data terhadap data lainnya [11].

Kelebihan algoritma KNN:

- KNN bersifat sangat nonlinear, karena KNN merupakan algoritma pembelajaran yang bersifat non-parametrik, yang berarti algoritma ini tidak mengasumsikan apa-apa mengenai distribusi instance dalam data maupun dokumen
- KNN mudah dipahami dan diimplementasikan

Kekurangan algoritma KNN:

- Perlu memberikan parameter K (jumlah tetangga terdekat).
- Tidak menangani missing value secara implisit.
- Sensitif terhadap data outlier.
- Rentan terhadap variabel yang tidak informatif.
- Rentan terhadap dimensionalitas (jumlah variabel) yang tinggi, karena semakin banyak dimensi, ruang yang bisa ditempati instance semakin besar, sehingga semakin besar pula kemungkinan bahwa neighbor terdekat dari suatu instance sebenarnya memiliki jarak yang sangat jauh.
- Rentan terhadap perbedaan rentang variabel.
- Memiliki nilai komputasi yang tinggi.

### Pengembangan Model dan Pemberian Rekomendasi

#### 1. Content Based Filtering

##### 1.1 Pembobotan genre dengan TF-IDF

Dalam pengembangan Content Based Filtering, Genre-genre setiap data film dalam dataset gabungan movies dan ratings akan diberi bobot terlebih dahulu dengan teknik TF-IDF. Teknik tersebut dapat dilakukan dengan metode `TfidfVectorizer()` yang berasal dari [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Tabel 17 menunjukkan matriks TF-IDF yang dihasilkan setelah mengimplementasi teknik TF-IDF.

Tabel 17. Matriks TF-IDF dengan 5 sampel.
|title|Drama|Western|Musical|Adventure|Film-Noir|Crime|Thriller|IMAX|Fantasy|Horror|Action|Animation|Mystery|Comedy|Romance|Documentary|Sci-Fi|War|Children|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Insomnia \(2002\)|0\.2795932899443996|0\.0|0\.0|0\.0|0\.0|0\.4799136016052382|0\.409129446047134|0\.0|0\.0|0\.0|0\.4140603390113626|0\.0|0\.593866701581522|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|
|Doors, The \(1991\)|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|
|Fantasia \(1940\)|0\.0|0\.0|0\.5677951716979863|0\.0|0\.0|0\.0|0\.0|0\.0|0\.4576930064662924|0\.0|0\.0|0\.4892744360131963|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.47826381955179204|
|April Morning \(1988\)|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|
|Rainmaker, The \(1997\)|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|

##### 1.2 Pengukuran kemiripan data film dengan Cosine Similarity

Untuk menentukan kemiripan antara satu data film dengan data film lainnya, metode yang digunakan adalah *Cosine Similarity*. Metode yang digunakan untuk mengimplementasi *Cosine Similarity* adalah ` cosine_similarity()` yang berasal dari [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html). Tabel 18 menunjukkan contoh hasil perhitungan Cosine Similiarity antara satu data film dengan data film lainnya.

Tabel 18. Hasil perhitungan Cosine Similarity terhadap 5 sampel.
|title|Bones \(2001\)|The Meddler \(2016\)|Loaded Weapon 1 \(National Lampoon&\#39;s Loaded Weapon 1\) \(1993\)|Passion of the Christ, The \(2004\)|Last Kiss, The \(2006\)|
|---|---|---|---|---|---|
|Toy Story \(1995\)|0\.0|0\.2|0\.16|0\.0|0\.2|
|Jumanji \(1995\)|0\.0|0\.0|0\.0|0\.0|0\.0|
|Grumpier Old Men \(1995\)|0\.0|0\.42|0\.34|0\.0|0\.42|
|Waiting to Exhale \(1995\)|0\.0|0\.69|0\.3|0\.47|0\.69|
|Father of the Bride Part II \(1995\)|0\.0|0\.73|0\.59|0\.0|0\.73|

Hasil perhitungan Cosine Similarity akan menjadi dasar dalam pemberian rekomendasi kepada pengguna.

Tabel 19 menunjukkan hasil rekomendasi dalam bentuk Top 5 rekomendasi film untuk film yang berjudul  "Jumanji (1995)" dengan genre Adventure, Children, dan Fantasy.

Tabel 19. Contoh hasil top 5 rekomendasi film dengan Content Based Filtering
|index|title|movieId|genres|rating|
|---|---|---|---|---|
|0|Bridge to Terabithia \(2007\)|50601|Adventure,Children,Fantasy|2\.78|
|1|Harry Potter and the Sorcerer's Stone \(a\.k\.a\. Harry Potter and the Philosopher's Stone\) \(2001\)|4896|Adventure,Children,Fantasy|3\.76|
|2|Chronicles of Narnia: The Voyage of the Dawn Treader, The \(2010\)|82169|Adventure,Children,Fantasy|3\.86|
|3|Return to Oz \(1985\)|2093|Adventure,Children,Fantasy|3\.0|
|4|NeverEnding Story, The \(1984\)|2161|Adventure,Children,Fantasy|3\.58|

#### 2. Collaborative Filtering

##### 2.1 Hyperparameter Tuning

Dalam pengembangan Collaborative Filtering, model yang menerapkan teknik SVD dan model dengan algoritma KNN akan terlebih dahulu dicari parameter terbaik dengan *hyperparameter tuning*. *Hyperparameter tuning* bertujuan untuk melakukan memberikan *improvement* terhadap model yang akan dilatih dengan mengatur parameter-parameter dalam metode algoritma yang digunakan.

Dalam kasus ini proses hyperparameter tuning dilakukan dengan *Grid Search*. Grid Search dilakukan dengan metode [GridSearchCV() yang berasal dari library surprise](https://surprise.readthedocs.io/en/stable/model_selection.html#surprise.model_selection.search.GridSearchCV) untuk mendukung metode algoritma yang juga berasal dari library surprise.

Tabel 20 menunjukkan parameter terbaik untuk setiap algoritma dengan nilai Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE) terendah.

Tabel 20. Hasil Hyperparameter Tuning
|index|model|best\_score|best\_params|
|---|---|---|---|
|0|SVD|\{'rmse': 0\.8705362246862414, 'mae': 0\.6687825907387696\}|\{'rmse': \{'n\_factors': 50, 'lr\_all': 0\.005, 'reg\_all': 0\.02\}, 'mae': \{'n\_factors': 50, 'lr\_all': 0\.005, 'reg\_all': 0\.02\}\}|
|1|KNNBasic|\{'rmse': 0\.9391932568212908, 'mae': 0\.7208409369513925\}|\{'rmse': \{'k': 40, 'min\_k': 2\}, 'mae': \{'k': 40, 'min\_k': 2\}\}|

Parameter-parameter yang tertera pada Tabel 20 akan digunakan dalam pengembangan model SVD dan KNN.

##### 2.2 Pemberian rekomendasi dengan model SVD dan KNN

Setelah pelatihan model SVD dan KNN, kedua model tersebut dapat digunakan untuk memberikan rekomendasi berdasarkan rating oleh pengguna lain.

Misalkan pengguna yang akan diberi rekomendasi adalah pengguna dengan userId 610. Pengguna tersebut telah memberikan rating terhadap 1300 film. Tabel 21 menunjukkan 10 film yang diberi rating tertinggi oleh pengguna tersebut.

Tabel 21. 10 film dengan rating tertinggi oleh pengguna dengan userId 610
|index|movieId|title|genres|rating rata-rata|rating user 610|
|---|---|---|---|---|---|
|0|1|Toy Story \(1995\)|Adventure,Animation,Children,Comedy,Fantasy|3\.92|5\.0|
|394|5833|Dog Soldiers \(2002\)|Action,Horror|4\.67|5\.0|
|296|4437|Suspiria \(1977\)|Horror|3\.94|5\.0|
|1207|122920|Captain America: Civil War \(2016\)|Action,Sci-Fi,Thriller|3\.61|5\.0|
|317|4794|Opera \(1987\)|Crime,Horror,Mystery|3\.0|5\.0|
|321|4848|Mulholland Drive \(2001\)|Crime,Drama,Film-Noir,Mystery,Thriller|3\.84|5\.0|
|326|4878|Donnie Darko \(2001\)|Drama,Mystery,Sci-Fi,Thriller|3\.98|5\.0|
|1201|122882|Mad Max: Fury Road \(2015\)|Action,Adventure,Sci-Fi,Thriller|3\.82|5\.0|
|341|4993|Lord of the Rings: The Fellowship of the Ring, The \(2001\)|Adventure,Fantasy|4\.11|5\.0|
|1200|121231|It Follows \(2014\)|Horror|4\.12|5\.0|

Berdasarkan rating yang diberikan terhadap film-film oleh pengguna tersebut, berikut rekomendasi yang diberikan dalam bentuk Top 10 rekomendasi film dengan model SVD dan KNN.

Tabel 22. Top 10 rekomendasi film dengan model SVD
|index|movieId|title|genres|rating|
|---|---|---|---|---|
|585|720|Wallace & Gromit: The Best of Aardman Animation \(1996\)|Adventure,Animation,Comedy|4\.09|
|679|898|Philadelphia Story, The \(1940\)|Comedy,Drama,Romance|4\.31|
|680|899|Singin' in the Rain \(1952\)|Comedy,Musical,Romance|4\.07|
|866|1148|Wallace & Gromit: The Wrong Trousers \(1993\)|Animation,Children,Comedy,Crime|4\.04|
|881|1178|Paths of Glory \(1957\)|Drama,War|4\.54|
|959|1262|Great Escape, The \(1963\)|Action,Adventure,Drama,War|4\.13|
|1216|1617|L\.A\. Confidential \(1997\)|Crime,Film-Noir,Mystery,Thriller|4\.06|
|1530|2067|Doctor Zhivago \(1965\)|Drama,Romance,War|4\.14|
|1732|2329|American History X \(1998\)|Crime,Drama|4\.22|
|2279|3030|Yojimbo \(1961\)|Action,Adventure|4\.23|

Tabel 23. Top 10 rekomendasi film dengan model KNN
|index|movieId|title|genres|rating|
|---|---|---|---|---|
|48|53|Lamerica \(1994\)|Adventure,Drama|5\.0|
|87|99|Heidi Fleiss: Hollywood Madam \(1995\)|Documentary|5\.0|
|868|1151|Lesson Faust \(1994\)|Animation,Comedy,Drama,Fantasy|5\.0|
|2591|3473|Jonah Who Will Be 25 in the Year 2000 \(Jonas qui aura 25 ans en l'an 2000\) \(1976\)|Comedy|5\.0|
|4381|6442|Belle époque \(1992\)|Comedy,Romance|5\.0|
|4387|6460|Trial, The \(Procès, Le\) \(1962\)|Drama|4\.9|
|4580|6818|Come and See \(Idi i smotri\) \(1985\)|Drama,War|5\.0|
|5001|7767|Best of Youth, The \(La meglio gioventù\) \(2003\)|Drama|4\.75|
|5564|26810|Bad Boy Bubby \(1993\)|Drama|4\.83|
|7230|74282|Anne of Green Gables: The Sequel \(a\.k\.a\. Anne of Avonlea\) \(1987\)|Children,Drama,Romance|4\.75|

Kedua model tersebut menghasilkan rekomendasi yang sangat berbeda. Model KNN memberikan rekomendasi film dengan rating yang lebih tinggi daripada model SVD. 

## Evaluation

### 1. Content Based Filtering

Model Content Based Filtering yang dihasilkan akan diuji dengan metrik evaluasi *Prediction at k* (Prediction@k). Prediction@k merupakan pembagian jumlah item relevan yang direkomendasikan dengan jumlah rekomendasi yang diberikan [11]. Berikut formula dari metrik tersebut.

![Formula Precision@k](https://i.ibb.co/3SwbdDG/precisionatkform.png)

Keterangan:
- `Recom-and-Relev-Items`: jumlah item relevan yang direkomendasikan
- `Recom-Items`: Jumlah item rekomendasi yang akan diperiksa (k)

Suatu item atau data dalam hasil rekomendasi akan dianggap relevan jika nilainya sama dengan nilai aktual. Sebaliknya, item akan dianggap tidak relevan jika nilainya tidak sama dengan nilai aktual [11].

Dalam kasus ini, fitur yang akan dievaluasi adalah fitur genre. Genre film yang direkomendasikan akan dibandingkan dengan genre dari judul film *input*. Evaluasi akan dilakukan terhadap hasil rekomendasi dari 10 sampel input. Berikut data film yang akan digunakan sebagai input.

Tabel 24. Data-data input untuk menguji hasil rekomendasi
|index|movieId|title|genres|rating|
|---|---|---|---|---|
|6965|67087|I Love You, Man \(2009\)|Comedy|3\.53|
|7611|87660|Too Big to Fail \(2011\)|Drama|4\.5|
|9013|141749|The Danish Girl \(2015\)|Drama|3\.75|
|1100|1431|Beverly Hills Ninja \(1997\)|Action,Comedy|2\.47|
|5580|26903|Whisper of the Heart \(Mimi wo sumaseba\) \(1995\)|Animation,Drama,Romance|3\.8|
|1214|1615|Edge, The \(1997\)|Adventure,Drama|3\.0|
|7805|93022|Miss Nobody \(2010\)|Comedy,Crime|5\.0|
|1509|2040|Computer Wore Tennis Shoes, The \(1969\)|Children,Comedy|2\.86|
|6711|59118|Happy-Go-Lucky \(2008\)|Comedy,Drama|3\.5|
|8264|106002|Ender's Game \(2013\)|Action,Adventure,Sci-Fi,IMAX|3\.44|

Berikut hasil pengujian model Content Based Filtering yang telah dibuat dengan metrik Prediction@k dalam bentuk persen dengan k = 5.

Tabel 25. Hasil pengujian dengan Prediction@5
|index|Nilai Prediction@k %|
|---|---|
|0|100\.0|
|1|100\.0|
|2|100\.0|
|3|100\.0|
|4|100\.0|
|5|100\.0|
|6|100\.0|
|7|100\.0|
|8|100\.0|
|9|100\.0|

Semua hasil pengujian dengan Prediction@5 memiliki nilai 100%. Hal ini menunjukkan bahwa model tersebut dapat memberikan rekomendasi film berdasarkan genre yang dipilih atau disukai dengan akurat. 

### 2. Collaborative Filtering

Evaluasi model Collaborative Filtering dilakukan dengan Cross Validation. Metode Cross Validation yang digunakan adalah **K-Fold**. K-Fold merupakan metode evaluasi model *machine learning* yang menguji model dengan membagi dataset menjadi k bagian dengan ukuran yang sama. k-1 bagian digunakan untuk training dan 1 bagian sisanya digunakan untuk testing. K-Fold Cross Validation akan melakukan proses silang dimana data testing dijadikan sebagai data training dan sebaliknya data training sebelumnya dijadikan sebagai data testing [12].

K-Fold Cross Validation dapat dilakukan dengan metode `cross_validation()` yang berasal dari [library surprise](https://surprise.readthedocs.io/en/stable/model_selection.html#surprise.model_selection.validation.cross_validate). Untuk dapat menggunakan metode tersebut, metode algoritma yang digunakan harus berasal dari library tersebut. 

Terdapat dua metrik evaluasi yang akan digunakan, yaitu RMSE dan MAE.

#### 2.1 Pembahasan metrik evaluasi yang digunakan

##### Root Mean Square Error (RMSE)

Root Mean Square Error (RMSE) merupakan salah satu pengukuran kesalahan atau error antara nilai yang diprediksi dengan nilai aktual [13]. Berikut formula untuk metrik ini:

![Formula RMSE](https://i.ibb.co/y5W8Nkd/rmse.png)

Keterangan:

- n = jumlah data
- yᵢ = nilai aktual
- ŷᵢ = nilai prediksi

Perhitungan RMSE dilakukan dengan cara mengkuadratkan hasil perjumlahan nilai selisih antara nilai aktual dengan nilai prediksi setiap data, lalu hasil perjumlahan tersebut dibagi dengan n untuk menentukan nilai rata-ratanya. Setelah itu, operasi akar akan dilakukan terhadap nilai rata-rata tersebut agar satuan dari RMSE sama dengan satuan nilai aktual. Nilai RMSE yang rendah atau mendekati nol menunjukkan bahwa hasil prediksi sesuai dengan data aktual.

##### Mean Absolute Error (MAE)

*Mean Absolute Error* (MAE) juga merupakan salah satu pengukuran kesalahan atau error antara nilai yang diprediksi dengan nilai aktual [13]. Berikut formula perhitungan MAE:

![Formula MAE](https://i.ibb.co/p4zgXQk/mae.png)

Keterangan:

- n = jumlah data
- yᵢ = nilai aktual
- ŷᵢ = nilai prediksi

Perhitungan MAE dilakukan dengan melakukan pengurangan nilai aktual dengan nilai prediksi setiap data, dimana hasil pengurangan tersebut selalu merupakan bilangan positif, yang kemudian dijumlahkan secara keseluruhan dan membaginya dengan jumlah data yang ada. Nilai MAE yang rendah atau mendekati nol menunjukkan bahwa hasil prediksi sesuai dengan data aktual.

#### 2.2 Hasil Evaluasi

Tabel 26 dan 27 menunjukkan hasil Cross Validation untuk model SVD dan KNN dengan metode 10-fold.

Tabel 26. Hasil 10-fold Cross Validation model SVD 
||Fold 1|Fold 2|Fold 3|Fold 4|Fold 5|Fold 6|Fold 7|Fold 8|Fold 9|Fold 10|Mean|Std|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
**RMSE (testset)**|    0.8778|  0.8577|  0.8758|  0.8719|  0.8647|  0.8573|  0.8696|  0.8581|  0.8612|  0.8644|  0.8659|  0.0072| 
**MAE (testset)**|     0.6728|  0.6585|  0.6696|  0.6691|  0.6644|  0.6587|  0.6682|  0.6609|  0.6628|  0.6597|  0.6645|  0.0049|  
**Fit time**|          3.80|    3.88|    4.49|    3.78|    3.83|    3.80|    3.80|    3.87|    3.88|    3.81|    3.90|    0.20|  
**Test time**|         0.07|    0.10|    0.07|    0.11|    0.09|    0.07|    0.07|    0.07|    0.08|    0.08|    0.08|    0.01|   

Tabel 27. Hasil 10-fold Cross Validation model KNN
||Fold 1|Fold 2|Fold 3|Fold 4|Fold 5|Fold 6|Fold 7|Fold 8|Fold 9|Fold 10|Mean|Std|
|---|---|---|---|---|---|---|---|---|---|---|---|---|   
**RMSE (testset)**|    0.9306|  0.9356|  0.9220|  0.9243|  0.9377|  0.9403|  0.9432|  0.9248|  0.9331|  0.9429|  0.9334|  0.0074|  
**MAE (testset)**|     0.7136|  0.7185|  0.7117|  0.7060|  0.7209|  0.7202|  0.7226|  0.7116|  0.7160|  0.7244|  0.7165|  0.0055|  
**Fit time**|          0.19|    0.23|    0.21|    0.22|    0.22|    0.22|    0.22|    0.21|    0.22|    0.21|    0.22|    0.01|    
**Test time**|         0.86|    0.93|    0.86|    0.89|    0.85|    0.90|    0.87|    0.90|    0.86|    0.95|    0.89|    0.03| 

Tabel 26 dan Tabel 27 menunjukkan bahwa model SVD memiliki nilai mean RMSE dan MAE yang lebih rendah daripada model KNN. Oleh karena itu, model SVD memberikan hasil rekomendasi *Collaborative Filtering* yang lebih akurat daripada model KNN.

## Kesimpulan

- Model Content Based Filtering dengan kombinasi teknik TF-IDF dan algoritma Cosine Similarity menghasilkan rekomendasi film yang akurat, yang ditunjukkan dengan nilai 100 dalam pengukuran mean Prediction@5 terhadap 10 sampel.
- Model Collaborative Filtering dapat dikembangkan dengan teknik faktorisasi matriks seperti SVD atau dengan algoritma KNN untuk menentukan persamaan preferensi antar pengguna berdasarkan rating.
- Di antara teknik SVD dan algoritma KNN , metode yang dapat memberikan rekomendasi film yang belum pernah ditonton berdasarkan rating yang pernah dibuat oleh pengguna lain dengan akurasi terbaik adalah teknik SVD. Nilai RMSE dan MAE untuk teknik SVD tergolong baik, dan bisa diimprovisasi lebih lanjut dengan metode lain, seperti menggunakan teknik SVD yang telah diimprovisasi (SVD++) atau dengan KNN Baseline.


## Referensi

[1] S. Reddy, S. Nalluri, S. Kunisetti, S. Ashok, and B. Venkatesh, “[Content-Based Movie Recommendation System Using Genre Correlation](https://link.springer.com/chapter/10.1007/978-981-13-1927-3_42),” in Smart Intelligent Computing and Applications, Singapore, 2019, pp. 391–397. doi: 10.1007/978-981-13-1927-3_42.

[2] J. Zhang, Y. Wang, Z. Yuan, and Q. Jin, “[Personalized real-time movie recommendation system: Practical prototype and evaluation](https://ieeexplore.ieee.org/abstract/document/8821512),” Tsinghua Science and Technology, vol. 25, no. 2, pp. 180–191, Apr. 2020, doi: 10.26599/TST.2018.9010118.

[3] T. O. Wibowo, “[Fenomena Website Streaming Film di Era Media Baru: Godaan, Perselisihan, dan Kritik](http://jurnal.unpad.ac.id/jkk/article/view/15623),” Jurnal Kajian Komunikasi, vol. 6, no. 2, Art. no. 2, Dec. 2018, doi: 10.24198/jkk.v6i2.15623.

[4] N. Juliandhono and M. P. Berlianto, “[FAKTOR-FAKTOR YANG MEMPENGARUHI PERCEIVED VALUE DAN IMPLIKASINYA KEPADA INTENTION TO SUBSCRIBE SERTA PENGARUHNYA TERHADAP SOCIAL INFLUENCE PADA APLIKASI STREAMING FILM DISNEY PLUS HOTSTAR](https://jurnalpemasaran.petra.ac.id/index.php/mar/article/view/24822),” Jurnal Manajemen Pemasaran, vol. 16, no. 2, Art. no. 2, Oct. 2022, doi: 10.9744/pemasaran.16.2.77-86.

[5] D. A. Wongso, “[ANALISA USER EXPERIENCE TERHADAP CUSTOMER LOYALTY DENGAN TRUST SEBAGAI VARIABEL INTERVENING PADA APLIKASI OVO DIGITAL PAYMENT](https://publication.petra.ac.id/index.php/manajemen-pemasaran/article/view/10026),” Jurnal Strategi Pemasaran, vol. 7, no. 1, Art. no. 1, Jul. 2020.

[6] S. Riyanto and A. A. Hatmawan, [Metode Riset Penelitian Kuantitatif Penelitian Di Bidang Manajemen, Teknik, Pendidikan Dan Eksperimen](https://books.google.co.id/books?id=W2vXDwAAQBAJ&printsec=copyright&hl=id#v=onepage&q&f=false). Deepublish, 2020.

[7] A. Wijaya and D. Alfian, “[Sistem Rekomendasi Laptop Menggunakan Collaborative Filtering Dan Content-Based Filtering](http://www.jurnal.stmik-mi.ac.id/index.php/jcb/article/view/167),” Jurnal Computech & Bisnis (e-Journal), vol. 12, no. 1, Art. no. 1, Jun. 2018, doi: 10.55281/jcb.v12i1.167.

[8] A. R. Harischandra, M. F. A. Pratama, F. Felix, and A. P. Laia, “[Aplikasi Pendukung Desain Interior dengan Sistem Rekomendasi Berdasarkan Nama Brand Perabot Menggunakan Algoritma Content-Based Filtering Berbasis Web](https://www.mikroskil.ac.id/ejurnal/index.php/jsm/article/view/816),” Jurnal SIFO Mikroskil, vol. 23, no. 1, Art. no. 1, Apr. 2022, doi: 10.55601/jsm.v23i1.816.

[9] M. G. Vozalis and K. G. Margaritis, “[Using SVD and demographic data for the enhancement of generalized Collaborative Filtering](https://www.sciencedirect.com/science/article/abs/pii/S0020025507001223),” Information Sciences, vol. 177, no. 15, pp. 3017–3037, Aug. 2007, doi: 10.1016/j.ins.2007.02.036.

[10] B. S. Fitrianti, M. Fachurrozi, and N. Yusliani, “[Sistem Rekomendasi Artikel Ilmiah Berbasis Web Menggunakan Content-based Learning dan Collaborative Filtering](http://generic.ilkom.unsri.ac.id/index.php/generic/article/view/81),” Generic, vol. 10, no. 1, Art. no. 1, Jan. 2018.

[11] S. Airen and J. Agrawal, “[Movie Recommender System Using K-Nearest Neighbors Variants](https://link.springer.com/article/10.1007/s40009-021-01051-0),” Natl. Acad. Sci. Lett., vol. 45, no. 1, pp. 75–82, Feb. 2022, doi: 10.1007/s40009-021-01051-0.

[12] T. Fushiki, “[Estimation of prediction error by using K-fold cross-validation](https://link.springer.com/article/10.1007/s11222-009-9153-8),” Stat Comput, vol. 21, no. 2, pp. 137–146, Apr. 2011, doi: 10.1007/s11222-009-9153-8.

[13] T. O. Hodson, “[Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not](https://gmd.copernicus.org/articles/15/5481/2022/),” Geoscientific Model Development, vol. 15, no. 14, pp. 5481–5487, Jul. 2022, doi: 10.5194/gmd-15-5481-2022.