# TK KASDAD 2023/2024: Valorant Pro Matches

Repository TK KASDAD dengan topik **Valorant Pro Matches**

- [TK KASDAD 2023/2024: Valorant Pro Matches](#tk-kasdad-20232024-valorant-pro-matches)
  - [Informasi Umum](#informasi-umum)
    - [Panduan Pengerjaan](#panduan-pengerjaan)
  - [Dataset](#dataset)
    - [`matches.csv`](#matchescsv)
    - [`games.csv`](#gamescsv)
    - [`scores.csv`](#scorescsv)
    - [`agent_encode.csv`](#agent_encodecsv)
    - [`agent_role.csv`](#agent_rolecsv)
  - [Problem Guideline](#problem-guideline)

## Informasi Umum

Pengerjaan proyek dilakukan pada file utama: `TK_Data+.ipynb`

### Panduan Pengerjaan

Berikut adalah panduan pengerjaan secara umum terkait proyek akhir.

1. Semua kode yang ditulis pada file `TK_Data+.ipynb` harus disertakan **komen yang menjelaskan apa yang dilakukan oleh kode tersebut.** Gunakan komen secara bijaksana (tidak perlu komen *line-by-line*).
2. Gunakan penamaan variabel yang jelas, intuitif, dan konsisten.
    - Variabel yang menyimpan objek dengan class `DataFrame` harus diberi penamaan dengan awalan `df_`
    - Variabel yang menyimpan objek dengan class `Series` harus diberi penamaan dengan awalan `srs_`
    - Model *machine learning* yang dibuat masing-masing harus diikuti awalan yang mendeskripsikan model tersebut.

        | Model                    | awalan  |
        | ------------------------ | ------- |
        | Decision Tree Classifier | `dtc_`  |
        | Decision Tree Regressor  | `dtr_`  |
        | Random Forest Classifier | `rfc_`  |
        | Random Forest Regressor  | `rfr_`  |
        | K-NN Classifier          | `knnc_` |
        | K-NN Regressor           | `knnr_` |
        | ...                      | ...     |

        > Tambahakan konvensi penaaman variabel pada tabel di atas sesuai dengan kebutuhan.
    - Penamaan variabel untuk hal-hal lain, seperti `GridSearchCV` dan `SequentialFeatureSelector`,  dapat ditambahkan di sini. **Ingat untuk mengisi tabel di bawah setiap kali ada penambahan awalan baru.**

        | Informasi                 | awalan  |
        | ------------------------- | ------- |
        | SequentialFeatureSelector | `sfs_`  |
        | Custom Utility Class      | `util_` |
        | ...                       | ...     |
3. Perhatikan penggunaan dan penambahan header dan subheader pada `TK_Data+.ipynb` agar mudah untuk melakukan navigasi pada file tersebut.

    > :bulb: **Tips:** Pada VSCode, kalian dapat mengakses bagian `Outline` pada `Explorer` untuk melakukan navigasi pada file `TK_Data+.ipynb`.
4. Implementasikan semua prosedur untuk melakukan data pre-processing pada `Utility Function` dengan membuat method pada utility class yang sesuai.

## Dataset

Seluruh dataset terletak pada folder `dataset`.

> :memo: **Note:** Data lengkap jika dibutuhkan: <https://www.vlr.gg/>

### `matches.csv`

Berisi informasi tentang tim yang berpartisipasi dalam sebuah turnamen. Pertandingan umumnya diadakan dengan aturan "best of 3," di mana tim yang memenangkan babak terbanyak akan menjadi pemenang pertandingan.

> :memo: **Note:** Pertandingan final diadakan dengan aturan "best of 5"

Jumlah Kolom x Baris: 12 kolom x 6401 baris

Deskripsi Kolom:

1. MatchID : Id pertandingan
2. Date : Tanggal pertandingan
3. Patch : Versi Valorant saat pertandingan dilakukan
4. EventID : Id turnamen
5. EventName : Nama turnamen
6. EventStage : Tahap turnamen
7. Team1ID : Id tim 1
8. Team2ID : Id tim 2
9. Team1 : Nama tim 1
10. Team2 : Nama tim 2
11. Team1_MapScore : Banyak babak yang dimenangkan tim 1
12. Team2_MapScore : Banyak babak yang dimenangkan tim 2

### `games.csv`

Mendokumentasikan detail babak yang dimainkan oleh kedua tim yang bersaing dalam turnamen.

Jumlah Kolom x Baris: 18 kolom x 12961 baris

Deskripsi Kolom:

1. GameID : Id babak
2. MatchID : Id pertandingan
3. Map : Peta yang digunakan
4. Team1ID : Id tim 1
5. Team2ID : Id tim 2
6. Team1 : Nama tim 1
7. Team2 : Nama tim 2
8. Winner : Tim yang menjadi pemenang
9. Team1_Eco : Banyak ronde eco yang dilakukan tim 1
10. Team1_SemiEco : Banyak ronde semi eco yangdilakukan tim 1
11. Team1_SemiBuy : Banyak ronde semi buy yangdilakukan tim 1
12. Team1_FullBuy : Banyak ronde full buy yang dilakukan tim 1
13. Team1_TotalRounds : Banyak ronde yang dimenangkan tim 1
14. Team2_Eco : Banyak ronde eco yang dilakukan tim 2
15. Team2_SemiEco : Banyak ronde semi eco yang dilakukan tim 2
16. Team2_SemiBuy : Banyak ronde semi buy yang dilakukan tim 2
17. Team2_FullBuy : Banyak ronde full buy yang dilakukan tim 2
18. Team2_TotalRounds : Banyak ronde yang dimenangkan tim 2

> :memo: **Note:** Valorant Economy Guide: <https://www.ign.com/wikis/valorant/Valorant_Economy_Guide>
> Penting untuk tahap pemahaman data.

### `scores.csv`

Merupakan catatan akhir dari status para pemain profesional yang bertanding dalam babak tersebut

Jumlah Kolom x Baris: 28 kolom x 128779 baris

Deskripsi Kolom:

1. GameID : Id babak
2. PlayerID: Id pemain
3. PlayerName : Username pemain
4. TeamAbbreviation : Singkatan nama tim
5. Agent : Agent/karakter yang digunakan
6. ACS : Skor rata-rata dari pemain
7. Kills : Jumlah musuh yang dikalahkan
8. Deaths : Total pemain berhasil dieliminasi
9. Assists : Jumlah kontribusi pemain dalam membantu rekannya untuk mengalahkan musuh
10. PlusMinus : Selisih antara jumlah mengalahkan musuh (Kills) dan jumlah tereliminasi (Deaths)
11. KAST_Percent : Besar kontribusi pemain dalam pertandingan
12. ADR : Rata-rata serangan yang diberikan oleh pemain ke pihak musuh
13. HS_Percent : Persentase serangan yang diberikan mengenai kepala musuh
14. FirstKills : Total pemain berhasil melakukan eliminasi pertama kali pada suatu ronde
15. FirstDeaths : Total pemain dieliminasi pertama kali pada suatu ronde
16. FKFD_PlusMinus : Selisih antara jumlah First Kills dan First Deaths
17. Num_2Ks, Num_3ks, Num_4ks, Num_5ks : Jumlah pemain berhasil mengeliminasi X musuh dalam suatu ronde
18. OnevOne, OnevTwo, OnevThree, OnevFour, OnevFive : Jumlah pemain berhasil mengalahkan X musuh pada suatu timeframe
19. Econ : Besar serangan yang dihasilkan untuk setiap 1000 creds (mata uang pada permainan) yang digunakan untuk membeli perlengkapan
20. Plants : Jumlah pemain berhasil menanam spike (bom) pada suatu ronde
21. Defuses : Jumlah pemain berhasil menjinakkan spike (bom) pada suatu ronde

> :memo: **Notes:** Valorant Key Metric: <https://digitalgamersdream.com/key-metrics-for-analyzing-valorant-matches/>
> Penting untuk pemahaman terkait metric-metric tertentu, seperti ACS (Average Combat Score) dan KAST.

> Num_5ks = Ace

### `agent_encode.csv`

Label encoding dari nama agent di Valorant.

Jumlah Kolom x Baris: 2 kolom x 17 baris

Deskripsi Kolom:

1. Agent: nama karakter
2. Encode: hasil encoding

### `agent_role.csv`

Kategori dari agent-agent pada valorant

Jumlah Kolom x Baris: 2 kolom x 17 baris

Deskripsi Kolom:

1. Agent: nama karakter
2. Role: peran karakter

Kategori Kolom Role:

1. controllers: agen-agen dengan tipe ini dapat memblokir area tertentu di peta untuk mempermudah serangan atau pertahanan, serta mengontrol alur permainan.
2. duelists: agen-agen dengan tipe ini bersifat ofensif dan unggul dalam pertarungan satu lawan satu dengan tujuan untuk menerobos pertahanan musuh dan memenangkan duel.
3. initiators: agen-agen dengan tipe ini berfungsi sebagai pendukung dan sangat cocok untuk memulai pertarungan dan membantu tim Anda memenangkannya. Agen ini biasanya memiliki kemampuan yang memungkinkan mereka memberikan informasi kepada tim, atau mengacaukan musuh.
4. sentinels: agen-agen dengan tipe ini bersifat defensif dan biasanya memperlambat, mengulur waktu, atau terkadang langsung menghentikan serangan musuh di area tertentu di peta.

## Problem Guideline

1. Lakukan eksplorasi sederhana pada data dan ceritakan insight yang Anda dapatkan dari data tersebut! (dapat ditambah analisis lain yang menurut kelompok Anda akan bermanfaat)

   - [ ] Bagaimana distribusi Agent yang digunakan oleh pemain untuk setiap Patch 1.X, 2.X, dan 3.X? Apakah terdapat Agent yang mengalami penurunan pengguna pada patch yang berbeda?
   - [ ] Bagaimana persebaran nilai ACS dari setiap Agent yang digunakan? Apakah pemilihan map yang berbeda menghasilkan persebaran nilai ACS yang berbeda juga?
   - [ ] Siapa tim yang paling banyak memenangkan pertandingan grand final dari sebuah Event?
   - [ ] Komposisi tim seperti apakah yang paling banyak memenangkan pertandingan?
   - [ ] Mahasiswa perlu menambahkan dua poin eksplorasi sendiri

2. Buatlah beberapa model yang mungkin akan diperlukan, di antaranya: (dapat ditambah analisis lain yang menurut kelompok Anda akan bermanfaat)

    - [ ] Lakukan klasifikasi untuk menentukan Agent yang digunakan oleh setiap pemain pada setiap babak!
    - [ ] Buatlah model untuk memprediksi nilai rata-rata ACS dari masing-masing tim untuk sebuah pertandingan!
    - [ ] Lakukan clustering pada dataset, dan berikan analisis dan insight yang kaliat dapatkan dari cluster yang dibuat.
