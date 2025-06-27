
1. Clustering Nilai Mahasiswa dengan K-Means

Proyek ini bertujuan untuk mengelompokkan (clustering) data nilai mahasiswa berdasarkan nilai UTS dan UAS menggunakan algoritma K-Means. Dataset yang digunakan berupa file CSV berisi data mahasiswa dan nilai-nilainya.


2.  Teknologi yang Digunakan

- Python 3
- Pandas
- Matplotlib
- Scikit-learn (sklearn)
- Jupyter Notebook



3. Struktur Dataset

Dataset nilaimahasiswa.csv  memiliki kolom:

- Mahasiswam: Nama mahasiswa
- uts: Nilai UTS
- uas : Nilai UAS



4. Langkah-Langkah Analisis

1) Import Library
python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


2) Load dan Eksplorasi Data

python
dfmhs = pd.read_csv('nilaimahasiswa.csv', sep=';', engine='python')
print(dfmhs.describe())
print(dfmhs.info())


3) Visualisasi Awal (Scatter Plot)

python
plt.scatter(dfmhs['uts'], dfmhs['uas'])
plt.xlabel("UTS")
plt.ylabel("UAS")
plt.title("Grafik Nilai Mahasiswa")


4) Persiapan Data dan Normalisasi

python
x_train = dfmhs[['uts','uas']].values
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)


5) Clustering dengan K-Means (k=3)

python
kmean = KMeans(n_clusters=3)
y_cluster = kmean.fit_predict(x_train)
dfmhs['cluster'] = y_cluster


6) Visualisasi Hasil Clustering

python
plt.scatter(x_train[:,0], x_train[:,1], c=kmean.labels_)
plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1], marker='*', s=150)
plt.xlabel("UTS")
plt.ylabel("UAS")
plt.legend()


7) Menentukan K Optimal (Elbow Method)

python
inertias = []
k_range = range(1, 10)
for k in k_range:
    km = KMeans(n_clusters=k).fit(x_train)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias)
plt.xlabel("k")
plt.ylabel("Sum of Error")
plt.grid()




5. Hasil Akhir

* Dataset berhasil dikelompokkan menjadi 3 cluster berdasarkan kemiripan nilai.
* Centroid dari masing-masing cluster berhasil divisualisasikan.
* Jumlah cluster optimal ditentukan menggunakan **metode Elbow**.



6. File Penting

* nilaimahasiswa.csv – Dataset nilai UTS dan UAS
* kmeans_clustering.ipynb – Notebook Jupyter
* README.md – Dokumentasi proyek


7. Kontributor

* Individu
* Dhilan Djalani Kusuma Putra
* Kampus: Universitas Pamulang (UNPAM)
* Proyek: UAS Data Mining - Clustering Nilai Mahasiswa






