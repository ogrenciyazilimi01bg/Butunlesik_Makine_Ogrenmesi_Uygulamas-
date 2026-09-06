# Butunlesik_Makine_Ogrenmesi_Uygulamasi-
Proje Ödevi - Bütünleşik Makine Öğrenmesi
Uygulaması
Giriş
Bu proje ödevinde amaç, dönem boyunca yapılan ödevlerde öğrenilen tüm yöntemleri tek
bir uygulamada birleştirmektir. Uygulama menü tabanlı bir yapıya sahiptir: kullanıcı
çalıştırmak istediği analizi menüden seçer, program ilgili yöntemi çalıştırır ve elde edilen
sonuçları bir dosyaya kaydeder. Böylece hem temel programlama becerileri (menü, koşul,
döngü, dosya işlemleri) hem de makine öğrenmesi yöntemleri (regresyon, kümeleme,
tekrarlayan sinir ağı, evrişimli sinir ağı) bir arada kullanılır.
Her analiz için varsayılan bir veri seti sağlanmıştır. İsteyen öğrenciler, aynı yapıyı
bozmadan kendi buldukları veri setlerini de kullanabilir; yeter ki veri, ilgili yöntemin
beklediği biçimde olsun (regresyon için sürekli hedefli tablo, kümeleme için etiketsiz tablo,
duygu analizi için metin, görüntü sınıflandırma için görüntü). Projenin genel yapısı
aşağıdaki gibidir:
Menü tabanlı yapı: kullanıcı bir analiz seçer, ilgili modül çalışır ve sonuçlar ortak bir dosyaya yazılır. Her
modül daha önceki ödevlerdeki yöntemlere karşılık gelir.
Genel Program Yapısı
Program bir menü döngüsü ile çalışır (Ödev #2'deki menü yapısına benzer şekilde).
Ekrana şu seçenekler yazdırılır:
● 1 - Regresyon Analizi
● 2 - Kümeleme Analizi
● 3 - Metin Duygu Analizi
● 4 - Görüntü Sınıflandırma
● 5 - Çıkış
Kullanıcı bir seçim yapar ve seçime göre ilgili modül çalıştırılır. Her modül
tamamlandığında elde edilen sonuçlar (kullanılan yöntem, başarı ölçütü ve değeri)
sonuclar.txt adlı bir dosyaya eklenir; bu, Ödev #3'teki dosyaya kayıt mantığının aynısıdır.
Menü, kullanıcı çıkışı seçene kadar tekrar eder.
Bölüm 1 - Regresyon Analizi (Ödev #4, #5, #6)
Bu bölümde, sürekli bir hedef değişkeni tahmin eden farklı regresyon modelleri eğitilip
karşılaştırılır. Varsayılan veri seti proje_ev_fiyatlari.csv olup 400 konuta ait beş özellik
(alan, oda_sayisi, bina_yasi, merkeze_uzaklik, kat) ve hedef değişken fiyat içerir.
Veri eğitim ve test kümesine ayrılır, özellikler standartlaştırılır ve şu modeller eğitilir:
Doğrusal Regresyon, Ridge, PCR, PLS ve bir yapay sinir ağı (MLP). Her modelin başarısı
MSE ve R2
 ile ölçülüp karşılaştırılır. Varsayılan veri setiyle elde edilecek sonuçlar yaklaşık
olarak şöyledir:
Model MSE R^2
---------------------------------------
Doğrusal Regresyon 43796 0.911
Ridge 43846 0.911
PCR (3 bileşen) 138874 0.719
PLS (3 bileşen) 50873 0.897
Yapay Sinir Ağı 23599 0.952
Yorum: Doğrusal modeller (Doğrusal, Ridge, PLS) benzer ve iyi sonuç verir; ancak veride
konuma bağlı doğrusal olmayan bir etki bulunduğundan yapay sinir ağı en düşük hatayı
elde eder. PCR, az sayıda bileşenle bilgi kaybettiği için daha düşük kalır; PLS ise hedefi de
dikkate aldığından PCR'den iyidir. Bu karşılaştırma, hangi modelin ne zaman uygun
olduğunu somut olarak gösterir.
Bölüm 2 - Kümeleme Analizi (Ödev #7)
Bu bölümde, etiketsiz bir veri seti K-Means ile gruplara ayrılır. Varsayılan veri seti
proje_musteri_segmentasyonu.csv olup 300 müşteriye ait iki özellik içerir: yıllık gelir
(yillik_gelir, TL) ve harcama puanı (harcama_puani). Veri setinde etiket yoktur.
Özellikler standartlaştırılır, dirsek (elbow) yöntemiyle uygun küme sayısı belirlenir,
K-Means uygulanır ve kümeler gelir-harcama düzleminde görselleştirilip yorumlanır. Bu
veri setinde gelir (on binlerce TL) ile harcama puanı (1–100) çok farklı ölçeklerde
olduğundan, standardizasyon sonucu belirgin biçimde etkiler; standartlaştırma yapılmazsa
gelir, uzaklık hesabına baskın gelir ve kümeleme bozulur.
Varsayılan veri setiyle dirsek yöntemi uygun küme sayısını K = 5 olarak işaret eder; bu K
için silhouette skoru yaklaşık 0.72'dir (kümeler iyi ayrışmıştır). Bulunan kümeler beş
müşteri segmentine karşılık gelir: düşük gelir–düşük harcama, düşük gelir–yüksek
harcama, orta gelir–orta harcama, yüksek gelir–düşük harcama ve yüksek gelir–yüksek
harcama.
Bölüm 3 - Metin Duygu Analizi (Ödev #8)
Bu bölümde, film yorumlarının olumlu mu olumsuz mu olduğunu tahmin eden bir
SimpleRNN modeli eğitilir. Varsayılan veri seti, Kaggle'daki “IMDB Dataset of 50K Movie
Reviews” veri setidir:
kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Metinler Tokenizer ve pad_sequences ile sayısal dizilere çevrilir; Embedding →
SimpleRNN → Dense (sigmoid) yapısındaki model kurulur, eğitilir ve test doğruluğu
ölçülür. İsteyen öğrenciler kendi metin/duygu veri setlerini kullanabilir.
Bölüm 4 - Görüntü Sınıflandırma (Ödev #9)
Bu bölümde, görüntüleri sınıflara ayıran bir evrişimli sinir ağı (CNN) eğitilir. Varsayılan veri
seti, Keras içinde hazır bulunan Fashion-MNIST'tir (28×28 gri tonlamalı kıyafet görüntüleri,
10 sınıf). Görüntüler ölçeklenir; Conv2D ve MaxPooling bloklarından oluşan bir CNN
kurulur, eğitilir ve test doğruluğu ölçülür. İsteyen öğrenciler kendi görüntü veri setlerini
kullanabilir.
Teslim
Öğrenciler proje kapsamında şunları teslim eder:
● Menü tabanlı programın kodu (dört analiz modülünü ve sonuçların dosyaya
kaydedildiği bölümü içeren).
● Programın çalıştırılmasıyla üretilen sonuç dosyası (sonuclar.txt).
● Her modül için kısa bir yorum içeren bir rapor: hangi veri seti kullanıldı, hangi
yöntem ve ayarlar seçildi, elde edilen sonuç ve bu sonucun yorumu. Kendi veri setini
kullananlar veri setinin kaynağını da belirtir.
Değerlendirmede; dört modülün de doğru çalışması, menü ve dosya yapısının düzgün
kurulması ve sonuçların doğru yorumlanması dikkate alınır.
