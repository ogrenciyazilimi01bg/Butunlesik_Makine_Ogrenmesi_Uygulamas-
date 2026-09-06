# ================================================================
# BÜTÜNLEŞİK MAKİNE ÖĞRENMESİ UYGULAMASI
# ================================================================
# Proje:
# 1 - Regresyon Analizi
# 2 - Kümeleme Analizi
# 3 - Metin Duygu Analizi - SimpleRNN
# 4 - Görüntü Sınıflandırma - CNN
# 5 - Çıkış
#
# Google Colab + Google Drive uyumludur.
# ================================================================


# ================================================================
# 0. KÜTÜPHANELER
# ================================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score
)

# TensorFlow / Keras
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping

# TensorFlow uyarılarını azalt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Tekrarlanabilir sonuçlar
np.random.seed(42)
tf.random.set_seed(42)


# ================================================================
# 1. GOOGLE DRIVE
# ================================================================

from google.colab import drive

print("\nGoogle Drive bağlanıyor...")
drive.mount('/content/drive')

print("Google Drive bağlantısı tamamlandı.")


# ================================================================
# 2. DOSYA YOLLARI
# ================================================================
# Dosyalarının bulunduğu klasör:
# MyDrive/Tech İstanbul/
#
# Buradaki isimleri Drive'daki gerçek dosya isimlerinle aynı
# olacak şekilde bıraktık.
# ================================================================

ANA_KLASOR = "/content/drive/MyDrive/Tech Istanbul"

REGRESYON_DOSYASI = os.path.join(
    ANA_KLASOR,
    "proje_ev_fiyatlari.csv"
)

KUMELEME_DOSYASI = os.path.join(
    ANA_KLASOR,
    "proje_musteri_segmentasyonu.csv"
)

IMDB_DOSYASI = os.path.join(
    ANA_KLASOR,
    "IMDB Dataset.csv"
)

SONUC_DOSYASI = os.path.join(
    ANA_KLASOR,
    "sonuclar.txt"
)

GRAFIK_KLASORU = os.path.join(
    ANA_KLASOR,
    "grafikler"
)

os.makedirs(GRAFIK_KLASORU, exist_ok=True)


# ================================================================
# 3. SONUÇ DOSYASINA YAZMA FONKSİYONU
# ================================================================

def sonuc_kaydet(metin):
    """
    Sonuçları sonuclar.txt dosyasına ekler.
    """

    with open(
        SONUC_DOSYASI,
        "a",
        encoding="utf-8"
    ) as dosya:

        dosya.write(metin)
        dosya.write("\n")

    print("\nSonuçlar dosyaya kaydedildi:")
    print(SONUC_DOSYASI)


# ================================================================
# 4. DOSYA KONTROLÜ
# ================================================================

def dosya_kontrol(dosya_yolu):

    if not os.path.exists(dosya_yolu):

        print("\nHATA!")
        print("Dosya bulunamadı:")
        print(dosya_yolu)

        print("\nLütfen dosyanın Google Drive'da")
        print("Tech İstanbul klasörünün içinde olduğundan emin olun.")

        return False

    return True


# ================================================================
# 5. CSV SÜTUN TEMİZLEME
# ================================================================

def sutunlari_temizle(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# ================================================================
# 6. BÖLÜM 1
# REGRESYON ANALİZİ
# ================================================================

def regresyon_analizi():

    print("\n")
    print("=" * 70)
    print("BÖLÜM 1 - REGRESYON ANALİZİ")
    print("=" * 70)

    # ------------------------------------------------------------
    # Dosya kontrolü
    # ------------------------------------------------------------

    if not dosya_kontrol(REGRESYON_DOSYASI):
        return

    print("\nVeri seti okunuyor...")

    df = pd.read_csv(REGRESYON_DOSYASI)

    df = sutunlari_temizle(df)

    print("\nVeri setinin ilk 5 satırı:")
    print(df.head())

    print("\nVeri seti boyutu:")
    print(df.shape)

    # ------------------------------------------------------------
    # Beklenen sütunlar
    # ------------------------------------------------------------

    beklenen_sutunlar = [
        "alan",
        "oda_sayisi",
        "bina_yasi",
        "merkeze_uzaklik",
        "kat",
        "fiyat"
    ]

    eksik_sutunlar = [
        sutun
        for sutun in beklenen_sutunlar
        if sutun not in df.columns
    ]

    if len(eksik_sutunlar) > 0:

        print("\nHATA!")
        print("Eksik sütunlar:")
        print(eksik_sutunlar)

        print("\nBulunan sütunlar:")
        print(list(df.columns))

        return

    # ------------------------------------------------------------
    # Eksik değer kontrolü
    # ------------------------------------------------------------

    print("\nEksik değerler:")
    print(df[beklenen_sutunlar].isnull().sum())

    df = df[beklenen_sutunlar].dropna()

    # ------------------------------------------------------------
    # X ve y
    # ------------------------------------------------------------

    X = df[
        [
            "alan",
            "oda_sayisi",
            "bina_yasi",
            "merkeze_uzaklik",
            "kat"
        ]
    ]

    y = df["fiyat"]

    print("\nÖzellikler:")
    print(list(X.columns))

    print("\nHedef değişken: fiyat")

    # ------------------------------------------------------------
    # Eğitim / test ayrımı
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nEğitim veri sayısı:", len(X_train))
    print("Test veri sayısı:", len(X_test))

    # ------------------------------------------------------------
    # STANDARDİZASYON
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # SONUÇLAR
    # ------------------------------------------------------------

    sonuclar = []

    # ============================================================
    # MODEL 1 - DOĞRUSAL REGRESYON
    # ============================================================

    print("\n")
    print("-" * 70)
    print("1. Doğrusal Regresyon")
    print("-" * 70)

    linear_model = LinearRegression()

    linear_model.fit(
        X_train_scaled,
        y_train
    )

    linear_pred = linear_model.predict(
        X_test_scaled
    )

    linear_mse = mean_squared_error(
        y_test,
        linear_pred
    )

    linear_r2 = r2_score(
        y_test,
        linear_pred
    )

    print("MSE :", round(linear_mse, 2))
    print("R²  :", round(linear_r2, 4))

    sonuclar.append(
        [
            "Doğrusal Regresyon",
            linear_mse,
            linear_r2
        ]
    )

    # ============================================================
    # MODEL 2 - RIDGE REGRESYON
    # ============================================================

    print("\n")
    print("-" * 70)
    print("2. Ridge Regresyon")
    print("-" * 70)

    ridge_model = Ridge(
        alpha=1.0
    )

    ridge_model.fit(
        X_train_scaled,
        y_train
    )

    ridge_pred = ridge_model.predict(
        X_test_scaled
    )

    ridge_mse = mean_squared_error(
        y_test,
        ridge_pred
    )

    ridge_r2 = r2_score(
        y_test,
        ridge_pred
    )

    print("MSE :", round(ridge_mse, 2))
    print("R²  :", round(ridge_r2, 4))

    sonuclar.append(
        [
            "Ridge",
            ridge_mse,
            ridge_r2
        ]
    )

    # ============================================================
    # MODEL 3 - PCR
    # ============================================================

    print("\n")
    print("-" * 70)
    print("3. PCR - Principal Component Regression")
    print("-" * 70)

    pcr_model = Pipeline(
        [
            (
                "pca",
                PCA(n_components=3)
            ),
            (
                "regression",
                LinearRegression()
            )
        ]
    )

    pcr_model.fit(
        X_train_scaled,
        y_train
    )

    pcr_pred = pcr_model.predict(
        X_test_scaled
    )

    pcr_mse = mean_squared_error(
        y_test,
        pcr_pred
    )

    pcr_r2 = r2_score(
        y_test,
        pcr_pred
    )

    print("Bileşen sayısı: 3")
    print("MSE :", round(pcr_mse, 2))
    print("R²  :", round(pcr_r2, 4))

    sonuclar.append(
        [
            "PCR (3 bileşen)",
            pcr_mse,
            pcr_r2
        ]
    )

    # ============================================================
    # MODEL 4 - PLS
    # ============================================================

    print("\n")
    print("-" * 70)
    print("4. PLS - Partial Least Squares")
    print("-" * 70)

    pls_model = PLSRegression(
        n_components=3
    )

    pls_model.fit(
        X_train_scaled,
        y_train
    )

    pls_pred = pls_model.predict(
        X_test_scaled
    )

    # PLS bazen (n,1) şeklinde sonuç döndürür
    pls_pred = np.ravel(pls_pred)

    pls_mse = mean_squared_error(
        y_test,
        pls_pred
    )

    pls_r2 = r2_score(
        y_test,
        pls_pred
    )

    print("Bileşen sayısı: 3")
    print("MSE :", round(pls_mse, 2))
    print("R²  :", round(pls_r2, 4))

    sonuclar.append(
        [
            "PLS (3 bileşen)",
            pls_mse,
            pls_r2
        ]
    )

    # ============================================================
    # MODEL 5 - YAPAY SİNİR AĞI / MLP
    # ============================================================

    print("\n")
    print("-" * 70)
    print("5. Yapay Sinir Ağı - MLP")
    print("-" * 70)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        random_state=42
    )

    mlp_model.fit(
        X_train_scaled,
        y_train
    )

    mlp_pred = mlp_model.predict(
        X_test_scaled
    )

    mlp_mse = mean_squared_error(
        y_test,
        mlp_pred
    )

    mlp_r2 = r2_score(
        y_test,
        mlp_pred
    )

    print("Gizli katmanlar: (64, 32)")
    print("Aktivasyon: ReLU")
    print("MSE :", round(mlp_mse, 2))
    print("R²  :", round(mlp_r2, 4))

    sonuclar.append(
        [
            "Yapay Sinir Ağı (MLP)",
            mlp_mse,
            mlp_r2
        ]
    )

    # ============================================================
    # SONUÇ TABLOSU
    # ============================================================

    sonuc_df = pd.DataFrame(
        sonuclar,
        columns=[
            "Model",
            "MSE",
            "R2"
        ]
    )

    print("\n")
    print("=" * 70)
    print("REGRESYON MODELLERİ KARŞILAŞTIRMASI")
    print("=" * 70)

    print(
        sonuc_df.to_string(
            index=False
        )
    )

    # ============================================================
    # EN İYİ MODEL
    # ============================================================

    en_iyi_mse = sonuc_df.loc[
        sonuc_df["MSE"].idxmin()
    ]

    en_iyi_r2 = sonuc_df.loc[
        sonuc_df["R2"].idxmax()
    ]

    print("\nEn düşük MSE:")
    print(
        en_iyi_mse["Model"],
        "->",
        round(en_iyi_mse["MSE"], 2)
    )

    print("\nEn yüksek R²:")
    print(
        en_iyi_r2["Model"],
        "->",
        round(en_iyi_r2["R2"], 4)
    )

    # ============================================================
    # GRAFİK 1 - MSE
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        sonuc_df["Model"],
        sonuc_df["MSE"]
    )

    plt.title(
        "Regresyon Modellerinin MSE Karşılaştırması"
    )

    plt.xlabel("Model")
    plt.ylabel("MSE")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    mse_grafik = os.path.join(
        GRAFIK_KLASORU,
        "regresyon_mse.png"
    )

    plt.savefig(
        mse_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # GRAFİK 2 - R2
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        sonuc_df["Model"],
        sonuc_df["R2"]
    )

    plt.title(
        "Regresyon Modellerinin R² Karşılaştırması"
    )

    plt.xlabel("Model")
    plt.ylabel("R²")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    r2_grafik = os.path.join(
        GRAFIK_KLASORU,
        "regresyon_r2.png"
    )

    plt.savefig(
        r2_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # TAHMİN - GERÇEK DEĞER GRAFİĞİ
    # ============================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        mlp_pred,
        alpha=0.7
    )

    min_deger = min(
        y_test.min(),
        mlp_pred.min()
    )

    max_deger = max(
        y_test.max(),
        mlp_pred.max()
    )

    plt.plot(
        [min_deger, max_deger],
        [min_deger, max_deger],
        linestyle="--"
    )

    plt.title(
        "MLP Gerçek Değer - Tahmin Değerleri"
    )

    plt.xlabel("Gerçek Fiyat")
    plt.ylabel("Tahmin Edilen Fiyat")

    plt.tight_layout()

    tahmin_grafik = os.path.join(
        GRAFIK_KLASORU,
        "mlp_gercek_tahmin.png"
    )

    plt.savefig(
        tahmin_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # SONUCU DOSYAYA KAYDET
    # ============================================================

    metin = "\n"
    metin += "=" * 70 + "\n"
    metin += "BÖLÜM 1 - REGRESYON ANALİZİ\n"
    metin += "=" * 70 + "\n"
    metin += f"Veri seti: proje_ev_fiyatlari.csv\n"
    metin += f"Toplam kayıt: {len(df)}\n"
    metin += f"Eğitim kayıt sayısı: {len(X_train)}\n"
    metin += f"Test kayıt sayısı: {len(X_test)}\n"
    metin += "Test oranı: %20\n"
    metin += "Standardizasyon: StandardScaler\n"
    metin += "\n"

    for _, satir in sonuc_df.iterrows():

        metin += (
            f"{satir['Model']}: "
            f"MSE = {satir['MSE']:.2f}, "
            f"R2 = {satir['R2']:.4f}\n"
        )

    metin += "\n"
    metin += (
        f"En düşük MSE değerine sahip model: "
        f"{en_iyi_mse['Model']} "
        f"({en_iyi_mse['MSE']:.2f})\n"
    )

    metin += (
        f"En yüksek R2 değerine sahip model: "
        f"{en_iyi_r2['Model']} "
        f"({en_iyi_r2['R2']:.4f})\n"
    )

    metin += "\n"
    metin += "Yorum:\n"
    metin += (
        "Regresyon modelleri ev fiyatlarını tahmin etmek amacıyla "
        "karşılaştırılmıştır. Özellikler standartlaştırıldıktan sonra "
        "Doğrusal Regresyon, Ridge, PCR, PLS ve MLP modelleri "
        "eğitilmiştir. MSE değerinin düşük olması daha başarılı bir "
        "tahmin performansını, R2 değerinin yüksek olması ise modelin "
        "hedef değişkendeki değişimi daha iyi açıklayabildiğini gösterir. "
        "MLP modeli doğrusal olmayan ilişkileri öğrenebildiği için "
        "özellikle konum ve diğer özellikler arasındaki karmaşık "
        "ilişkiler bulunduğunda doğrusal modellere göre daha başarılı "
        "olabilir.\n"
    )

    sonuc_kaydet(metin)

    print("\nBÖLÜM 1 TAMAMLANDI.")


# ================================================================
# 7. BÖLÜM 2
# KÜMELEME ANALİZİ
# ================================================================

def kumeleme_analizi():

    print("\n")
    print("=" * 70)
    print("BÖLÜM 2 - KÜMELEME ANALİZİ")
    print("=" * 70)

    # ------------------------------------------------------------
    # Dosya kontrolü
    # ------------------------------------------------------------

    if not dosya_kontrol(KUMELEME_DOSYASI):
        return

    print("\nVeri seti okunuyor...")

    # CSV virgül veya boşluk ayracına karşı tolerans
    try:

        df = pd.read_csv(
            KUMELEME_DOSYASI
        )

    except Exception:

        df = pd.read_csv(
            KUMELEME_DOSYASI,
            sep=r",|\s+",
            engine="python"
        )

    df = sutunlari_temizle(df)

    print("\nİlk 5 kayıt:")
    print(df.head())

    print("\nVeri boyutu:")
    print(df.shape)

    # ------------------------------------------------------------
    # Sütun kontrolü
    # ------------------------------------------------------------

    if (
        "yillik_gelir" not in df.columns
        or
        "harcama_puani" not in df.columns
    ):

        print("\nHATA!")
        print(
            "Beklenen sütunlar: "
            "yillik_gelir, harcama_puani"
        )

        print("Mevcut sütunlar:")
        print(list(df.columns))

        return

    # ------------------------------------------------------------
    # Sayısal dönüşüm
    # ------------------------------------------------------------

    df["yillik_gelir"] = pd.to_numeric(
        df["yillik_gelir"],
        errors="coerce"
    )

    df["harcama_puani"] = pd.to_numeric(
        df["harcama_puani"],
        errors="coerce"
    )

    df = df[
        [
            "yillik_gelir",
            "harcama_puani"
        ]
    ].dropna()

    # ------------------------------------------------------------
    # X
    # ------------------------------------------------------------

    X = df[
        [
            "yillik_gelir",
            "harcama_puani"
        ]
    ]

    # ------------------------------------------------------------
    # STANDARDİZASYON
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nÖzellikler standartlaştırıldı.")

    # ============================================================
    # ELBOW YÖNTEMİ
    # ============================================================

    print("\n")
    print("-" * 70)
    print("Elbow / Dirsek Yöntemi")
    print("-" * 70)

    k_degerleri = range(2, 11)

    inertia_degerleri = []

    for k in k_degerleri:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertia_degerleri.append(
            model.inertia_
        )

    # ------------------------------------------------------------
    # Elbow grafiği
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        list(k_degerleri),
        inertia_degerleri,
        marker="o"
    )

    plt.title(
        "K-Means Elbow / Dirsek Yöntemi"
    )

    plt.xlabel("Küme Sayısı (K)")
    plt.ylabel("Inertia")

    plt.xticks(
        list(k_degerleri)
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    elbow_grafik = os.path.join(
        GRAFIK_KLASORU,
        "kumeleme_elbow.png"
    )

    plt.savefig(
        elbow_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # K = 5
    # ============================================================
    # Ödev metninde varsayılan veri seti için K=5 belirtilmiş.
    # Bu nedenle nihai K-Means modeli 5 küme ile kuruluyor.
    # ============================================================

    optimal_k = 5

    print("\nÖdevde belirtilen uygun küme sayısı:")
    print("K =", optimal_k)

    # ============================================================
    # K-MEANS
    # ============================================================

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    etiketler = kmeans.fit_predict(
        X_scaled
    )

    df["Kume"] = etiketler

    # ============================================================
    # SILHOUETTE
    # ============================================================

    silhouette = silhouette_score(
        X_scaled,
        etiketler
    )

    print("\nSilhouette skoru:")
    print(round(silhouette, 4))

    # ============================================================
    # KÜME MERKEZLERİ
    # ============================================================

    merkezler_scaled = kmeans.cluster_centers_

    merkezler_original = scaler.inverse_transform(
        merkezler_scaled
    )

    merkez_df = pd.DataFrame(
        merkezler_original,
        columns=[
            "yillik_gelir",
            "harcama_puani"
        ]
    )

    merkez_df.index.name = "Kume"

    print("\nKüme merkezleri:")
    print(
        merkez_df.round(2)
    )

    # ============================================================
    # KÜMELERİ GELİR VE HARCAMA DURUMUNA GÖRE YORUMLAMA
    # ============================================================

    gelir_ort = df["yillik_gelir"].mean()
    harcama_ort = df["harcama_puani"].mean()

    print("\n")
    print("-" * 70)
    print("KÜME YORUMLARI")
    print("-" * 70)

    kume_yorumlari = {}

    for kume in range(optimal_k):

        merkez_gelir = merkez_df.loc[
            kume,
            "yillik_gelir"
        ]

        merkez_harcama = merkez_df.loc[
            kume,
            "harcama_puani"
        ]

        if (
            merkez_gelir < gelir_ort
            and
            merkez_harcama < harcama_ort
        ):

            yorum = "Düşük gelir - düşük harcama"

        elif (
            merkez_gelir < gelir_ort
            and
            merkez_harcama >= harcama_ort
        ):

            yorum = "Düşük gelir - yüksek harcama"

        elif (
            merkez_gelir >= gelir_ort
            and
            merkez_harcama < harcama_ort
        ):

            yorum = "Yüksek gelir - düşük harcama"

        else:

            yorum = "Yüksek gelir - yüksek harcama"

        kume_yorumlari[kume] = yorum

        print(
            f"Küme {kume}: "
            f"{yorum} | "
            f"Gelir={merkez_gelir:.2f} TL | "
            f"Harcama={merkez_harcama:.2f}"
        )

    # ============================================================
    # KÜME GRAFİĞİ
    # ============================================================

    plt.figure(
        figsize=(10, 7)
    )

    for kume in range(optimal_k):

        kume_veri = df[
            df["Kume"] == kume
        ]

        plt.scatter(
            kume_veri["yillik_gelir"],
            kume_veri["harcama_puani"],
            label=f"Küme {kume}",
            alpha=0.7
        )

    plt.scatter(
        merkez_df["yillik_gelir"],
        merkez_df["harcama_puani"],
        marker="X",
        s=250,
        label="Küme Merkezleri"
    )

    plt.title(
        "K-Means Müşteri Segmentasyonu"
    )

    plt.xlabel(
        "Yıllık Gelir (TL)"
    )

    plt.ylabel(
        "Harcama Puanı"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.2
    )

    plt.tight_layout()

    kume_grafik = os.path.join(
        GRAFIK_KLASORU,
        "kumeleme_sonuclari.png"
    )

    plt.savefig(
        kume_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # KÜME SAYILARI
    # ============================================================

    print("\nKüme büyüklükleri:")

    kume_sayilari = (
        df["Kume"]
        .value_counts()
        .sort_index()
    )

    print(kume_sayilari)

    # ============================================================
    # SONUÇLARI DOSYAYA KAYDET
    # ============================================================

    metin = "\n"
    metin += "=" * 70 + "\n"
    metin += "BÖLÜM 2 - KÜMELEME ANALİZİ\n"
    metin += "=" * 70 + "\n"
    metin += (
        "Veri seti: "
        "proje_musteri_segmentasyonu.csv\n"
    )

    metin += (
        f"Toplam müşteri sayısı: "
        f"{len(df)}\n"
    )

    metin += (
        "Kullanılan yöntem: "
        "K-Means\n"
    )

    metin += (
        "Standardizasyon: "
        "StandardScaler\n"
    )

    metin += (
        "Elbow için K aralığı: "
        "2-10\n"
    )

    metin += (
        f"Seçilen K: "
        f"{optimal_k}\n"
    )

    metin += (
        f"Silhouette skoru: "
        f"{silhouette:.4f}\n"
    )

    metin += "\nKüme merkezleri:\n"

    for kume in range(optimal_k):

        metin += (
            f"Küme {kume}: "
            f"Gelir={merkez_df.loc[kume, 'yillik_gelir']:.2f} TL, "
            f"Harcama={merkez_df.loc[kume, 'harcama_puani']:.2f} | "
            f"{kume_yorumlari[kume]}\n"
        )

    metin += "\nYorum:\n"

    metin += (
        "Kümeleme analizinde müşterilerin yıllık gelir ve "
        "harcama puanlarına göre benzer gruplara ayrılması "
        "amaçlanmıştır. Gelir ve harcama puanı farklı ölçeklerde "
        "olduğu için K-Means öncesinde StandardScaler ile "
        "standardizasyon yapılmıştır. Dirsek yöntemi kullanılarak "
        "küme sayısı incelenmiş ve proje veri seti için K=5 "
        "kullanılmıştır. Silhouette skorunun yüksek olması, "
        "oluşturulan kümelerin birbirinden iyi ayrıştığını gösterir.\n"
    )

    sonuc_kaydet(metin)

    print("\nBÖLÜM 2 TAMAMLANDI.")


# ================================================================
# 8. BÖLÜM 3
# METİN DUYGU ANALİZİ - SimpleRNN
# ================================================================

def imdb_sutunlarini_bul(df):

    """
    IMDB veri setindeki metin ve etiket sütunlarını bulur.
    """

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # Standart IMDB Dataset.csv:
    # review, sentiment

    metin_adaylari = [
        "review",
        "reviews",
        "text",
        "reviewsentiment"
    ]

    etiket_adaylari = [
        "sentiment",
        "label",
        "target"
    ]

    metin_sutunu = None
    etiket_sutunu = None

    for sutun in metin_adaylari:

        if sutun in df.columns:

            metin_sutunu = sutun
            break

    for sutun in etiket_adaylari:

        if sutun in df.columns:

            etiket_sutunu = sutun
            break

    # Eğer standart isimler bulunamadıysa,
    # object/string sütunlarından tahmin et.

    if metin_sutunu is None:

        object_sutunlari = df.select_dtypes(
            include=["object"]
        ).columns

        if len(object_sutunlari) >= 1:

            # En uzun ortalama metne sahip sütunu seç
            uzunluklar = {}

            for sutun in object_sutunlari:

                uzunluklar[sutun] = (
                    df[sutun]
                    .astype(str)
                    .str.len()
                    .mean()
                )

            metin_sutunu = max(
                uzunluklar,
                key=uzunluklar.get
            )

    if etiket_sutunu is None:

        object_sutunlari = df.select_dtypes(
            include=["object"]
        ).columns

        for sutun in object_sutunlari:

            if sutun != metin_sutunu:

                benzersiz = (
                    df[sutun]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .unique()
                )

                if set(benzersiz).issubset(
                    {
                        "positive",
                        "negative"
                    }
                ):

                    etiket_sutunu = sutun
                    break

    return metin_sutunu, etiket_sutunu


def metin_duygu_analizi():

    print("\n")
    print("=" * 70)
    print("BÖLÜM 3 - METİN DUYGU ANALİZİ")
    print("=" * 70)

    # ------------------------------------------------------------
    # Dosya kontrolü
    # ------------------------------------------------------------

    if not dosya_kontrol(IMDB_DOSYASI):
        return

    print("\nIMDB veri seti okunuyor...")

    df = pd.read_csv(
        IMDB_DOSYASI
    )

    print("\nBulunan sütunlar:")
    print(list(df.columns))

    # ------------------------------------------------------------
    # Sütunları bul
    # ------------------------------------------------------------

    metin_sutunu, etiket_sutunu = imdb_sutunlarini_bul(
        df
    )

    print("\nMetin sütunu:", metin_sutunu)
    print("Etiket sütunu:", etiket_sutunu)

    if metin_sutunu is None or etiket_sutunu is None:

        print("\nHATA!")
        print(
            "IMDB veri setinde metin veya sentiment sütunu bulunamadı."
        )

        print("\nMevcut sütunlar:")
        print(list(df.columns))

        return

    # ------------------------------------------------------------
    # Veriyi temizle
    # ------------------------------------------------------------

    df = df[
        [
            metin_sutunu,
            etiket_sutunu
        ]
    ].dropna()

    df[metin_sutunu] = (
        df[metin_sutunu]
        .astype(str)
    )

    df[etiket_sutunu] = (
        df[etiket_sutunu]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # ------------------------------------------------------------
    # positive / negative kontrolü
    # ------------------------------------------------------------

    df = df[
        df[etiket_sutunu].isin(
            [
                "positive",
                "negative"
            ]
        )
    ]

    if len(df) == 0:

        print("\nHATA!")
        print(
            "positive / negative etiketleri bulunamadı."
        )

        return

    # ------------------------------------------------------------
    # Etiketleri 0 ve 1'e dönüştür
    # ------------------------------------------------------------

    df["etiket"] = (
        df[etiket_sutunu]
        .map(
            {
                "negative": 0,
                "positive": 1
            }
        )
    )

    metinler = df[
        metin_sutunu
    ].values

    etiketler = df[
        "etiket"
    ].values

    print("\nToplam yorum:", len(metinler))

    print("\nSınıf dağılımı:")

    print(
        df["etiket"]
        .value_counts()
        .sort_index()
    )

    # ============================================================
    # HTML ETİKETLERİNİ TEMİZLE
    # ============================================================

    temiz_metinler = []

    for metin in metinler:

        metin = re.sub(
            r"<br\s*/?>",
            " ",
            metin
        )

        metin = re.sub(
            r"<[^>]+>",
            " ",
            metin
        )

        metin = re.sub(
            r"\s+",
            " ",
            metin
        )

        metin = metin.strip()

        temiz_metinler.append(
            metin
        )

    # ============================================================
    # TRAIN / TEST
    # ============================================================

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        temiz_metinler,
        etiketler,
        test_size=0.20,
        random_state=42,
        stratify=etiketler
    )

    print("\nEğitim yorum sayısı:", len(X_train_text))
    print("Test yorum sayısı:", len(X_test_text))

    # ============================================================
    # TOKENIZER
    # ============================================================

    MAX_WORDS = 20000
    MAX_LEN = 200

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(
        X_train_text
    )

    # ------------------------------------------------------------
    # Metinleri sayısal dizilere çevir
    # ------------------------------------------------------------

    X_train_seq = tokenizer.texts_to_sequences(
        X_train_text
    )

    X_test_seq = tokenizer.texts_to_sequences(
        X_test_text
    )

    # ============================================================
    # PAD SEQUENCES
    # ============================================================

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    print("\nTokenizer kelime sayısı:")
    print(
        min(
            len(tokenizer.word_index) + 1,
            MAX_WORDS
        )
    )

    print("Maksimum dizi uzunluğu:", MAX_LEN)

    # ============================================================
    # SimpleRNN MODELİ
    # Embedding -> SimpleRNN -> Dense(sigmoid)
    # ============================================================

    print("\nSimpleRNN modeli oluşturuluyor...")

    model = Sequential(
        [
            Embedding(
                input_dim=MAX_WORDS,
                output_dim=64,
                input_length=MAX_LEN
            ),

            SimpleRNN(
                64
            ),

            Dense(
                1,
                activation="sigmoid"
            )
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    print("\nModel özeti:")
    model.summary()

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # ============================================================
    # EĞİTİM
    # ============================================================

    print("\nModel eğitiliyor...")
    print(
        "Not: IMDB veri setinin tamamı ile eğitim birkaç dakika sürebilir."
    )

    history = model.fit(
        X_train_pad,
        np.array(y_train),
        validation_split=0.10,
        epochs=8,
        batch_size=128,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    # ============================================================
    # TEST
    # ============================================================

    test_loss, test_accuracy = model.evaluate(
        X_test_pad,
        np.array(y_test),
        verbose=0
    )

    print("\n")
    print("=" * 70)
    print("SimpleRNN TEST SONUCU")
    print("=" * 70)

    print(
        "Test Loss:",
        round(test_loss, 4)
    )

    print(
        "Test Accuracy:",
        round(test_accuracy, 4)
    )

    # ============================================================
    # EĞİTİM GRAFİĞİ - ACCURACY
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history.history["accuracy"],
        label="Eğitim"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Doğrulama"
    )

    plt.title(
        "SimpleRNN Eğitim ve Doğrulama Başarımı"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    rnn_accuracy_grafik = os.path.join(
        GRAFIK_KLASORU,
        "rnn_accuracy.png"
    )

    plt.savefig(
        rnn_accuracy_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # EĞİTİM GRAFİĞİ - LOSS
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history.history["loss"],
        label="Eğitim"
    )

    plt.plot(
        history.history["val_loss"],
        label="Doğrulama"
    )

    plt.title(
        "SimpleRNN Eğitim ve Doğrulama Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    rnn_loss_grafik = os.path.join(
        GRAFIK_KLASORU,
        "rnn_loss.png"
    )

    plt.savefig(
        rnn_loss_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # ÖRNEK TAHMİNLER
    # ============================================================

    print("\nÖrnek tahminler:")

    ornek_sayisi = min(
        10,
        len(X_test_text)
    )

    tahminler = model.predict(
        X_test_pad[:ornek_sayisi],
        verbose=0
    )

    for i in range(ornek_sayisi):

        gercek = (
            "positive"
            if y_test[i] == 1
            else "negative"
        )

        tahmin = (
            "positive"
            if tahminler[i][0] >= 0.5
            else "negative"
        )

        print(
            f"{i+1}. "
            f"Gerçek={gercek} | "
            f"Tahmin={tahmin} | "
            f"Olasılık={tahminler[i][0]:.4f}"
        )

    # ============================================================
    # SONUCU DOSYAYA KAYDET
    # ============================================================

    metin = "\n"
    metin += "=" * 70 + "\n"
    metin += "BÖLÜM 3 - METİN DUYGU ANALİZİ\n"
    metin += "=" * 70 + "\n"
    metin += "Veri seti: IMDB Dataset.csv\n"
    metin += f"Toplam yorum: {len(df)}\n"
    metin += f"Eğitim yorum sayısı: {len(X_train_text)}\n"
    metin += f"Test yorum sayısı: {len(X_test_text)}\n"
    metin += "Test oranı: %20\n"
    metin += "Model: Embedding -> SimpleRNN -> Dense\n"
    metin += "Embedding boyutu: 64\n"
    metin += "SimpleRNN hücre sayısı: 64\n"
    metin += "Dense aktivasyonu: sigmoid\n"
    metin += f"Maksimum kelime sayısı: {MAX_WORDS}\n"
    metin += f"Maksimum dizi uzunluğu: {MAX_LEN}\n"
    metin += f"Test Loss: {test_loss:.4f}\n"
    metin += f"Test Accuracy: {test_accuracy:.4f}\n"

    metin += "\nYorum:\n"

    metin += (
        "IMDB film yorumlarının olumlu veya olumsuz olduğunu "
        "tahmin etmek amacıyla SimpleRNN tabanlı bir duygu analizi "
        "modeli kullanılmıştır. Metinler önce Tokenizer kullanılarak "
        "sayısal dizilere dönüştürülmüş, ardından pad_sequences ile "
        "aynı uzunluğa getirilmiştir. Model Embedding, SimpleRNN ve "
        "sigmoid aktivasyonlu Dense katmanından oluşturulmuştur. "
        "Test doğruluğu modelin daha önce görmediği yorumlar üzerindeki "
        "sınıflandırma başarısını göstermektedir.\n"
    )

    sonuc_kaydet(metin)

    print("\nBÖLÜM 3 TAMAMLANDI.")


# ================================================================
# 9. BÖLÜM 4
# GÖRÜNTÜ SINIFLANDIRMA - CNN
# ================================================================

def goruntu_siniflandirma():

    print("\n")
    print("=" * 70)
    print("BÖLÜM 4 - GÖRÜNTÜ SINIFLANDIRMA")
    print("=" * 70)

    print("\nFashion-MNIST veri seti yükleniyor...")

    # ============================================================
    # FASHION-MNIST
    # ============================================================

    (
        X_train,
        y_train
    ), (
        X_test,
        y_test
    ) = fashion_mnist.load_data()

    print("\nEğitim görüntü sayısı:", len(X_train))
    print("Test görüntü sayısı:", len(X_test))

    print(
        "Görüntü boyutu:",
        X_train.shape[1:]
    )

    print("Sınıf sayısı:", len(np.unique(y_train)))

    # ============================================================
    # NORMALİZASYON
    # ============================================================

    X_train = X_train.astype(
        "float32"
    ) / 255.0

    X_test = X_test.astype(
        "float32"
    ) / 255.0

    # ============================================================
    # CNN için kanal boyutu
    # ============================================================

    X_train = np.expand_dims(
        X_train,
        axis=-1
    )

    X_test = np.expand_dims(
        X_test,
        axis=-1
    )

    print(
        "\nCNN giriş boyutu:",
        X_train.shape
    )

    # ============================================================
    # SINIF İSİMLERİ
    # ============================================================

    siniflar = [
        "Tişört/Üst",
        "Pantolon",
        "Kazak",
        "Elbise",
        "Ceket",
        "Sandalet",
        "Gömlek",
        "Spor Ayakkabı",
        "Çanta",
        "Bilek Bot"
    ]

    # ============================================================
    # CNN MODELİ
    # ============================================================

    print("\nCNN modeli oluşturuluyor...")

    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                input_shape=(28, 28, 1)
            ),

            MaxPooling2D(
                (2, 2)
            ),

            Conv2D(
                64,
                (3, 3),
                activation="relu"
            ),

            MaxPooling2D(
                (2, 2)
            ),

            Flatten(),

            Dense(
                128,
                activation="relu"
            ),

            Dropout(
                0.3
            ),

            Dense(
                10,
                activation="softmax"
            )
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    print("\nCNN model özeti:")
    model.summary()

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # ============================================================
    # EĞİTİM
    # ============================================================

    print("\nCNN eğitiliyor...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.10,
        epochs=10,
        batch_size=128,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    # ============================================================
    # TEST
    # ============================================================

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\n")
    print("=" * 70)
    print("CNN TEST SONUCU")
    print("=" * 70)

    print(
        "Test Loss:",
        round(test_loss, 4)
    )

    print(
        "Test Accuracy:",
        round(test_accuracy, 4)
    )

    # ============================================================
    # ÖRNEK GÖRÜNTÜLER
    # ============================================================

    plt.figure(
        figsize=(12, 8)
    )

    for i in range(12):

        plt.subplot(
            3,
            4,
            i + 1
        )

        plt.imshow(
            X_test[i].squeeze(),
            cmap="gray"
        )

        plt.title(
            siniflar[y_test[i]]
        )

        plt.axis("off")

    plt.tight_layout()

    fashion_grafik = os.path.join(
        GRAFIK_KLASORU,
        "fashion_mnist_ornekleri.png"
    )

    plt.savefig(
        fashion_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # ACCURACY GRAFİĞİ
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history.history["accuracy"],
        label="Eğitim"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Doğrulama"
    )

    plt.title(
        "CNN Eğitim ve Doğrulama Başarımı"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    cnn_accuracy_grafik = os.path.join(
        GRAFIK_KLASORU,
        "cnn_accuracy.png"
    )

    plt.savefig(
        cnn_accuracy_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # LOSS GRAFİĞİ
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history.history["loss"],
        label="Eğitim"
    )

    plt.plot(
        history.history["val_loss"],
        label="Doğrulama"
    )

    plt.title(
        "CNN Eğitim ve Doğrulama Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    cnn_loss_grafik = os.path.join(
        GRAFIK_KLASORU,
        "cnn_loss.png"
    )

    plt.savefig(
        cnn_loss_grafik,
        dpi=150
    )

    plt.show()

    # ============================================================
    # ÖRNEK TAHMİNLER
    # ============================================================

    print("\nÖrnek görüntü tahminleri:")

    tahminler = model.predict(
        X_test[:10],
        verbose=0
    )

    for i in range(10):

        tahmin_sinifi = np.argmax(
            tahminler[i]
        )

        gercek_sinif = y_test[i]

        print(
            f"{i+1}. "
            f"Gerçek={siniflar[gercek_sinif]} | "
            f"Tahmin={siniflar[tahmin_sinifi]} | "
            f"Olasılık={np.max(tahminler[i]):.4f}"
        )

    # ============================================================
    # SONUCU DOSYAYA KAYDET
    # ============================================================

    metin = "\n"
    metin += "=" * 70 + "\n"
    metin += "BÖLÜM 4 - GÖRÜNTÜ SINIFLANDIRMA\n"
    metin += "=" * 70 + "\n"
    metin += "Veri seti: Fashion-MNIST\n"
    metin += f"Eğitim görüntüsü: {len(X_train)}\n"
    metin += f"Test görüntüsü: {len(X_test)}\n"
    metin += "Görüntü boyutu: 28x28\n"
    metin += "Renk: Gri tonlamalı\n"
    metin += "Sınıf sayısı: 10\n"
    metin += "Normalizasyon: Piksel değerleri / 255\n"
    metin += "Model: CNN\n"
    metin += "Conv2D + MaxPooling2D blokları kullanıldı.\n"
    metin += "Dense katman: 128 nöron, ReLU\n"
    metin += "Dropout: 0.30\n"
    metin += "Çıkış: 10 sınıf, softmax\n"
    metin += f"Test Loss: {test_loss:.4f}\n"
    metin += f"Test Accuracy: {test_accuracy:.4f}\n"

    metin += "\nYorum:\n"

    metin += (
        "Görüntü sınıflandırma bölümünde Keras içerisinde hazır "
        "olarak bulunan Fashion-MNIST veri seti kullanılmıştır. "
        "Görüntüler 28x28 boyutunda gri tonlamalı kıyafet "
        "görüntülerinden oluşmaktadır. Piksel değerleri 0-255 "
        "aralığından 0-1 aralığına normalize edilmiştir. "
        "CNN modeli Conv2D ve MaxPooling katmanlarından oluşan "
        "evrişimsel bloklar kullanılarak oluşturulmuştur. "
        "Sonuç katmanında softmax kullanılarak 10 farklı kıyafet "
        "sınıfı arasından tahmin yapılmıştır. Test accuracy "
        "değeri modelin daha önce görmediği görüntüler üzerindeki "
        "sınıflandırma başarısını göstermektedir.\n"
    )

    sonuc_kaydet(metin)

    print("\nBÖLÜM 4 TAMAMLANDI.")


# ================================================================
# 10. MENÜ
# ================================================================

def menu():

    while True:

        print("\n")
        print("=" * 70)
        print("      BÜTÜNLEŞİK MAKİNE ÖĞRENMESİ UYGULAMASI")
        print("=" * 70)

        print("\n1 - Regresyon Analizi")
        print("2 - Kümeleme Analizi")
        print("3 - Metin Duygu Analizi")
        print("4 - Görüntü Sınıflandırma")
        print("5 - Çıkış")

        print("=" * 70)

        secim = input(
            "\nYapmak istediğiniz analizi seçiniz (1-5): "
        ).strip()

        # --------------------------------------------------------
        # 1
        # --------------------------------------------------------

        if secim == "1":

            regresyon_analizi()

        # --------------------------------------------------------
        # 2
        # --------------------------------------------------------

        elif secim == "2":

            kumeleme_analizi()

        # --------------------------------------------------------
        # 3
        # --------------------------------------------------------

        elif secim == "3":

            metin_duygu_analizi()

        # --------------------------------------------------------
        # 4
        # --------------------------------------------------------

        elif secim == "4":

            goruntu_siniflandirma()

        # --------------------------------------------------------
        # 5
        # --------------------------------------------------------

        elif secim == "5":

            print("\n")
            print("=" * 70)
            print("Program sonlandırılıyor...")
            print("=" * 70)

            print(
                "\nSonuç dosyası:"
            )

            print(
                SONUC_DOSYASI
            )

            print(
                "\nProgram başarıyla sonlandırıldı."
            )

            break

        # --------------------------------------------------------
        # Hatalı seçim
        # --------------------------------------------------------

        else:

            print(
                "\nGeçersiz seçim!"
            )

            print(
                "Lütfen 1, 2, 3, 4 veya 5 giriniz."
            )


# ================================================================
# 11. PROGRAMI BAŞLAT
# ================================================================

print("\n")
print("=" * 70)
print("BÜTÜNLEŞİK MAKİNE ÖĞRENMESİ UYGULAMASI")
print("=" * 70)

print("\nDosya yolları kontrol ediliyor...")

print(
    "\nRegresyon:",
    REGRESYON_DOSYASI
)

print(
    "Kümeleme:",
    KUMELEME_DOSYASI
)

print(
    "IMDB:",
    IMDB_DOSYASI
)

print(
    "Sonuç:",
    SONUC_DOSYASI
)

print(
    "\nProgram hazır."
)

print(
    "Menüden yapmak istediğiniz analizi seçebilirsiniz."
)

# Menü başlat
menu()
