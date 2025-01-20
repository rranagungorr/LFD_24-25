import pandas as pd
import numpy as np
from scipy.stats import zscore

# SMFDB dosyasını yükleme
file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\smfdb.csv"
data = pd.read_csv(file_path)

# Sayısal sütunları seç
numeric_cols = data.select_dtypes(include=['float64', 'int64'])

# Eksik (NaN) değerleri kontrol et
def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Eksik değer sayısı:")
    print(missing_values[missing_values > 0])
    print("\nToplam eksik değer sayısı:", missing_values.sum())

# Sonsuz değerleri kontrol et
def check_infinity_values(df):
    infinity_values = df[(df == float('inf')) | (df == -float('inf'))]
    print("Sonsuz değer içeren satır sayısı:", len(infinity_values))

# Negatif değerleri kontrol et
def check_negative_values(df):
    for col in df.columns:
        negative_values = df[df[col] < 0]
        if len(negative_values) > 0:
            print(f"{col} sütunundaki negatif değer sayısı: {len(negative_values)}")

# Uç değerleri Z-skoru ile kontrol et
def check_outliers_zscore(df):
    z_scores = df.apply(zscore)
    outliers = (z_scores.abs() > 3).sum()
    print("Uç değer sayısı (Z-skoru > 3):")
    print(outliers[outliers > 0])

# Uç değerleri IQR ile kontrol et
def check_outliers_iqr(df):
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"{col} sütunundaki uç değer sayısı (IQR yöntemi): {len(outliers)}")


def check_infinity(dataframe):
    """
    Bu fonksiyon, bir DataFrame'deki infinity ve -infinity değerlerini kontrol eder.
    Hangi sütunlarda bu değerlerin olduğunu ve sayısını döndürür.
    """
    inf_columns = []
    results = []

    for col in dataframe.columns:
        inf_count = (dataframe[col] == float('inf')).sum()
        neg_inf_count = (dataframe[col] == -float('inf')).sum()

        if inf_count > 0 or neg_inf_count > 0:
            inf_columns.append(col)
            results.append({
                "Column": col,
                "Infinity Count": inf_count,
                "Negative Infinity Count": neg_inf_count
            })

    # Sonuçları DataFrame olarak döndür
    result_df = pd.DataFrame(results)
    return inf_columns, result_df


# Fonksiyonun Kullanımı:
# DataFrame'inizi yükleyin
file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\smfdb.csv"
data = pd.read_csv(file_path)

# Dolar ve Euro sütunlarını float türüne dönüştür
data['Dolar'] = pd.to_numeric(data['Dolar'], errors='coerce')
data['Euro'] = pd.to_numeric(data['Euro'], errors='coerce')

# Infinity değerlerin olup olmadığını tekrar kontrol et
inf_rows = data[(data['Dolar'] == float('inf')) | (data['Euro'] == float('inf'))]
print("Infinity değer içeren satırlar (dönüşüm sonrası):")
print(inf_rows)


# Infinity kontrolünü çalıştır
infinity_columns, infinity_summary = check_infinity(data)

# Çıktılar
print("Infinity değer içeren sütunlar:", infinity_columns)
print("\nInfinity ve -Infinity değerlerin özeti:")
print(infinity_summary)


# Fonksiyonları çalıştır
print("--- Eksik Değer Kontrolü ---")
check_missing_values(data)

print("\n--- Sonsuz Değer Kontrolü ---")
check_infinity_values(data)

print("\n--- Negatif Değer Kontrolü ---")
check_negative_values(numeric_cols)

print("\n--- Uç Değer Kontrolü (Z-Skoru) ---")
check_outliers_zscore(numeric_cols)

print("\n--- Uç Değer Kontrolü (IQR) ---")
check_outliers_iqr(numeric_cols)


def handle_infinity_values(dataframe, columns):
    """
    Bu fonksiyon, belirtilen sütunlarda infinity değerleri tespit eder,
    infinity değerleri önce NaN yapar, ardından bu değerleri tekrar yerine koyar.

    Args:
    dataframe (pd.DataFrame): İşlenecek veri seti
    columns (list): Infinity kontrolü yapılacak sütunların isimleri

    Returns:
    pd.DataFrame: Infinity değerlerin temizlendiği veri seti
    """
    # Infinity değerleri geçici bir değişkene kaydet
    saved_values = {}
    for col in columns:
        saved_values[col] = dataframe.loc[dataframe[col] == float('inf'), col].copy()

    # Infinity değerlerini NaN ile değiştir
    for col in columns:
        dataframe[col] = dataframe[col].replace(float('inf'), None)

    # Tutulan değerleri geri yerine koy
    for col in columns:
        for index in saved_values[col].index:
            dataframe.at[index, col] = saved_values[col][index]

    return dataframe


# Veri setini yükleyin
file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\smfdb.csv"
data = pd.read_csv(file_path)

# Fonksiyonu çağır
columns_to_fix = ['Dolar', 'Euro']
data = handle_infinity_values(data, columns_to_fix)

# Sonuçları kontrol et
print(data[(data['Dolar'] == float('inf')) | (data['Euro'] == float('inf'))])

# Temizlenmiş veriyi kaydedin
output_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\cleaned_smfdb.csv"
data.to_csv(output_path, index=False)
print(r"Temizlenmiş veri başarıyla kaydedildi: C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\cleaned_smfdb_fixed.csv")


# Temizlenmiş dosya yolunu girin
cleaned_file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\cleaned_smfdb.csv"

# Dosyayı yükle
cleaned_data = pd.read_csv(cleaned_file_path)


import pandas as pd

# SMFDB dosyasını yükleme
file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\smfdb.csv"
data = pd.read_csv(file_path)

# Infinity değerlerin bir önceki satırın değeriyle doldurulması
def fix_infinity_values(dataframe, columns):
    """
    Infinity değerleri bir önceki satırdaki uygun değerle doldurur.
    Eğer bir önceki satırda da infinity varsa, daha önceki satırları kontrol eder.

    Args:
    - dataframe: Pandas DataFrame
    - columns: Infinity kontrolü yapılacak sütunlar

    Returns:
    - DataFrame: Düzeltmeler uygulanmış DataFrame
    """
    for col in columns:
        for i in range(len(dataframe)):
            if dataframe.loc[i, col] == float('inf'):  # Infinity değeri varsa
                j = i - 1  # Bir önceki satırı kontrol et
                while j >= 0:  # Satırın başına gelene kadar devam et
                    if dataframe.loc[j, col] != float('inf'):  # Infinity değilse
                        dataframe.loc[i, col] = dataframe.loc[j, col]  # Değeri al
                        break
                    j -= 1  # Bir öncekine geç

    return dataframe


# Fonksiyonu çağır ve düzeltmeleri uygula
columns_to_fix = ['Dolar', 'Euro']
data = fix_infinity_values(data, columns_to_fix)

# Sonuçları kontrol et
print("Infinity değerlerden sonra düzeltme yapılmış satırlar:")
print(data[(data['Dolar'] == float('inf')) | (data['Euro'] == float('inf'))])

# Düzeltmeleri içeren dosyayı kaydet
output_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\cleaned_smfdb_fixed.csv"
data.to_csv(output_path, index=False)
print(f"Temizlenmiş ve düzeltmeler yapılmış veri başarıyla kaydedildi: {output_path}")

import pandas as pd

# Temizlenmiş dosyanın yolunu girin
cleaned_file_path_fixed = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\cleaned_smfdb_fixed.csv"
cleaned_data_fixed = pd.read_csv(cleaned_file_path_fixed)

# Infinity kontrol fonksiyonu
def check_infinity_in_cleaned_data(dataframe):
    """
    Check for infinity and -infinity values in the cleaned DataFrame.
    Returns a summary of their occurrences.
    """
    inf_columns = dataframe.columns[(dataframe == float('inf')).any(axis=0)].tolist()
    neg_inf_columns = dataframe.columns[(dataframe == -float('inf')).any(axis=0)].tolist()

    results = []
    for col in dataframe.columns:
        inf_count = (dataframe[col] == float('inf')).sum()
        neg_inf_count = (dataframe[col] == -float('inf')).sum()
        if inf_count > 0 or neg_inf_count > 0:
            results.append({
                "Column": col,
                "Infinity Count": inf_count,
                "Negative Infinity Count": neg_inf_count
            })

    result_df = pd.DataFrame(results)
    return inf_columns, neg_inf_columns, result_df

# Kontrolü çalıştır
infinity_columns_cleaned, negative_infinity_columns_cleaned, infinity_summary_cleaned = check_infinity_in_cleaned_data(cleaned_data_fixed)

# Sonuçları yazdır
print("Infinity değer içeren sütunlar:", infinity_columns_cleaned)
print("Negative Infinity değer içeren sütunlar:", negative_infinity_columns_cleaned)
print("\nInfinity ve -Infinity değerlerin özeti:")
print(infinity_summary_cleaned)
