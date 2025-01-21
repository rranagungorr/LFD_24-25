import pandas as pd
import numpy as np
from scipy.stats import zscore

def check_missing_values(df):
    """
    Verideki eksik (NaN) değerleri kontrol eder.
    """
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print("Eksik değer bulunan sütunlar:")
        print(missing_values[missing_values > 0])
        print(f"Toplam eksik değer sayısı: {total_missing}")
    else:
        print("Veride eksik değer bulunmamaktadır.")

def check_infinity_values(df):
    """
    Infinity ve -Infinity değerleri kontrol eder.
    """
    inf_count = (df == float('inf')).sum().sum()
    neg_inf_count = (df == -float('inf')).sum().sum()
    if inf_count > 0 or neg_inf_count > 0:
        print(f"Infinity değer sayısı: {inf_count}")
        print(f"Negative Infinity değer sayısı: {neg_inf_count}")
    else:
        print("Veride Infinity veya Negative Infinity değer bulunmamaktadır.")

def handle_infinity_and_nan_with_previous(df, columns):
    """
    NaN, None, Infinity ve -Infinity değerlerini bir önceki geçerli değerle doldurur.
    Eğer bir önceki değer de geçerli değilse, daha önceki değerleri kontrol eder.
    """
    for col in columns:
        for i in range(len(df)):
            if pd.isnull(df.loc[i, col]) or df.loc[i, col] in [float('inf'), -float('inf')]:
                j = i - 1  # Bir önceki satıra bak
                while j >= 0:
                    if not pd.isnull(df.loc[j, col]) and df.loc[j, col] not in [float('inf'), -float('inf')]:
                        df.loc[i, col] = df.loc[j, col]  # Geçerli değeri doldur
                        break
                    j -= 1  # Daha önceki satıra geç
    return df

def check_outliers(df, method="zscore", threshold=3):
    """
    Uç değerleri kontrol eder. Z-Score veya IQR yöntemini kullanır.
    """
    if method == "zscore":
        z_scores = df.apply(zscore)
        outliers = (z_scores.abs() > threshold).sum()
        print("Uç değer sayısı (Z-Score):")
        print(outliers[outliers > 0])
    elif method == "iqr":
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"{col} sütunundaki uç değer sayısı (IQR): {len(outliers)}")

def preprocess_smfdb(file_path, output_path):
    """
    SMFDB verisini temizler ve işlenmiş veriyi kaydeder.
    """
    # Veri yükleme
    data = pd.read_csv(file_path)

    # Kontroller
    print("--- Eksik Değer Kontrolü ---")
    check_missing_values(data)

    print("\n--- Infinity Değer Kontrolü ---")
    check_infinity_values(data)

    # Dolar ve Euro sütunlarındaki infinity ve NaN değerlerini önceki değerle doldurma
    columns_to_fix = ['Dolar', 'Euro']
    data = handle_infinity_and_nan_with_previous(data, columns_to_fix)

    # Uç değer kontrolü (Z-Score)
    print("\n--- Uç Değer Kontrolü (Z-Score) ---")
    check_outliers(data.select_dtypes(include=[np.number]), method="zscore")

    # Temizlenmiş veriyi kaydet
    data.to_csv(output_path, index=False)
    print(f"Temizlenmiş veri başarıyla kaydedildi: {output_path}")

# Kullanım
if __name__ == "__main__":
    input_file = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\raw\smfdb.csv"
    output_file = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\cleaned_smfdb_fixedd.csv"

    preprocess_smfdb(input_file, output_file)
