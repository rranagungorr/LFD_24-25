import pandas as pd


def clean_and_impute(df, threshold=0.5):
    """
    Eksik değerleri temizler ve ortalama ile doldurur.
    :param df: DataFrame
    :param threshold: Eksik oranı için sınır (varsayılan 0.5)
    :return: Temizlenmiş DataFrame
    """
    # Eksik oranı threshold'dan yüksek olan sütunları sil
    df = df.loc[:, df.isnull().mean() < threshold]

    # Kalan eksik değerleri sütun ortalaması ile doldur
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df.loc[:, column] = df[column].fillna(df[column].mean())

    return df


def merge_weather_data():
    """
    Üç şehrin verilerini birleştirir, başlıklarını günceller ve tek bir dosyada kaydeder.
    """
    # Dosyaları yükle
    antalya_df = pd.read_csv(r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\Antalya_hourly_weather.csv')
    istanbul_df = pd.read_csv(r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\Istanbul_hourly_weather.csv')
    izmir_df = pd.read_csv(r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\Izmir_hourly_weather.csv')

    # 'coco' sütununu kaldır
    columns_to_remove = ['coco']
    antalya_df = antalya_df.drop(columns=columns_to_remove, errors='ignore')
    istanbul_df = istanbul_df.drop(columns=columns_to_remove, errors='ignore')

    # Temizleme işlemleri
    antalya_cleaned = clean_and_impute(antalya_df)
    istanbul_cleaned = clean_and_impute(istanbul_df)
    izmir_cleaned = clean_and_impute(izmir_df)

    # Sütun isimlerini şehir isimlerine göre güncelle
    antalya_cleaned.columns = [f"{col}_antalya" for col in antalya_cleaned.columns]
    istanbul_cleaned.columns = [f"{col}_istanbul" for col in istanbul_cleaned.columns]
    izmir_cleaned.columns = [f"{col}_izmir" for col in izmir_cleaned.columns]

    # Verileri birleştir
    merged_df = pd.concat([istanbul_cleaned, antalya_cleaned, izmir_cleaned], axis=1)

    # Birleştirilmiş veriyi kaydet
    merged_path = r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\Merged_weather_data.csv'
    merged_df.to_csv(merged_path, index=False)

    print(f"Birleştirilmiş veri '{merged_path}' konumuna kaydedildi.")
    return merged_df


def integrate_with_smfdb(smfdb_path, merged_weather_data):
    """
    Merged weather data ile SMFDB verilerini entegre eder.
    :param smfdb_path: SMFDB dosya yolu
    :param merged_weather_data: Birleştirilmiş hava durumu verisi
    """
    # SMFDB verisini yükle
    smfdb_data = pd.read_csv(smfdb_path)

    # Verileri birleştir
    integrated_data = pd.concat([smfdb_data, merged_weather_data], axis=1)

    # Birleştirilmiş veriyi kaydet
    integrated_path = r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\smfdb_weather_data.csv'
    integrated_data.to_csv(integrated_path, index=False)

    print(f"SMFDB ve hava durumu verisi '{integrated_path}' konumuna kaydedildi.")
    return integrated_data


if __name__ == "__main__":
    # Hava durumu verilerini birleştir
    merged_weather = merge_weather_data()

    # SMFDB ile entegre et
    smfdb_file_path = r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\smfdb.csv'
    integrate_with_smfdb(smfdb_file_path, merged_weather)
