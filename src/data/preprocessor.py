import pandas as pd
import numpy as np

def feature_engineering(data):
    """
    SMF ve hava durumu verileri üzerinde özellik mühendisliği uygular.

    Args:
        data (pd.DataFrame): Girdi veri seti.

    Returns:
        pd.DataFrame: İşlenmiş ve yeni özellikler eklenmiş veri seti.
    """
    # 1. Zaman Tabanlı Özellikler
    data['Tarih'] = pd.to_datetime(data['Tarih'], dayfirst=True, errors='coerce')
    data['month'] = data['Tarih'].dt.month
    data['day_of_week'] = data['Tarih'].dt.dayofweek
    data['hour'] = data['Tarih'].dt.hour

    # 2. Talep ve Arz Farkı
    data['demand_supply_diff'] = data['Talepislemhacmi'] - data['Arzislemhacmi']

    # 3. Log Dönüşümü
    data['log_Smf'] = np.log1p(data['Smf'])
    data['log_Ptf'] = np.log1p(data['Ptf'])

    # 4. Yenilenebilir Enerji Yüzdesi
    data['renewable_percentage'] = (
        (data['Gerceklesenruzgar'] + data['Gerceklesengunes']) / data['Gerceklesentoplam']
    ) * 100

    # 5. Hareketli Ortalama (Rolling Mean)
    data['Smf_rolling_mean'] = data['Smf'].rolling(window=3, min_periods=1).mean()

    # 6. Hava Durumu Özelliklerinden Bağıl Farklar
    data['temp_diff_istanbul_izmir'] = data['temp_istanbul'] - data['temp_izmir']
    data['temp_diff_istanbul_antalya'] = data['temp_istanbul'] - data['temp_antalya']
    data['humidity_diff'] = data['rhum_istanbul'] - data['rhum_izmir']

    # 7. Rüzgar Hızı Ağırlıklandırma
    data['weighted_wind_speed'] = (
        (data['wspd_istanbul'] + data['wspd_antalya'] + data['wspd_izmir']) / 3
    )

    return data



def process_and_save_features(input_path, output_path):
    """
    Veri setini işler, özellik mühendisliği uygular ve kaydeder.

    Args:
        input_path (str): Girdi veri seti dosya yolu.
        output_path (str): İşlenmiş veri setinin kaydedileceği dosya yolu.
    """
    # Veri setini yükle
    data = pd.read_csv(input_path)

    # Özellik mühendisliği uygula
    processed_data = feature_engineering(data)

    # İşlenmiş veriyi kaydet
    processed_data.to_csv(output_path, index=False)
    print(f"İşlenmiş veri başarıyla kaydedildi: {output_path}")


def merge_and_save_datasets(weather_data_path, smfdb_data_path, output_path):
    """
    Hava durumu verisi ve SMFDB verisini birleştirir ve kaydeder.

    Args:
        weather_data_path (str): Hava durumu verisinin dosya yolu.
        smfdb_data_path (str): SMFDB verisinin dosya yolu.
        output_path (str): Birleştirilmiş verinin kaydedileceği dosya yolu.

    Returns:
        None
    """
    # Veri dosyalarını yükle
    weather_data = pd.read_csv(weather_data_path)
    smfdb_data = pd.read_csv(smfdb_data_path)

    # Verileri sütun bazında birleştir (axis=1)
    merged_data = pd.concat([smfdb_data, weather_data], axis=1)

    # Birleştirilmiş veriyi kaydet
    merged_data.to_csv(output_path, index=False)
    print(f"Birleştirilmiş veri '{output_path}' konumuna başarıyla kaydedildi.")


if __name__ == "__main__":
    # Birleştirme için dosya yolları

    input_file = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\smfdb_weather_dataa.csv"

    # Özellik mühendisliği için dosya yolları
    engineered_output_file = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\new_smf_weather_data.csv"



    # Birleştirilmiş veriye özellik mühendisliği uygula
    process_and_save_features(input_file, engineered_output_file)


