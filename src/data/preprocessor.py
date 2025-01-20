import pandas as pd

# Dosya yollarını belirleyin
merged_weather_path = r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\Merged_weather_data.csv'
smfdb_path = r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\smfdb.csv'

# CSV dosyalarını yükleyin
merged_weather_data = pd.read_csv(merged_weather_path)
smfdb_data = pd.read_csv(smfdb_path)

# Verileri sütun bazında birleştirin (axis=1)
integrated_data = pd.concat([smfdb_data, merged_weather_data], axis=1)

# Birleştirilmiş veriyi kaydet
integrated_output_path = r'C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\processed\smfdb_weather_data.csv'
integrated_data.to_csv(integrated_output_path, index=False)

print(f"Birleştirilmiş veri '{integrated_output_path}' konumuna kaydedildi.")