from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np


def mrmr_selection(data, target_variable, k, ignore_columns):
    """
    mRMR (Minimum Redundancy Maximum Relevance) yöntemi ile önemli özellikleri seçer.

    Args:
        data (pd.DataFrame): Veri seti.
        target_variable (str): Hedef değişken.
        k (int): Seçilecek özellik sayısı.
        ignore_columns (list): Göz ardı edilecek sütunlar.

    Returns:
        list: Seçilen özelliklerin isimleri.
    """
    # Hedef değişkeni ayır
    y = data[target_variable]
    X = data.drop(columns=[target_variable] + ignore_columns, errors='ignore')

    # Sayısal olmayan sütunları çıkar
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        print("Sayısal olmayan sütunlar çıkarılıyor:", non_numeric_columns)
        X = X.drop(columns=non_numeric_columns, errors='ignore')

    # Mutual Information hesapla
    mi_scores = mutual_info_regression(X, y)

    # Skorları DataFrame'e dönüştür ve sırala
    mi_df = pd.DataFrame({'Feature': X.columns, 'Score': mi_scores})
    mi_df = mi_df.sort_values(by='Score', ascending=False)

    # En iyi k özelliği seç
    selected_features = mi_df.head(k)['Feature'].tolist()

    return selected_features



if __name__ == "__main__":
    # Yeni veri setini yükle
    data_path = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\new_smf_weather_data.csv"
    data = pd.read_csv(data_path)

    # Manuel olarak göz ardı edilecek sütunlar
    ignore_columns = ['Ptfdolar', 'Ptfeuro', 'Smfdolar', 'Smfeuro', 'log_Ptf', 'log_Smf']

    # Hedef değişken ve seçilecek özellik sayısı
    target_variable = 'Smf'
    num_features_to_select = 10

    # mRMR seçimi
    selected_features = mrmr_selection(data, target_variable, num_features_to_select, ignore_columns)

    # Tarih sütununu seçilen sütunların başına ekle
    updated_data = data[["Tarih"] + selected_features + [target_variable]]

    # Güncellenmiş veri setini kaydet
    updated_data_path = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv"
    updated_data.to_csv(updated_data_path, index=False)

    print("Göz ardı edilen sütunlar:", ignore_columns)
    print("Seçilen Özellikler:")
    print(selected_features)
    print(f"Güncellenmiş veri seti '{updated_data_path}' konumuna kaydedildi.")


