from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def train_baseline_model(data_path):
    """
    Seçilen özelliklerle bir temel model oluşturur ve performansı değerlendirir.

    Args:
        data_path (str): Veri setinin yolu.
    """
    # Veri setini yükle
    data = pd.read_csv(data_path)

    # Hedef değişken ve özellikler
    target_variable = 'Smf'
    features = [
        'Ptf', 'Smf_rolling_mean', 'Maxeslesmefiyati', 'Maxalisfiyati',
        'Dolar', 'Euro', 'Maxsatisfiyati', 'Talepislemhacmi',
        'Arzislemhacmi', 'Mineslesmefiyati'
    ]

    # X ve y ayırma
    X = data[features]
    y = data[target_variable]

    # Eğitim ve test verisine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verileri standartlaştırma
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modeli eğitme
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Tahminler
    y_pred = model.predict(X_test_scaled)

    # Performans değerlendirme
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları yazdır
    print("Baseline Model Performansı:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return model, scaler


# Örnek Kullanım
if __name__ == "__main__":
    data_path = r"C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv"
    train_baseline_model(data_path)
