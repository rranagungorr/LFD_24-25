import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def run_svr(train_features, train_target, test_features, test_target):
    """
    Support Vector Regression (SVR) modeli çalıştıran fonksiyon.
    """
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Modeli tanımla ve eğit
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(train_features_scaled, train_target)

    # Tahminler
    predictions = svr.predict(test_features_scaled)

    # Performans metriklerini hesapla
    mae = mean_absolute_error(test_target, predictions)
    mse = mean_squared_error(test_target, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_target - predictions) / test_target)) * 100

    print("SVR Model Performansı:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return svr, predictions

def SupportVectorRegression():
    # Veri setini yükle
    data_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv'
    data = pd.read_csv(data_path)

    # Hedef değişkeni ve özellikleri seç
    target_variable = 'Smf'
    features = ['Ptf', 'Smf_rolling_mean', 'Maxeslesmefiyati', 'Maxalisfiyati',
                'Dolar', 'Euro', 'Maxsatisfiyati', 'Talepislemhacmi', 'Arzislemhacmi', 'Mineslesmefiyati']

    # Train/Test split
    train_size = int(len(data) * 0.8)
    train_features = data[features][:train_size]
    train_target = data[target_variable][:train_size]
    test_features = data[features][train_size:]
    test_target = data[target_variable][train_size:]

    # SVR modelini çalıştır
    svr_model, svr_predictions = run_svr(train_features, train_target, test_features, test_target)

    # Tahmin sonuçlarını kaydet
    results_path = r'/data/processed/svr_predictions.csv'
    pd.DataFrame({'Actual': test_target.values, 'Predicted': svr_predictions}).to_csv(results_path, index=False)
    print(f"SVR tahminleri '{results_path}' konumuna kaydedildi.")


# Performans metriklerini hesaplama fonksiyonu
def calculate_mape(y_true, y_pred):
    y_true = np.where(y_true == 0, np.nan, y_true)  # Sıfır değerleri nan yap
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100  # nan'ları dışarda tut

# XGBoost modeli oluşturma
def run_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)

    # Tahminler
    predictions = model.predict(X_test)

    # Performans değerlendirmesi
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = calculate_mape(y_test.values, predictions)

    print("XGBoost Model Performansı:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return model, predictions

def xgb_regressor():
    # Veri setini yükleyin
    data_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv'
    data = pd.read_csv(data_path)

    # Özellikleri ve hedef değişkeni belirleme
    target_variable = 'Smf'
    features = ['Ptf', 'Smf_rolling_mean', 'Maxeslesmefiyati', 'Maxalisfiyati', 'Dolar', 'Euro', 'Maxsatisfiyati', 'Talepislemhacmi', 'Arzislemhacmi', 'Mineslesmefiyati']

    # Train/Test Split
    train_size = int(len(data) * 0.8)
    X = data[features]
    y = data[target_variable]
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # XGBoost modelini çalıştırın
    xgboost_model, xgboost_predictions = run_xgboost(X_train, X_test, y_train, y_test)

    # Tahmin sonuçlarını kaydedin
    results_path = r'/data/processed/xgboost_predictions.csv'
    xgboost_results = pd.DataFrame({'Actual': y_test, 'Predicted': xgboost_predictions})
    xgboost_results.to_csv(results_path, index=False)
    print(f"XGBoost tahminleri '{results_path}' konumuna kaydedildi.")



if __name__ == "__main__":
    xgb_regressor()
    SupportVectorRegression()
