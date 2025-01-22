import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Performans metriklerini hesaplama fonksiyonu
def calculate_mape(y_true, y_pred):
    y_true = np.where(y_true == 0, np.nan, y_true)  # Sıfır değerleri nan yap
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100  # nan'ları dışarda tut

# Veri setini yükleme ve ön işleme
def load_and_preprocess_data(data_path, target_variable, features):
    data = pd.read_csv(data_path)
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    X_train, y_train = train_data[features].values, train_data[target_variable].values
    X_test, y_test = test_data[features].values, test_data[target_variable].values

    return X_train, X_test, y_train, y_test, scaler, data

# MLP modeli için hiperparametre optimizasyonu
def optimize_mlp(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'max_iter': [500, 1000]
    }

    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("En iyi MLP parametreler:", grid_search.best_params_)
    print("En iyi MLP skor:", -grid_search.best_score_)

    return grid_search.best_estimator_

# MLP modeli oluşturma ve değerlendirme
def run_mlp(model, X_test, y_test):
    # Tahminler
    predictions = model.predict(X_test)

    # Performans değerlendirmesi
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = calculate_mape(y_test, predictions)

    print("MLP Model Performansı:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return predictions

# XGBoost modeli için hiperparametre optimizasyonu
def optimize_xgboost(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200]
    }

    xgb = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("En iyi XGBoost parametreler:", grid_search.best_params_)
    print("En iyi XGBoost skor:", -grid_search.best_score_)

    return grid_search.best_estimator_

# XGBoost modeli oluşturma ve değerlendirme
def run_xgboost(model, X_test, y_test):
    # Tahminler
    predictions = model.predict(X_test)

    # Performans değerlendirmesi
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = calculate_mape(y_test, predictions)

    print("XGBoost Model Performansı:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return predictions

# Ana fonksiyon
def main():
    # Veri seti yolu
    data_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv'

    # Hedef değişken ve özellikler
    target_variable = 'Smf'
    features = ['Ptf', 'Smf_rolling_mean', 'Maxeslesmefiyati', 'Maxalisfiyati', 'Dolar', 'Euro', 'Maxsatisfiyati', 'Talepislemhacmi', 'Arzislemhacmi', 'Mineslesmefiyati']

    # Veriyi yükle ve ön işle
    X_train, X_test, y_train, y_test, scaler, data = load_and_preprocess_data(data_path, target_variable, features)

    # MLP hiperparametre optimizasyonu
    best_mlp_model = optimize_mlp(X_train, y_train)

    # Optimize edilen MLP modelini çalıştır
    mlp_predictions = run_mlp(best_mlp_model, X_test, y_test)

    # Tahmin sonuçlarını kaydet
    mlp_results_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\optimized_mlp_predictions.csv'
    mlp_results_df = pd.DataFrame({'Actual': y_test, 'Predicted': mlp_predictions})
    mlp_results_df.to_csv(mlp_results_path, index=False)
    print(f"Optimize edilmiş MLP tahminleri '{mlp_results_path}' konumuna kaydedildi.")

    # XGBoost hiperparametre optimizasyonu
    best_xgb_model = optimize_xgboost(X_train, y_train)

    # Optimize edilen XGBoost modelini çalıştır
    xgb_predictions = run_xgboost(best_xgb_model, X_test, y_test)

    # Tahmin sonuçlarını kaydet
    xgb_results_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\optimized_xgb_predictions.csv'
    xgb_results_df = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_predictions})
    xgb_results_df.to_csv(xgb_results_path, index=False)
    print(f"Optimize edilmiş XGBoost tahminleri '{xgb_results_path}' konumuna kaydedildi.")

if __name__ == "__main__":
    main()
