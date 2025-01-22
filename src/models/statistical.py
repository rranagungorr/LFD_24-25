import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Veri setini yükleme ve ön işleme

def load_data(data_path, target_variable):
    data = pd.read_csv(data_path)
    data['Tarih'] = pd.to_datetime(data['Tarih'])
    data.set_index('Tarih', inplace=True)
    features = ['Ptf', 'Smf_rolling_mean', 'Maxeslesmefiyati', 'Maxalisfiyati', 'Dolar', 'Euro', 'Maxsatisfiyati',
                'Talepislemhacmi', 'Arzislemhacmi', 'Mineslesmefiyati']
    return data, features

# Train-test split fonksiyonu

def split_data(data, target_variable, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Performans metriklerini ölçmek için

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"{model_name} Model Performansı:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return mae, mse, rmse

# ARIMA modeli

def run_arima(train, test, order=(1, 1, 1)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))

    return model_fit, predictions

# SARIMA modeli

def run_sarima(train, test, target_variable, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
    model = SARIMAX(train[target_variable], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(test))

    return model_fit, predictions

# Random Forest modeli

def run_random_forest(train, test, features, target_variable):
    X_train = train[features]
    y_train = train[target_variable]
    X_test = test[features]
    y_test = test[target_variable]

    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, predictions

# Ana fonksiyon

def main():
    data_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv'
    target_variable = 'Smf'

    # Veriyi yükle ve ayır
    data, features = load_data(data_path, target_variable)
    train_data, test_data = split_data(data, target_variable)

    # ARIMA Model
    arima_model, arima_predictions = run_arima(train_data[target_variable], test_data[target_variable])
    evaluate_model(test_data[target_variable], arima_predictions, "ARIMA")

    # ARIMA Tahminlerini Kaydetme
    arima_results_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\arima_predictions.csv'
    pd.Series(arima_predictions, index=test_data.index).to_csv(arima_results_path, header=['Predictions'])
    print(f"ARIMA tahminleri '{arima_results_path}' konumuna kaydedildi.")

    # SARIMA Model
    sarima_model, sarima_predictions = run_sarima(train_data, test_data, target_variable)
    evaluate_model(test_data[target_variable], sarima_predictions, "SARIMA")

    # SARIMA Tahminlerini Kaydetme
    sarima_results_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\sarima_predictions.csv'
    pd.Series(sarima_predictions, index=test_data.index).to_csv(sarima_results_path, header=['Predictions'])
    print(f"SARIMA tahminleri '{sarima_results_path}' konumuna kaydedildi.")

    # Random Forest Model
    rf_model, rf_predictions = run_random_forest(train_data, test_data, features, target_variable)
    evaluate_model(test_data[target_variable], rf_predictions, "Random Forest")

    # Random Forest Tahminlerini Kaydetme
    rf_results_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\random_forest_predictions.csv'
    pd.DataFrame({'Actual': test_data[target_variable], 'Predicted': rf_predictions}).to_csv(rf_results_path, index=False)
    print(f"Random Forest tahminleri '{rf_results_path}' konumuna kaydedildi.")

def calculate_mape(y_true, y_pred):
    """
    MAPE (Mean Absolute Percentage Error) hesaplama fonksiyonu.
    Sıfırdan bölme hatalarını önlemek için `y_true` sıfır olan değerleri filtreler.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # Sıfır değerleri filtrele
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

def load_predictions(arima_path, sarima_path, rf_path, data_path):
    """
    Tahmin sonuçlarını ve test verisini yükleyen fonksiyon.
    """
    arima_preds = pd.read_csv(arima_path)['Predictions']
    sarima_preds = pd.read_csv(sarima_path)['Predictions']
    rf_preds = pd.read_csv(rf_path)['Predicted']
    data = pd.read_csv(data_path)

    return arima_preds, sarima_preds, rf_preds, data

def calculate_metrics(test_data, arima_preds, sarima_preds, rf_preds, target_variable):
    """
    Modellerin performans metriklerini hesaplayan fonksiyon.
    """
    metrics = []
    for model_name, predictions in [('ARIMA', arima_preds), ('SARIMA', sarima_preds), ('Random Forest', rf_preds)]:
        mae = mean_absolute_error(test_data, predictions)
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mape = calculate_mape(test_data, predictions)
        metrics.append({'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})

    return pd.DataFrame(metrics)

def save_metrics(metrics_df, output_path):
    """
    Performans metriklerini bir CSV dosyasına kaydeden fonksiyon.
    """
    metrics_df.to_csv(output_path, index=False)
    print(f"Model performansı karşılaştırması '{output_path}' konumuna kaydedildi.")

def mainn():
    # Dosya yolları
    arima_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\arima_predictions.csv'
    sarima_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\sarima_predictions.csv'
    rf_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\random_forest_predictions.csv'
    data_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\data\processed\final_feature_selection.csv'
    output_path = r'C:\Users\PC\Documents\GitHub\LFD_24-25\results\model_performance_metrics.csv'

    # Verileri yükle
    arima_preds, sarima_preds, rf_preds, data = load_predictions(arima_path, sarima_path, rf_path, data_path)

    # Test verisini seç
    target_variable = 'Smf'
    train_size = int(len(data) * 0.8)
    test_data = data[target_variable][train_size:]

    # Performans metriklerini hesapla
    metrics_df = calculate_metrics(test_data, arima_preds, sarima_preds, rf_preds, target_variable)

    # Performans metriklerini göster
    print(metrics_df)

    # Performans metriklerini kaydet
    save_metrics(metrics_df, output_path)


if __name__ == "__main__":
    mainn()