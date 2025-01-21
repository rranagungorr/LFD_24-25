import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def univariate_analysis(data, numerical_columns):
    """
    Univariate analiz: Histogram ve özet istatistikler. Hatalı sütunları atlar.

    Args:
        data (pd.DataFrame): Veri çerçevesi.
        numerical_columns (list): Sayısal sütunlar.
    """
    for col in numerical_columns:
        try:
            # Histogram çiz
            plt.figure(figsize=(8, 4))
            data[col].hist(bins=20, edgecolor='black', alpha=0.7)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

            # Özet istatistikler
            print(f"Summary Statistics for {col}:")
            print(data[col].describe())
            print(f"Skewness: {data[col].skew()}, Kurtosis: {data[col].kurt()}")
            print("-" * 50)

        except Exception as e:
            print(f"Hata veren sütun atlandı: {col}")
            print(f"Hata detayı: {e}")


def multivariate_analysis(data, numerical_columns, target_column):
    """
    Multivariate analiz: Korelasyon matrisi.

    Args:
        data (pd.DataFrame): Veri çerçevesi.
        numerical_columns (list): Sayısal sütunlar.
        target_column (str): Hedef sütun.
    """
    if target_column not in data.columns:
        print(f"Target column '{target_column}' not found.")
        return

    columns_to_analyze = numerical_columns + [target_column]
    correlation_matrix = data[columns_to_analyze].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, square=True)
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()
