#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для предобработки данных Telco Customer Churn.
Выполняет очистку, преобразование категориальных признаков и
подготовку данных для обучения моделей машинного обучения.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Добавление корневой директории проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Импорт функций из модуля src.data
from src.data.download import load_dataset, RAW_DATA_PATH, DATA_DIR

# Определение путей для сохранения результатов
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT_DIR / "models"

# Имена файлов для сохранения
PROCESSED_DATA_FILE = "processed_telco_data.csv"
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"
PREPROCESSOR_FILE = "preprocessor.joblib"

def create_directories():
    """Создание необходимых директорий, если они не существуют"""
    directories = [PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR]
    
    for directory in directories:
        if not directory.exists():
            print(f"Создание директории: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def clean_data(df):
    """
    Очистка данных: обработка пропущенных значений, исправление типов данных и т.д.
    
    Args:
        df (pandas.DataFrame): Исходный датасет
    
    Returns:
        pandas.DataFrame: Очищенный датасет
    """
    print("Выполнение очистки данных...")
    
    # Создаем копию датафрейма
    df_cleaned = df.copy()
    
    # Проверка на пропущенные значения
    null_counts = df_cleaned.isnull().sum()
    print(f"Пропущенные значения до очистки:\n{null_counts[null_counts > 0]}")
    
    # Обработка пропущенных значений в TotalCharges
    # TotalCharges содержит пробелы, которые pandas интерпретирует как строки
    if 'TotalCharges' in df_cleaned.columns:
        # Преобразование TotalCharges в числовой формат
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        
        # Заполнение пропущенных значений в TotalCharges
        # Если клиент новый (tenure = 0 или 1), то TotalCharges = MonthlyCharges
        mask_new_customers = df_cleaned['tenure'].isin([0, 1]) & df_cleaned['TotalCharges'].isna()
        df_cleaned.loc[mask_new_customers, 'TotalCharges'] = df_cleaned.loc[mask_new_customers, 'MonthlyCharges']
        
        # Для остальных пропущенных значений используем среднее значение
        if df_cleaned['TotalCharges'].isna().sum() > 0:
            print(f"Заполнение {df_cleaned['TotalCharges'].isna().sum()} пропущенных значений в TotalCharges")
            df_cleaned['TotalCharges'].fillna(df_cleaned['TotalCharges'].mean(), inplace=True)
    
    # Преобразование SeniorCitizen (0/1) в категориальный тип
    if 'SeniorCitizen' in df_cleaned.columns:
        df_cleaned['SeniorCitizen'] = df_cleaned['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    
    # Проверка после очистки
    null_counts_after = df_cleaned.isnull().sum()
    print(f"Пропущенные значения после очистки:\n{null_counts_after[null_counts_after > 0]}")
    
    # Удаление дубликатов, если они есть
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        print(f"Удаление {duplicates} дубликатов")
        df_cleaned.drop_duplicates(inplace=True)
    
    # Удаление ненужных столбцов (например, customerID)
    if 'customerID' in df_cleaned.columns:
        df_cleaned.drop(columns=['customerID'], inplace=True)
        print("Удален столбец customerID")
    
    print("Очистка данных завершена.")
    return df_cleaned

def prepare_features_and_target(df, target_column='Churn'):
    """
    Разделение данных на признаки и целевую переменную
    
    Args:
        df (pandas.DataFrame): Очищенный датасет
        target_column (str): Имя столбца с целевой переменной
    
    Returns:
        tuple: (X, y) - признаки и целевая переменная
    """
    print("Подготовка признаков и целевой переменной...")
    
    # Преобразование целевой переменной в бинарный формат
    if target_column in df.columns:
        # Копирование датафрейма
        df_copy = df.copy()
        
        # Преобразование Churn в бинарный формат (1 - Yes, 0 - No)
        df_copy[target_column] = df_copy[target_column].map({'Yes': 1, 'No': 0})
        
        # Разделение на признаки и целевую переменную
        X = df_copy.drop(columns=[target_column])
        y = df_copy[target_column]
        
        print(f"Подготовлено {X.shape[1]} признаков и {len(y)} экземпляров")
        print(f"Распределение целевой переменной: 0 - {(y == 0).sum()}, 1 - {(y == 1).sum()}")
        
        return X, y
    else:
        print(f"Предупреждение: Столбец {target_column} не найден в датасете")
        return df, None

def create_preprocessor(X):
    """
    Создание пайплайна для предобработки данных (кодирование категориальных признаков и масштабирование числовых)
    
    Args:
        X (pandas.DataFrame): Признаки
    
    Returns:
        sklearn.compose.ColumnTransformer: Пайплайн предобработки
    """
    print("Создание пайплайна для предобработки данных...")
    
    # Определение числовых и категориальных признаков
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Числовые признаки ({len(numeric_features)}): {numeric_features}")
    print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")
    
    # Создание пайплайнов предобработки
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Создание ColumnTransformer для применения соответствующих преобразований к каждому типу признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Разделение данных на обучающую и тестовую выборки
    
    Args:
        X (pandas.DataFrame): Признаки
        y (pandas.Series): Целевая переменная
        test_size (float): Доля тестовой выборки (от 0 до 1)
        random_state (int): Зерно генератора случайных чисел для воспроизводимости
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - разделенные выборки
    """
    print(f"Разделение данных на обучающую ({1-test_size:.0%}) и тестовую ({test_size:.0%}) выборки...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Размеры выборок: обучающая - {X_train.shape}, тестовая - {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor=None):
    """
    Сохранение предобработанных данных и препроцессора
    
    Args:
        X_train (pandas.DataFrame): Признаки обучающей выборки
        X_test (pandas.DataFrame): Признаки тестовой выборки
        y_train (pandas.Series): Целевая переменная обучающей выборки
        y_test (pandas.Series): Целевая переменная тестовой выборки
        preprocessor (sklearn object): Пайплайн предобработки данных
    """
    print("Сохранение предобработанных данных...")
    
    # Сохранение обучающей выборки
    train_data = X_train.copy()
    train_data['Churn'] = y_train
    train_data.to_csv(PROCESSED_DATA_DIR / TRAIN_DATA_FILE, index=False)
    print(f"Обучающая выборка сохранена в {PROCESSED_DATA_DIR / TRAIN_DATA_FILE}")
    
    # Сохранение тестовой выборки
    test_data = X_test.copy()
    test_data['Churn'] = y_test
    test_data.to_csv(PROCESSED_DATA_DIR / TEST_DATA_FILE, index=False)
    print(f"Тестовая выборка сохранена в {PROCESSED_DATA_DIR / TEST_DATA_FILE}")
    
    # Сохранение препроцессора, если он предоставлен
    if preprocessor is not None:
        # Создание директории models, если она не существует
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(preprocessor, MODELS_DIR / PREPROCESSOR_FILE)
        print(f"Препроцессор сохранен в {MODELS_DIR / PREPROCESSOR_FILE}")

def main():
    """Основная функция для запуска скрипта предобработки данных"""
    print("Начало процесса предобработки данных...")
    
    # Создание директорий для сохранения результатов
    create_directories()
    
    # Загрузка данных
    df = load_dataset()
    if df is None:
        print("Не удалось загрузить данные. Выход.")
        sys.exit(1)
    
    # Очистка данных
    df_cleaned = clean_data(df)
    
    # Разделение на признаки и целевую переменную
    X, y = prepare_features_and_target(df_cleaned)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Создание и обучение препроцессора
    preprocessor = create_preprocessor(X)
    
    # Сохранение данных и препроцессора
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor)
    
    print("Процесс предобработки данных успешно завершен.")

if __name__ == "__main__":
    main() 