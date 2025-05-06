#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для создания и обучения препроцессора данных Telco Customer Churn.
Этот скрипт исправляет проблему с необученным ColumnTransformer.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Добавление корневой директории проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Определение путей для загрузки и сохранения данных
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
TRAIN_DATA_FILE = "train_data.csv"
PREPROCESSOR_FILE = "preprocessor.joblib"

def load_training_data():
    """
    Загрузка обучающих данных из файла
    
    Returns:
        pandas.DataFrame: Загруженный датафрейм с данными
    """
    data_path = PROCESSED_DATA_DIR / TRAIN_DATA_FILE
    
    if not data_path.exists():
        print(f"Ошибка: Файл данных не найден: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        print(f"Данные успешно загружены из {data_path}: {df.shape[0]} строк, {df.shape[1]} столбцов")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных: {str(e)}")
        return None

def create_and_fit_preprocessor(df):
    """
    Создание и обучение пайплайна для предобработки данных
    
    Args:
        df (pandas.DataFrame): Данные для обучения препроцессора
        
    Returns:
        sklearn.compose.ColumnTransformer: Обученный пайплайн предобработки
    """
    print("Создание и обучение пайплайна для предобработки данных...")
    
    # Разделение на признаки и целевую переменную
    X = df.drop(columns=['Churn'], errors='ignore')
    
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
    
    # Важно: обучаем препроцессор на данных
    print("Обучение препроцессора на данных...")
    preprocessor.fit(X)
    print("Препроцессор успешно обучен!")
    
    return preprocessor

def save_preprocessor(preprocessor):
    """
    Сохранение обученного препроцессора в файл
    
    Args:
        preprocessor: Обученный пайплайн предобработки
    """
    # Создание директории models, если она не существует
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    preprocessor_path = MODELS_DIR / PREPROCESSOR_FILE
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Препроцессор сохранен в {preprocessor_path}")

def main():
    """Основная функция для исправления препроцессора"""
    print("Начало процесса создания и обучения препроцессора...")
    
    # Загрузка обучающих данных
    train_data = load_training_data()
    if train_data is None:
        print("Не удалось загрузить данные. Выход.")
        sys.exit(1)
    
    # Создание и обучение препроцессора
    preprocessor = create_and_fit_preprocessor(train_data)
    
    # Сохранение препроцессора
    save_preprocessor(preprocessor)
    
    print("Процесс создания и обучения препроцессора успешно завершен.")

if __name__ == "__main__":
    main() 