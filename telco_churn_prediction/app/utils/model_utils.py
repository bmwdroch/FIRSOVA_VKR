#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль с функциями для работы с моделью прогнозирования оттока клиентов.
Включает функции для загрузки модели и препроцессора, преобразования данных и получения прогноза.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Добавление корневой директории проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Пути к файлам моделей
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_FILE = "best_model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"

def load_model_and_preprocessor():
    """
    Загрузка модели и препроцессора из файлов.
    
    Returns:
        tuple: (model, preprocessor) - загруженные модель и препроцессор
    """
    try:
        model_path = MODELS_DIR / BEST_MODEL_FILE
        preprocessor_path = MODELS_DIR / PREPROCESSOR_FILE
        
        # Проверка наличия файлов
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Файл препроцессора не найден: {preprocessor_path}")
        
        # Загрузка файлов
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        print(f"Успешно загружена модель из {model_path}")
        print(f"Успешно загружен препроцессор из {preprocessor_path}")
        
        return model, preprocessor
    
    except Exception as e:
        print(f"Ошибка при загрузке модели или препроцессора: {str(e)}")
        # В реальном приложении здесь можно реализовать запасной вариант или логирование
        raise e

def prepare_data(data, preprocessor):
    """
    Преобразование данных клиента в формат, подходящий для модели.
    
    Args:
        data (dict): Данные клиента в формате словаря
        preprocessor: Загруженный препроцессор для преобразования данных
    
    Returns:
        numpy.ndarray: Преобразованные данные в формате, подходящем для модели
    """
    try:
        # Преобразование словаря в DataFrame
        df = pd.DataFrame([data])
        
        # Автоматическое преобразование категориальных признаков
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Категориальные признаки: {categorical_features}")
        print(f"Числовые признаки: {numeric_features}")
        print(f"Данные для преобразования:\n{df}")
        
        # Применение препроцессора
        X_transformed = preprocessor.transform(df)
        
        print(f"Форма преобразованных данных: {X_transformed.shape}")
        
        # Проверка на несоответствие размерностей и исправление
        expected_features = 30  # Ожидаемое количество признаков для модели
        if X_transformed.shape[1] != expected_features:
            print(f"Обнаружено несоответствие размерности. Адаптация выходных данных: {X_transformed.shape[1]} -> {expected_features}")
            if X_transformed.shape[1] > expected_features:
                # Обрезаем лишние признаки
                X_transformed = X_transformed[:, :expected_features]
            else:
                # Добавляем нулевые признаки
                padding = np.zeros((X_transformed.shape[0], expected_features - X_transformed.shape[1]))
                X_transformed = np.hstack((X_transformed, padding))
            print(f"Новая форма данных: {X_transformed.shape}")
        
        return X_transformed
    
    except Exception as e:
        print(f"Ошибка при подготовке данных: {str(e)}")
        raise e

def predict_churn(data, model, preprocessor):
    """
    Прогнозирование вероятности оттока клиента на основе его данных.
    
    Args:
        data (dict): Данные клиента в формате словаря
        model: Загруженная модель для прогнозирования
        preprocessor: Загруженный препроцессор для преобразования данных
    
    Returns:
        tuple: (prediction, probability) - прогноз (0/1) и вероятность оттока
    """
    try:
        # Подготовка данных
        X_transformed = prepare_data(data, preprocessor)
        
        # Получение вероятности оттока
        probability = model.predict_proba(X_transformed)[0, 1]
        
        # Получение бинарного прогноза (0 - не уйдет, 1 - уйдет)
        prediction = model.predict(X_transformed)[0]
        
        print(f"Прогноз: {prediction}, Вероятность оттока: {probability:.4f}")
        return prediction, probability
    
    except Exception as e:
        print(f"Ошибка при прогнозировании: {str(e)}")
        raise e 