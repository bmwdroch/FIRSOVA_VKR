#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для загрузки датасета Telco Customer Churn.
"""

import os
import sys
import pandas as pd
import requests
from pathlib import Path

# URL для загрузки данных
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Определение путей к директориям
ROOT_DIR = Path(__file__).resolve().parents[2]  # Корневая директория проекта
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Имя файла для сохранения
RAW_DATA_FILE = "telco_customer_churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILE

def create_directories():
    """Создание необходимых директорий, если они не существуют"""
    directories = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]
    
    for directory in directories:
        if not directory.exists():
            print(f"Создание директории: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def download_dataset(url=DATA_URL, output_path=RAW_DATA_PATH):
    """
    Загрузка датасета с указанного URL и сохранение в указанный путь
    
    Args:
        url (str): URL для загрузки датасета
        output_path (Path): Путь для сохранения файла
    
    Returns:
        bool: True, если загрузка успешна, иначе False
    """
    try:
        print(f"Загрузка данных с {url}...")
        
        # Проверка, существует ли файл
        if output_path.exists():
            print(f"Файл {output_path} уже существует. Повторная загрузка не требуется.")
            return True
        
        # Загрузка данных
        response = requests.get(url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        
        # Сохранение файла
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Данные успешно загружены и сохранены в {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке данных: {e}")
        return False
    except IOError as e:
        print(f"Ошибка при сохранении файла: {e}")
        return False

def load_dataset(file_path=RAW_DATA_PATH):
    """
    Загрузка датасета из CSV-файла
    
    Args:
        file_path (Path): Путь к файлу CSV
    
    Returns:
        pandas.DataFrame: Загруженный датасет или None в случае ошибки
    """
    try:
        print(f"Чтение данных из {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Загружено {df.shape[0]} строк и {df.shape[1]} столбцов.")
        return df
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def main():
    """Основная функция для запуска скрипта"""
    print("Начало процесса загрузки данных...")
    
    # Создание директорий
    create_directories()
    
    # Загрузка датасета
    success = download_dataset()
    if not success:
        print("Не удалось загрузить данные. Выход.")
        sys.exit(1)
    
    # Проверка загруженных данных
    df = load_dataset()
    if df is not None:
        print("Проверка структуры данных:")
        print(df.head())
        print("\nСтатистика по данным:")
        print(df.describe())
        print("\nПроверка завершена успешно!")
    else:
        print("Не удалось загрузить и проверить данные.")
        sys.exit(1)
    
    print("Процесс загрузки данных успешно завершен.")

if __name__ == "__main__":
    main() 