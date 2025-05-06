#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для исследовательского анализа данных набора Telco Customer Churn.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Добавление корневой директории проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Импорт функций из модуля src.data
from src.data.download import load_dataset, RAW_DATA_PATH, DATA_DIR

# Настройка для отображения более полной информации в DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Настройка стиля визуализаций
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Пути для сохранения результатов
OUTPUT_DIR = ROOT_DIR / "docs" / "figures"
DATA_DICT_PATH = ROOT_DIR / "docs" / "data_dictionary.md"

def create_output_dir(dir_path):
    """Создание директории, если она не существует"""
    if not dir_path.exists():
        print(f"Создание директории: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)

def describe_dataset(df):
    """Вывод основной информации о датасете"""
    print(f"\n{'='*50}")
    print("ОПИСАНИЕ ДАТАСЕТА")
    print(f"{'='*50}")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nТипы данных:")
    print(df.dtypes)
    print("\nСтатистика по числовым признакам:")
    print(df.describe())
    print("\nСтатистика по категориальным признакам:")
    print(df.describe(include=['object']))
    print("\nПропущенные значения:")
    print(df.isnull().sum())
    
    # Проверка на дубликаты
    duplicates = df.duplicated().sum()
    print(f"\nКоличество дубликатов: {duplicates}")

def analyze_target_distribution(df, target_column='Churn'):
    """Анализ распределения целевой переменной"""
    print(f"\n{'='*50}")
    print("АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print(f"{'='*50}")
    
    # Распределение классов
    class_counts = df[target_column].value_counts()
    print("Распределение классов:")
    print(class_counts)
    
    # Процентное соотношение
    class_percentage = df[target_column].value_counts(normalize=True) * 100
    print("\nПроцентное соотношение классов:")
    print(class_percentage)
    
    # Визуализация распределения
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=target_column, data=df)
    plt.title('Распределение целевой переменной (Churn)')
    plt.ylabel('Количество клиентов')
    
    # Добавление текстовых аннотаций
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', fontsize=12)
        
    plt.savefig(OUTPUT_DIR / 'target_distribution.png')
    
    # Круговая диаграмма
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', 
            colors=['#66b3ff','#ff9999'])
    plt.title('Соотношение клиентов по оттоку')
    plt.savefig(OUTPUT_DIR / 'target_pie_chart.png')

def analyze_categorical_features(df, target_column='Churn'):
    """Анализ категориальных признаков"""
    print(f"\n{'='*50}")
    print("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print(f"{'='*50}")
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != target_column and col != 'customerID']
    
    print(f"Категориальные признаки ({len(categorical_columns)}): {categorical_columns}")
    
    # Создание сводной таблицы для всех категориальных признаков и их взаимосвязи с целевой переменной
    plt.figure(figsize=(16, 20))
    rows = 4  # Увеличиваем количество рядов
    cols = 4  # Количество столбцов
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(rows, cols, i)  # 4x4 сетка для размещения до 16 признаков
        # Используем готовый бинарный столбец для расчета процента
        churn_rate = df.groupby(column)['Churn_Binary'].mean() * 100
        churn_rate.sort_values(ascending=False).plot(kind='bar', color='skyblue')
        plt.title(f'{column} vs. Churn Rate')
        plt.ylabel('Churn Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'categorical_features_churn_rate.png')
    
    # Детальный анализ каждого категориального признака
    for column in categorical_columns:
        print(f"\nРаспределение признака '{column}':")
        value_counts = df[column].value_counts()
        print(value_counts)
        
        # Связь с целевой переменной
        cross_tab = pd.crosstab(df[column], df[target_column], normalize='index') * 100
        print(f"\nОтток по категориям признака '{column}' (%):")
        print(cross_tab)
        
        # Визуализация
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.countplot(x=column, data=df)
        plt.title(f'Распределение признака {column}')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 2, 2)
        sns.countplot(x=column, hue=target_column, data=df)
        plt.title(f'Отток по категориям {column}')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'categorical_{column}.png')

def analyze_numerical_features(df, target_column='Churn'):
    """Анализ числовых признаков"""
    print(f"\n{'='*50}")
    print("АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print(f"{'='*50}")
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Числовые признаки ({len(numerical_columns)}): {numerical_columns}")
    
    # Общая визуализация распределения числовых признаков
    plt.figure(figsize=(16, 12))
    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(3, 3, i)  # Адаптируйте размер сетки в зависимости от количества признаков
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Распределение {column}')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'numerical_features_distribution.png')
    
    # Детальный анализ каждого числового признака
    for column in numerical_columns:
        print(f"\nСтатистика признака '{column}':")
        print(df[column].describe())
        
        # Визуализация распределения и связи с целевой переменной
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Распределение признака {column}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=target_column, y=column, data=df)
        plt.title(f'Распределение {column} по классам')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'numerical_{column}.png')
        
        # Дополнительный анализ для специфических признаков
        if column == 'tenure':
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='tenure', y=target_column, data=df, estimator='mean', errorbar=None)
            plt.title('Вероятность оттока в зависимости от срока обслуживания')
            plt.xlabel('Срок обслуживания (месяцы)')
            plt.ylabel('Вероятность оттока')
            plt.savefig(OUTPUT_DIR / 'tenure_churn_probability.png')

def correlation_analysis(df):
    """Анализ корреляций между признаками"""
    print(f"\n{'='*50}")
    print("АНАЛИЗ КОРРЕЛЯЦИЙ")
    print(f"{'='*50}")
    
    # Преобразование категориальных признаков для анализа корреляций
    # Включаем только числовые и бинарные категориальные признаки
    df_corr = df.copy()
    
    # Преобразование 'Yes'/'No' в 1/0
    for column in df_corr.select_dtypes(include=['object']).columns:
        if df_corr[column].nunique() <= 2:
            if set(df_corr[column].unique()) == {'Yes', 'No'}:
                df_corr[column] = df_corr[column].map({'Yes': 1, 'No': 0})
            elif column == 'Churn':
                df_corr[column] = df_corr[column].map({'Yes': 1, 'No': 0})
    
    # Отбор числовых колонок после преобразования
    numeric_columns = df_corr.select_dtypes(include=['int64', 'float64']).columns
    numeric_columns = [col for col in numeric_columns if col != 'customerID']
    
    # Матрица корреляций
    corr_matrix = df_corr[numeric_columns].corr()
    print("\nМатрица корреляций:")
    print(corr_matrix.round(2))
    
    # Визуализация матрицы корреляций
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Матрица корреляций')
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png')
    
    # Корреляция с целевой переменной
    if 'Churn' in numeric_columns:
        target_corr = corr_matrix['Churn'].sort_values(ascending=False)
        print("\nКорреляция признаков с целевой переменной (Churn):")
        print(target_corr)
        
        # Визуализация корреляции с целевой переменной
        plt.figure(figsize=(12, 8))
        sns.barplot(x=target_corr.index, y=target_corr.values)
        plt.title('Корреляция признаков с целевой переменной (Churn)')
        plt.xticks(rotation=45, ha='right')
        plt.savefig(OUTPUT_DIR / 'target_correlation.png')

def create_data_dictionary(df):
    """Создание словаря данных с описанием признаков"""
    print(f"\n{'='*50}")
    print("СОЗДАНИЕ СЛОВАРЯ ДАННЫХ")
    print(f"{'='*50}")
    
    # Словарь с описаниями для каждого признака
    feature_descriptions = {
        'customerID': 'Уникальный идентификатор клиента',
        'gender': 'Пол клиента (Male/Female)',
        'SeniorCitizen': 'Является ли клиент пожилым (1) или нет (0)',
        'Partner': 'Имеет ли клиент партнера (Yes/No)',
        'Dependents': 'Имеет ли клиент иждивенцев (Yes/No)',
        'tenure': 'Количество месяцев, в течение которых клиент пользуется услугами компании',
        'PhoneService': 'Подключена ли телефонная услуга (Yes/No)',
        'MultipleLines': 'Подключены ли несколько телефонных линий (Yes/No/No phone service)',
        'InternetService': 'Тип интернет-услуги (DSL, Fiber optic, No)',
        'OnlineSecurity': 'Подключена ли услуга онлайн-безопасности (Yes/No/No internet service)',
        'OnlineBackup': 'Подключена ли услуга онлайн-резервного копирования (Yes/No/No internet service)',
        'DeviceProtection': 'Подключена ли услуга защиты устройства (Yes/No/No internet service)',
        'TechSupport': 'Подключена ли услуга технической поддержки (Yes/No/No internet service)',
        'StreamingTV': 'Подключена ли услуга потокового ТВ (Yes/No/No internet service)',
        'StreamingMovies': 'Подключена ли услуга потокового видео (Yes/No/No internet service)',
        'Contract': 'Тип контракта (Month-to-month, One year, Two year)',
        'PaperlessBilling': 'Использует ли клиент безбумажный расчет (Yes/No)',
        'PaymentMethod': 'Способ оплаты (Electronic check, Mailed check, Bank transfer, Credit card)',
        'MonthlyCharges': 'Ежемесячная плата',
        'TotalCharges': 'Общая сумма платежей',
        'Churn': 'Ушел ли клиент (Yes/No)'
    }
    
    # Создание словаря данных в формате Markdown
    with open(DATA_DICT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Словарь данных - Telco Customer Churn\n\n")
        f.write("## Общая информация\n\n")
        f.write(f"- Количество строк: {df.shape[0]}\n")
        f.write(f"- Количество столбцов: {df.shape[1]}\n")
        f.write(f"- Целевая переменная: Churn (отток клиентов)\n\n")
        
        f.write("## Описание признаков\n\n")
        f.write("| Признак | Тип данных | Описание | Уникальные значения | Пропущенные значения |\n")
        f.write("|---------|------------|----------|---------------------|----------------------|\n")
        
        for column in df.columns:
            data_type = str(df[column].dtype)
            description = feature_descriptions.get(column, "Нет описания")
            unique_values = df[column].nunique()
            unique_examples = ", ".join(map(str, df[column].unique()[:5]))
            if len(df[column].unique()) > 5:
                unique_examples += ", ..."
            missing_values = df[column].isnull().sum()
            
            f.write(f"| {column} | {data_type} | {description} | {unique_values} ({unique_examples}) | {missing_values} |\n")
    
    print(f"Словарь данных создан и сохранен в {DATA_DICT_PATH}")

def main():
    """Основная функция для запуска скрипта"""
    print("Начало исследовательского анализа данных...")
    
    # Создание директории для сохранения результатов
    create_output_dir(OUTPUT_DIR)
    
    # Загрузка данных
    df = load_dataset()
    if df is None:
        print("Ошибка при загрузке данных. Выход.")
        sys.exit(1)
    
    # Преобразование TotalCharges из строкового в числовой формат
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Обработка пропущенных значений в TotalCharges
    print(f"Количество пропущенных значений в TotalCharges: {df['TotalCharges'].isna().sum()}")
    
    # Заполняем пропущенные значения в TotalCharges средним значением
    if df['TotalCharges'].isna().sum() > 0:
        df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # Анализ датасета
    describe_dataset(df)
    
    # Анализ целевой переменной
    analyze_target_distribution(df)
    
    # Преобразование целевой переменной в бинарную для анализа
    df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Анализ категориальных признаков
    analyze_categorical_features(df, target_column='Churn')
    
    # Анализ числовых признаков
    analyze_numerical_features(df)
    
    # Анализ корреляций
    correlation_analysis(df)
    
    # Создание словаря данных
    create_data_dictionary(df)
    
    print("\nИсследовательский анализ данных завершен.")
    print(f"Визуализации сохранены в директории {OUTPUT_DIR}")
    print(f"Словарь данных сохранен в {DATA_DICT_PATH}")

if __name__ == "__main__":
    main() 