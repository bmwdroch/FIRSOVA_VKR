#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения и сравнения моделей машинного обучения 
для прогнозирования оттока клиентов телекоммуникационной компании.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import xgboost as xgb

# Добавление корневой директории проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Определение путей для загрузки и сохранения данных
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "docs" / "figures" / "models"

# Имена файлов
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"
PREPROCESSOR_FILE = "preprocessor.joblib"

# Настройка для визуализаций
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

def create_directories():
    """Создание необходимых директорий для сохранения моделей и визуализаций"""
    directories = [MODELS_DIR, FIGURES_DIR]
    
    for directory in directories:
        if not directory.exists():
            print(f"Создание директории: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def load_data():
    """Загрузка предобработанных данных для обучения и тестирования"""
    print("Загрузка предобработанных данных...")
    
    train_path = PROCESSED_DATA_DIR / TRAIN_DATA_FILE
    test_path = PROCESSED_DATA_DIR / TEST_DATA_FILE
    
    if not train_path.exists() or not test_path.exists():
        print(f"Ошибка: Файлы данных не найдены в {PROCESSED_DATA_DIR}")
        return None, None, None, None
    
    # Загрузка данных
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Загружено {train_data.shape[0]} строк тренировочных данных и {test_data.shape[0]} строк тестовых данных")
    
    # Преобразование категориальных признаков в числовые
    categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != 'Churn']
    
    print(f"Категориальные признаки для кодирования: {categorical_columns}")
    
    # Преобразование Yes/No в 1/0
    for col in categorical_columns:
        # Обработка бинарных признаков
        if set(train_data[col].unique()) == {'Yes', 'No'}:
            train_data[col] = train_data[col].map({'Yes': 1, 'No': 0})
            test_data[col] = test_data[col].map({'Yes': 1, 'No': 0})
        else:
            # One-hot encoding для небинарных категориальных признаков
            dummies_train = pd.get_dummies(train_data[col], prefix=col, drop_first=True)
            dummies_test = pd.get_dummies(test_data[col], prefix=col, drop_first=True)
            
            # Обеспечение одинаковых столбцов в тренировочном и тестовом наборах
            for column in dummies_train.columns:
                if column not in dummies_test.columns:
                    dummies_test[column] = 0
            for column in dummies_test.columns:
                if column not in dummies_train.columns:
                    dummies_train[column] = 0
            
            # Добавление столбцов к данным
            train_data = pd.concat([train_data, dummies_train], axis=1)
            test_data = pd.concat([test_data, dummies_test], axis=1)
            
            # Удаление исходного столбца
            train_data.drop(columns=[col], inplace=True)
            test_data.drop(columns=[col], inplace=True)
    
    # Преобразование Churn в числовой формат
    if 'Churn' in train_data.columns and train_data['Churn'].dtype == 'object':
        train_data['Churn'] = train_data['Churn'].map({'Yes': 1, 'No': 0})
        test_data['Churn'] = test_data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Разделение на признаки и целевую переменную
    X_train = train_data.drop(columns=['Churn'])
    y_train = train_data['Churn']
    X_test = test_data.drop(columns=['Churn'])
    y_test = test_data['Churn']
    
    print(f"Размерность данных после преобразования: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Распределение классов в тренировочных данных: {dict(y_train.value_counts())}")
    print(f"Распределение классов в тестовых данных: {dict(y_test.value_counts())}")
    
    return X_train, y_train, X_test, y_test

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Обучение модели логистической регрессии"""
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
    print("="*50)
    
    # Создание и обучение модели
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Время обучения: {train_time:.2f} секунд")
    
    # Оценка модели
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['model_name'] = 'Логистическая регрессия'
    metrics['train_time'] = train_time
    
    # Сохранение модели
    model_path = MODELS_DIR / "logistic_regression_model.joblib"
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    return model, metrics

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Обучение модели дерева решений"""
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ ДЕРЕВА РЕШЕНИЙ")
    print("="*50)
    
    # Создание и обучение модели
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Время обучения: {train_time:.2f} секунд")
    
    # Оценка модели
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['model_name'] = 'Дерево решений'
    metrics['train_time'] = train_time
    
    # Сохранение модели
    model_path = MODELS_DIR / "decision_tree_model.joblib"
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Обучение модели случайного леса"""
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ СЛУЧАЙНОГО ЛЕСА")
    print("="*50)
    
    # Создание и обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Время обучения: {train_time:.2f} секунд")
    
    # Оценка модели
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['model_name'] = 'Случайный лес'
    metrics['train_time'] = train_time
    
    # Сохранение модели
    model_path = MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    return model, metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """Обучение модели XGBoost"""
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ XGBOOST")
    print("="*50)
    
    # Создание и обучение модели
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Балансировка классов
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Время обучения: {train_time:.2f} секунд")
    
    # Оценка модели
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['model_name'] = 'XGBoost'
    metrics['train_time'] = train_time
    
    # Сохранение модели
    model_path = MODELS_DIR / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    return model, metrics

def evaluate_model(y_true, y_pred, y_prob):
    """Оценка модели по различным метрикам"""
    metrics = {}
    
    # Базовые метрики
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Вывод результатов
    print("\nМетрики качества модели:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    print("\nМатрица ошибок:")
    print(cm)
    
    # Подробный отчет
    print("\nПодробный отчет:")
    print(classification_report(y_true, y_pred))
    
    return metrics

def compare_models(metrics_list):
    """Сравнение моделей по различным метрикам"""
    print("\n" + "="*50)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*50)
    
    # Создание DataFrame с метриками всех моделей
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.set_index('model_name')
    
    print("\nСравнение метрик моделей:")
    print(df_metrics[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'train_time']])
    
    # Визуализация метрик
    plt.figure(figsize=(14, 8))
    
    # Преобразование данных для удобной визуализации
    df_plot = df_metrics[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].reset_index()
    df_plot_melted = pd.melt(df_plot, id_vars='model_name', var_name='Метрика', value_name='Значение')
    
    # Построение графика
    sns.barplot(x='model_name', y='Значение', hue='Метрика', data=df_plot_melted)
    plt.title('Сравнение метрик качества моделей')
    plt.ylabel('Значение метрики')
    plt.xlabel('Модель')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png")
    
    # Определение лучшей модели по F1-мере
    best_model_name = df_metrics['f1_score'].idxmax()
    print(f"\nЛучшая модель по F1-мере: {best_model_name} (F1-score: {df_metrics.loc[best_model_name, 'f1_score']:.4f})")
    
    # Определение лучшей модели по ROC-AUC
    best_model_name_auc = df_metrics['roc_auc'].idxmax()
    print(f"Лучшая модель по ROC-AUC: {best_model_name_auc} (ROC-AUC: {df_metrics.loc[best_model_name_auc, 'roc_auc']:.4f})")
    
    return best_model_name

def plot_feature_importance(model, X_train, model_name):
    """Визуализация важности признаков для модели"""
    if not hasattr(model, 'feature_importances_'):
        print(f"Модель {model_name} не поддерживает анализ важности признаков")
        return
    
    # Получение важности признаков
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Создание DataFrame и сортировка
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=df_importance.head(15))
    plt.title(f'Важность признаков для модели {model_name}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    
    return df_importance

def main():
    """Основная функция для запуска обучения и сравнения моделей"""
    print("Начало обучения и сравнения моделей...")
    
    # Создание директорий для сохранения результатов
    create_directories()
    
    # Загрузка данных
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None:
        print("Ошибка при загрузке данных. Выход.")
        sys.exit(1)
    
    # Обучение и оценка моделей
    models_metrics = []
    
    # Логистическая регрессия
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    models_metrics.append(lr_metrics)
    
    # Дерево решений
    dt_model, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test)
    models_metrics.append(dt_metrics)
    plot_feature_importance(dt_model, X_train, "Дерево решений")
    
    # Случайный лес
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models_metrics.append(rf_metrics)
    plot_feature_importance(rf_model, X_train, "Случайный лес")
    
    # XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models_metrics.append(xgb_metrics)
    plot_feature_importance(xgb_model, X_train, "XGBoost")
    
    # Сравнение моделей
    best_model_name = compare_models(models_metrics)
    
    print("\nОбучение и сравнение моделей завершено.")
    print(f"Визуализации сохранены в директории {FIGURES_DIR}")
    print(f"Модели сохранены в директории {MODELS_DIR}")

if __name__ == "__main__":
    main() 