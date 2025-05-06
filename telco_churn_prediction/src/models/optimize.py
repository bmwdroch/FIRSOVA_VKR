#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для оптимизации моделей машинного обучения
для прогнозирования оттока клиентов телекоммуникационной компании.
Включает методы для балансировки классов, настройки гиперпараметров,
создания новых признаков и ансамблевых моделей.
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import (
    KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, 
    train_test_split, StratifiedKFold
)
from sklearn.feature_selection import (
    SelectFromModel, RFE, RFECV, SelectKBest, f_classif
)
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
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
LR_MODEL_FILE = "logistic_regression_model.joblib"
XGB_MODEL_FILE = "xgboost_model.joblib"
OPTIMIZED_LR_MODEL_FILE = "optimized_logistic_regression_model.joblib"
OPTIMIZED_XGB_MODEL_FILE = "optimized_xgboost_model.joblib"
ENSEMBLE_MODEL_FILE = "ensemble_model.joblib"
BEST_MODEL_FILE = "best_model.joblib"

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

def load_models():
    """Загрузка обученных моделей из директории models"""
    print("Загрузка обученных моделей...")
    
    lr_path = MODELS_DIR / LR_MODEL_FILE
    xgb_path = MODELS_DIR / XGB_MODEL_FILE
    
    if not lr_path.exists() or not xgb_path.exists():
        print(f"Ошибка: Файлы моделей не найдены в {MODELS_DIR}")
        return None, None
    
    # Загрузка моделей
    lr_model = joblib.load(lr_path)
    xgb_model = joblib.load(xgb_path)
    
    print(f"Модели успешно загружены: Логистическая регрессия и XGBoost")
    
    return lr_model, xgb_model

def evaluate_model(y_true, y_pred, y_prob, model_name=""):
    """Оценка модели по различным метрикам"""
    metrics = {}
    
    # Базовые метрики
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Вывод результатов
    print(f"\nМетрики качества модели {model_name}:")
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

def balance_classes(X_train, y_train, method='smote', random_state=42):
    """
    Балансировка классов с использованием различных методов
    
    Args:
        X_train: Признаки тренировочной выборки
        y_train: Целевая переменная тренировочной выборки
        method: Метод балансировки ('smote', 'adasyn', 'undersampling', 'smoteenn', 'smotetomek')
        random_state: Зерно генератора случайных чисел
        
    Returns:
        X_resampled, y_resampled: Сбалансированные выборки
    """
    print(f"Балансировка классов с использованием метода {method}...")
    
    # Отображение исходного распределения классов
    print(f"Распределение классов до балансировки: {dict(pd.Series(y_train).value_counts())}")
    
    if method == 'smote':
        resampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        resampler = ADASYN(random_state=random_state)
    elif method == 'undersampling':
        resampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smoteenn':
        resampler = SMOTEENN(random_state=random_state)
    elif method == 'smotetomek':
        resampler = SMOTETomek(random_state=random_state)
    else:
        print(f"Неизвестный метод балансировки: {method}. Используется SMOTE.")
        resampler = SMOTE(random_state=random_state)
    
    # Применение ресэмплинга
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    
    # Отображение нового распределения классов
    print(f"Распределение классов после балансировки: {dict(pd.Series(y_resampled).value_counts())}")
    print(f"Новый размер выборки: {X_resampled.shape}")
    
    return X_resampled, y_resampled

def create_interaction_features(X_train, X_test):
    """
    Создание признаков-взаимодействий между числовыми переменными
    
    Args:
        X_train: Признаки тренировочной выборки
        X_test: Признаки тестовой выборки
        
    Returns:
        X_train_new, X_test_new: Выборки с добавленными признаками
    """
    print("Создание признаков-взаимодействий...")
    
    # Определение числовых признаков
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Числовые признаки для создания взаимодействий: {numeric_features}")
    
    if len(numeric_features) < 2:
        print("Недостаточно числовых признаков для создания взаимодействий")
        return X_train, X_test
    
    # Копии датафреймов
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # Создание новых признаков - произведений пар числовых признаков
    for i, feat1 in enumerate(numeric_features):
        for feat2 in numeric_features[i+1:]:
            new_feature = f"{feat1}_x_{feat2}"
            X_train_new[new_feature] = X_train[feat1] * X_train[feat2]
            X_test_new[new_feature] = X_test[feat1] * X_test[feat2]
            
            # Создание отношений признаков (только если знаменатель не равен 0)
            new_feature_ratio = f"{feat1}_div_{feat2}"
            X_train_new[new_feature_ratio] = X_train[feat1] / (X_train[feat2] + 1e-10)
            X_test_new[new_feature_ratio] = X_test[feat1] / (X_test[feat2] + 1e-10)
    
    print(f"Созданы новые признаки. Новые размерности: X_train {X_train_new.shape}, X_test {X_test_new.shape}")
    
    return X_train_new, X_test_new

def create_polynomial_features(X_train, X_test, degree=2):
    """
    Создание полиномиальных признаков
    
    Args:
        X_train: Признаки тренировочной выборки
        X_test: Признаки тестовой выборки
        degree: Степень полинома
        
    Returns:
        X_train_poly, X_test_poly: Выборки с полиномиальными признаками
    """
    print(f"Создание полиномиальных признаков степени {degree}...")
    
    # Определение числовых признаков
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Числовые признаки для полиномиальных преобразований: {numeric_features}")
    
    if len(numeric_features) == 0:
        print("Нет числовых признаков для полиномиальных преобразований")
        return X_train, X_test
    
    # Создание объекта для полиномиальных преобразований
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Применение преобразования к числовым признакам
    X_train_numeric = poly.fit_transform(X_train[numeric_features])
    X_test_numeric = poly.transform(X_test[numeric_features])
    
    # Получение имен новых признаков
    poly_feature_names = [f"{feat}^{degree}" for feat in numeric_features]
    
    # Создание новых датафреймов
    X_train_numeric_df = pd.DataFrame(X_train_numeric, columns=poly_feature_names)
    X_test_numeric_df = pd.DataFrame(X_test_numeric, columns=poly_feature_names)
    
    # Удаление исходных признаков из новых датафреймов (они будут добавлены из оригинальных датафреймов)
    X_train_numeric_df = X_train_numeric_df.iloc[:, len(numeric_features):]
    X_test_numeric_df = X_test_numeric_df.iloc[:, len(numeric_features):]
    
    # Объединение с оригинальными датафреймами
    X_train_poly = pd.concat([X_train, X_train_numeric_df], axis=1)
    X_test_poly = pd.concat([X_test, X_test_numeric_df], axis=1)
    
    print(f"Созданы полиномиальные признаки. Новые размерности: X_train {X_train_poly.shape}, X_test {X_test_poly.shape}")
    
    return X_train_poly, X_test_poly

def select_features(X_train, y_train, X_test, method='selectfrommodel', estimator=None, n_features=None):
    """
    Отбор наиболее важных признаков
    
    Args:
        X_train: Признаки тренировочной выборки
        y_train: Целевая переменная тренировочной выборки
        X_test: Признаки тестовой выборки
        method: Метод отбора признаков ('selectfrommodel', 'rfe', 'rfecv', 'selectkbest')
        estimator: Модель для оценки важности признаков (для 'selectfrommodel' и 'rfe')
        n_features: Количество признаков для отбора (для 'rfe' и 'selectkbest')
        
    Returns:
        X_train_selected, X_test_selected: Выборки с отобранными признаками
        selected_features: Список выбранных признаков
    """
    print(f"Отбор признаков с использованием метода {method}...")
    
    # Установка значений по умолчанию
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if n_features is None:
        n_features = min(20, X_train.shape[1])
    
    # Отбор признаков различными методами
    if method == 'selectfrommodel':
        # Обучение модели
        estimator.fit(X_train, y_train)
        
        # Создание селектора
        selector = SelectFromModel(estimator, prefit=True)
        
    elif method == 'rfe':
        # Создание селектора
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X_train, y_train)
        
    elif method == 'rfecv':
        # Создание селектора с кросс-валидацией
        selector = RFECV(estimator=estimator, step=1, cv=5, scoring='f1')
        selector.fit(X_train, y_train)
        
    elif method == 'selectkbest':
        # Создание селектора на основе статистических тестов
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X_train, y_train)
        
    else:
        print(f"Неизвестный метод отбора признаков: {method}. Используется SelectFromModel.")
        estimator.fit(X_train, y_train)
        selector = SelectFromModel(estimator, prefit=True)
    
    # Применение отбора
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Получение имен выбранных признаков
    if hasattr(selector, 'get_support'):
        selected_features = X_train.columns[selector.get_support()].tolist()
    else:
        # Для SelectFromModel
        selected_features = X_train.columns[selector.get_support()].tolist()
    
    print(f"Отобрано {len(selected_features)} признаков из {X_train.shape[1]}")
    print(f"Новые размерности: X_train {X_train_selected.shape}, X_test {X_test_selected.shape}")
    
    # Преобразование результатов обратно в DataFrame с правильными именами столбцов
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train_selected_df, X_test_selected_df, selected_features

def optimize_xgboost(X_train, y_train, X_val=None, y_val=None, cv=5):
    """
    Оптимизация гиперпараметров модели XGBoost
    
    Args:
        X_train: Признаки тренировочной выборки
        y_train: Целевая переменная тренировочной выборки
        X_val: Признаки валидационной выборки (опционально)
        y_val: Целевая переменная валидационной выборки (опционально)
        cv: Количество фолдов для кросс-валидации
        
    Returns:
        best_model: Оптимизированная модель XGBoost
        best_params: Лучшие параметры
    """
    print("\n" + "="*50)
    print("ОПТИМИЗАЦИЯ XGBOOST")
    print("="*50)
    
    # Если валидационная выборка не предоставлена, используем кросс-валидацию
    use_cv = X_val is None or y_val is None
    
    # Определение пространства поиска параметров
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    # Создание базовой модели XGBoost
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Балансировка классов
    )
    
    # Оптимизация с использованием RandomizedSearchCV
    print("Запуск RandomizedSearchCV для оптимизации гиперпараметров XGBoost...")
    
    if use_cv:
        # Использование кросс-валидации
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=20,  # Количество комбинаций параметров для проверки
            scoring='f1',  # Метрика для оптимизации
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Запуск поиска
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        print(f"Поиск завершен за {search_time:.2f} секунд")
        print(f"Лучшие параметры: {search.best_params_}")
        print(f"Лучшая F1-мера (CV): {search.best_score_:.4f}")
        
        # Обучение модели с лучшими параметрами
        best_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            **search.best_params_
        )
        
        best_model.fit(X_train, y_train)
        best_params = search.best_params_
        
    else:
        # Использование отдельной валидационной выборки
        best_score = 0
        best_params = None
        best_model = None
        
        # Определяем набор параметров для перебора
        n_estimators_list = param_grid['n_estimators']
        max_depth_list = param_grid['max_depth']
        learning_rate_list = param_grid['learning_rate']
        subsample_list = param_grid['subsample']
        colsample_bytree_list = param_grid['colsample_bytree']
        
        # Общее количество комбинаций
        total_combinations = (
            len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list) * 
            len(subsample_list) * len(colsample_bytree_list)
        )
        
        # Ограничиваем число комбинаций
        n_combinations = min(20, total_combinations)
        
        # Случайный выбор комбинаций
        np.random.seed(42)
        combinations_indices = np.random.choice(total_combinations, size=n_combinations, replace=False)
        
        print(f"Перебор {n_combinations} комбинаций параметров...")
        
        # Для каждой комбинации параметров
        for i, idx in enumerate(combinations_indices):
            # Определение параметров
            n_estimators = n_estimators_list[idx % len(n_estimators_list)]
            max_depth = max_depth_list[(idx // len(n_estimators_list)) % len(max_depth_list)]
            learning_rate = learning_rate_list[(idx // (len(n_estimators_list) * len(max_depth_list))) % len(learning_rate_list)]
            subsample = subsample_list[(idx // (len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list))) % len(subsample_list)]
            colsample_bytree = colsample_bytree_list[(idx // (len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list) * len(subsample_list))) % len(colsample_bytree_list)]
            
            # Создание модели
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
            )
            
            # Обучение модели
            print(f"\nКомбинация {i+1}/{n_combinations}:")
            print(f"n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}, colsample_bytree={colsample_bytree}")
            
            model.fit(X_train, y_train)
            
            # Оценка на валидационной выборке
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            
            print(f"F1-мера: {f1:.4f}")
            
            # Сохранение лучшей модели
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree
                }
                
        print(f"\nЛучшие параметры: {best_params}")
        print(f"Лучшая F1-мера: {best_score:.4f}")
    
    # Сохранение оптимизированной модели
    model_path = MODELS_DIR / OPTIMIZED_XGB_MODEL_FILE
    joblib.dump(best_model, model_path)
    print(f"Оптимизированная модель XGBoost сохранена в {model_path}")
    
    return best_model, best_params

def optimize_logistic_regression(X_train, y_train, X_val=None, y_val=None, cv=5):
    """
    Оптимизация гиперпараметров модели логистической регрессии
    
    Args:
        X_train: Признаки тренировочной выборки
        y_train: Целевая переменная тренировочной выборки
        X_val: Признаки валидационной выборки (опционально)
        y_val: Целевая переменная валидационной выборки (опционально)
        cv: Количество фолдов для кросс-валидации
        
    Returns:
        best_model: Оптимизированная модель логистической регрессии
        best_params: Лучшие параметры
    """
    print("\n" + "="*50)
    print("ОПТИМИЗАЦИЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
    print("="*50)
    
    # Если валидационная выборка не предоставлена, используем кросс-валидацию
    use_cv = X_val is None or y_val is None
    
    # Определение пространства поиска параметров
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'class_weight': ['balanced', None]
    }
    
    # Ограничения на комбинации параметров (не все солверы поддерживают все типы регуляризации)
    # - l1 поддерживается только 'liblinear' и 'saga'
    # - elasticnet поддерживается только 'saga'
    # - None (penalty) поддерживается 'newton-cg', 'lbfgs', 'sag'
    
    # Создание базовой модели логистической регрессии
    base_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    # Оптимизация с использованием GridSearchCV
    print("Запуск GridSearchCV для оптимизации гиперпараметров логистической регрессии...")
    
    # Корректировка сетки параметров для совместимых комбинаций
    valid_param_combinations = []
    
    for penalty in param_grid['penalty']:
        for solver in param_grid['solver']:
            for C in param_grid['C']:
                for class_weight in param_grid['class_weight']:
                    # Проверка совместимости penalty и solver
                    if (penalty == 'l1' and solver in ['liblinear', 'saga']) or \
                       (penalty == 'l2' and solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']) or \
                       (penalty == 'elasticnet' and solver == 'saga') or \
                       (penalty is None and solver in ['newton-cg', 'lbfgs', 'sag']):
                        valid_param_combinations.append({
                            'C': C,
                            'penalty': penalty,
                            'solver': solver,
                            'class_weight': class_weight
                        })
    
    print(f"Проверка {len(valid_param_combinations)} корректных комбинаций параметров")
    
    if use_cv:
        # Использование кросс-валидации
        
        # Создание объекта GridSearchCV
        search = GridSearchCV(
            estimator=base_model,
            param_grid=valid_param_combinations,
            scoring='roc_auc',  # Оптимизация по ROC-AUC
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        # Запуск поиска
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        print(f"Поиск завершен за {search_time:.2f} секунд")
        print(f"Лучшие параметры: {search.best_params_}")
        print(f"Лучший ROC-AUC (CV): {search.best_score_:.4f}")
        
        # Обучение модели с лучшими параметрами
        best_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            **search.best_params_
        )
        
        best_model.fit(X_train, y_train)
        best_params = search.best_params_
        
    else:
        # Использование отдельной валидационной выборки
        best_score = 0
        best_params = None
        best_model = None
        
        print(f"Перебор {len(valid_param_combinations)} корректных комбинаций параметров...")
        
        # Для каждой комбинации параметров
        for i, params in enumerate(valid_param_combinations):
            # Создание модели
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **params
            )
            
            # Обучение модели
            print(f"\nКомбинация {i+1}/{len(valid_param_combinations)}:")
            print(f"C={params['C']}, penalty={params['penalty']}, solver={params['solver']}, class_weight={params['class_weight']}")
            
            model.fit(X_train, y_train)
            
            # Оценка на валидационной выборке
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
            
            print(f"ROC-AUC: {roc_auc:.4f}")
            
            # Сохранение лучшей модели
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_params = params
                
        print(f"\nЛучшие параметры: {best_params}")
        print(f"Лучший ROC-AUC: {best_score:.4f}")
    
    # Сохранение оптимизированной модели
    model_path = MODELS_DIR / OPTIMIZED_LR_MODEL_FILE
    joblib.dump(best_model, model_path)
    print(f"Оптимизированная модель логистической регрессии сохранена в {model_path}")
    
    return best_model, best_params 

def create_ensemble_model(models, X_train, y_train, method='voting'):
    """
    Создание ансамблевой модели из набора базовых моделей
    
    Args:
        models: Список кортежей (имя_модели, модель)
        X_train: Признаки тренировочной выборки
        y_train: Целевая переменная тренировочной выборки
        method: Метод создания ансамбля ('voting', 'stacking')
        
    Returns:
        ensemble_model: Обученная ансамблевая модель
    """
    print("\n" + "="*50)
    print(f"СОЗДАНИЕ АНСАМБЛЕВОЙ МОДЕЛИ ({method.upper()})")
    print("="*50)
    
    if method == 'voting':
        # Создание ансамбля с помощью голосования
        ensemble_model = VotingClassifier(
            estimators=models,
            voting='soft',  # Использование вероятностей (мягкое голосование)
            n_jobs=-1
        )
    elif method == 'stacking':
        # Создание ансамбля с помощью стекинга
        # Использование логистической регрессии в качестве метамодели
        ensemble_model = StackingClassifier(
            estimators=models,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            n_jobs=-1
        )
    else:
        print(f"Неизвестный метод ансамблирования: {method}. Используется голосование.")
        ensemble_model = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
    
    # Обучение ансамблевой модели
    start_time = time.time()
    ensemble_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Время обучения ансамблевой модели: {train_time:.2f} секунд")
    
    # Сохранение ансамблевой модели
    model_path = MODELS_DIR / ENSEMBLE_MODEL_FILE
    joblib.dump(ensemble_model, model_path)
    print(f"Ансамблевая модель сохранена в {model_path}")
    
    return ensemble_model

def plot_roc_curves(models_probs, y_test):
    """
    Построение ROC-кривых для нескольких моделей
    
    Args:
        models_probs: Словарь {название_модели: вероятности_предсказаний}
        y_test: Истинные значения целевой переменной
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, y_prob in models_probs.items():
        # Расчет ROC-кривой
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Построение ROC-кривой
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Диагональная линия (случайная модель)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Настройка графика
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Сохранение графика
    plt.savefig(FIGURES_DIR / "optimized_roc_curves.png")
    plt.close()
    
    print(f"ROC-кривые сохранены в {FIGURES_DIR / 'optimized_roc_curves.png'}")

def plot_precision_recall_curves(models_probs, y_test):
    """
    Построение кривых точности-полноты для нескольких моделей
    
    Args:
        models_probs: Словарь {название_модели: вероятности_предсказаний}
        y_test: Истинные значения целевой переменной
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, y_prob in models_probs.items():
        # Расчет кривой точности-полноты
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        # Построение кривой
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.4f})')
    
    # Настройка графика
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    # Сохранение графика
    plt.savefig(FIGURES_DIR / "optimized_precision_recall_curves.png")
    plt.close()
    
    print(f"Кривые точности-полноты сохранены в {FIGURES_DIR / 'optimized_precision_recall_curves.png'}")

def compare_optimized_models(metrics_list):
    """
    Сравнение оптимизированных моделей по различным метрикам
    
    Args:
        metrics_list: Список словарей с метриками моделей
        
    Returns:
        best_model_name: Название лучшей модели
    """
    print("\n" + "="*50)
    print("СРАВНЕНИЕ ОПТИМИЗИРОВАННЫХ МОДЕЛЕЙ")
    print("="*50)
    
    # Создание DataFrame с метриками всех моделей
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.set_index('model_name')
    
    print("\nСравнение метрик моделей:")
    print(df_metrics[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])
    
    # Визуализация метрик
    plt.figure(figsize=(14, 8))
    
    # Преобразование данных для удобной визуализации
    df_plot = df_metrics[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].reset_index()
    df_plot_melted = pd.melt(df_plot, id_vars='model_name', var_name='Метрика', value_name='Значение')
    
    # Построение графика
    sns.barplot(x='model_name', y='Значение', hue='Метрика', data=df_plot_melted)
    plt.title('Сравнение метрик качества оптимизированных моделей')
    plt.ylabel('Значение метрики')
    plt.xlabel('Модель')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optimized_models_comparison.png")
    plt.close()
    
    # Определение лучшей модели по F1-мере
    best_model_f1 = df_metrics['f1_score'].idxmax()
    print(f"\nЛучшая модель по F1-мере: {best_model_f1} (F1-score: {df_metrics.loc[best_model_f1, 'f1_score']:.4f})")
    
    # Определение лучшей модели по ROC-AUC
    best_model_auc = df_metrics['roc_auc'].idxmax()
    print(f"Лучшая модель по ROC-AUC: {best_model_auc} (ROC-AUC: {df_metrics.loc[best_model_auc, 'roc_auc']:.4f})")
    
    # Сохранение сравнительной таблицы
    df_metrics.to_csv(FIGURES_DIR / "optimized_models_metrics.csv")
    print(f"Таблица метрик сохранена в {FIGURES_DIR / 'optimized_models_metrics.csv'}")
    
    # Определение лучшей модели по комбинации метрик
    # Можно использовать разные подходы к выбору лучшей модели
    # Например, взвешенную сумму метрик или модель с наилучшим F1
    best_model_name = best_model_f1
    
    return best_model_name

def main():
    """Основная функция для запуска оптимизации моделей"""
    print("Начало оптимизации моделей для прогнозирования оттока клиентов...")
    
    # Создание директорий для сохранения результатов
    create_directories()
    
    # Загрузка данных
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None:
        print("Ошибка при загрузке данных. Выход.")
        sys.exit(1)
    
    # Загрузка базовых моделей
    lr_model, xgb_model = load_models()
    if lr_model is None or xgb_model is None:
        print("Ошибка при загрузке моделей. Выход.")
        sys.exit(1)
    
    # 1. Балансировка классов
    print("\n" + "="*50)
    print("УЛУЧШЕНИЕ БАЛАНСА КЛАССОВ")
    print("="*50)
    
    # Применение SMOTE для балансировки классов
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, method='smote')
    
    # 2. Создание новых признаков
    print("\n" + "="*50)
    print("СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ")
    print("="*50)
    
    # Создание признаков-взаимодействий
    X_train_new, X_test_new = create_interaction_features(X_train_balanced, X_test)
    
    # Создание полиномиальных признаков (опционально)
    # X_train_new, X_test_new = create_polynomial_features(X_train_new, X_test_new, degree=2)
    
    # 3. Отбор важных признаков
    print("\n" + "="*50)
    print("ОТБОР ВАЖНЫХ ПРИЗНАКОВ")
    print("="*50)
    
    # Отбор признаков с использованием RandomForest
    X_train_selected, X_test_selected, selected_features = select_features(
        X_train_new, y_train_balanced, X_test_new, method='selectfrommodel'
    )
    
    # 4. Разделение данных на тренировочную и валидационную выборки
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_selected, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
    )
    
    print(f"Размеры выборок после предобработки:")
    print(f"X_train_final: {X_train_final.shape}, y_train_final: {y_train_final.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test_selected: {X_test_selected.shape}, y_test: {y_test.shape}")
    
    # 5. Оптимизация моделей
    
    # Оптимизация XGBoost
    optimized_xgb, xgb_best_params = optimize_xgboost(
        X_train_final, y_train_final, X_val, y_val
    )
    
    # Оптимизация логистической регрессии
    optimized_lr, lr_best_params = optimize_logistic_regression(
        X_train_final, y_train_final, X_val, y_val
    )
    
    # 6. Создание ансамблевой модели
    models = [
        ('lr', optimized_lr),
        ('xgb', optimized_xgb)
    ]
    
    ensemble_model = create_ensemble_model(
        models, X_train_selected, y_train_balanced, method='voting'
    )
    
    # 7. Оценка моделей на тестовой выборке
    print("\n" + "="*50)
    print("ОЦЕНКА ОПТИМИЗИРОВАННЫХ МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*50)
    
    # Оценка оптимизированного XGBoost
    y_pred_xgb = optimized_xgb.predict(X_test_selected)
    y_prob_xgb = optimized_xgb.predict_proba(X_test_selected)[:, 1]
    xgb_metrics = evaluate_model(y_test, y_pred_xgb, y_prob_xgb, "Оптимизированный XGBoost")
    xgb_metrics['model_name'] = 'Оптимизированный XGBoost'
    
    # Оценка оптимизированной логистической регрессии
    y_pred_lr = optimized_lr.predict(X_test_selected)
    y_prob_lr = optimized_lr.predict_proba(X_test_selected)[:, 1]
    lr_metrics = evaluate_model(y_test, y_pred_lr, y_prob_lr, "Оптимизированная логистическая регрессия")
    lr_metrics['model_name'] = 'Оптимизированная логистическая регрессия'
    
    # Оценка ансамблевой модели
    y_pred_ensemble = ensemble_model.predict(X_test_selected)
    y_prob_ensemble = ensemble_model.predict_proba(X_test_selected)[:, 1]
    ensemble_metrics = evaluate_model(y_test, y_pred_ensemble, y_prob_ensemble, "Ансамблевая модель")
    ensemble_metrics['model_name'] = 'Ансамблевая модель'
    
    # 8. Визуализация результатов
    # Построение ROC-кривых
    models_probs = {
        'Оптимизированный XGBoost': y_prob_xgb,
        'Оптимизированная логистическая регрессия': y_prob_lr,
        'Ансамблевая модель': y_prob_ensemble
    }
    
    plot_roc_curves(models_probs, y_test)
    plot_precision_recall_curves(models_probs, y_test)
    
    # 9. Сравнение оптимизированных моделей
    models_metrics = [xgb_metrics, lr_metrics, ensemble_metrics]
    best_model_name = compare_optimized_models(models_metrics)
    
    # 10. Сохранение лучшей модели
    print("\n" + "="*50)
    print("ФИНАЛИЗАЦИЯ ЛУЧШЕЙ МОДЕЛИ")
    print("="*50)
    
    # Определение лучшей модели
    if best_model_name == 'Оптимизированный XGBoost':
        best_model = optimized_xgb
    elif best_model_name == 'Оптимизированная логистическая регрессия':
        best_model = optimized_lr
    else:  # Ансамблевая модель
        best_model = ensemble_model
    
    # Сохранение лучшей модели
    best_model_path = MODELS_DIR / BEST_MODEL_FILE
    joblib.dump(best_model, best_model_path)
    print(f"Лучшая модель ({best_model_name}) сохранена в {best_model_path}")
    
    # Сохранение списка отобранных признаков
    with open(MODELS_DIR / "selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print("\nОптимизация моделей успешно завершена.")
    print(f"Лучшая модель: {best_model_name}")
    print(f"Метрики лучшей модели:")
    
    if best_model_name == 'Оптимизированный XGBoost':
        print(f"F1-score: {xgb_metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    elif best_model_name == 'Оптимизированная логистическая регрессия':
        print(f"F1-score: {lr_metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    else:  # Ансамблевая модель
        print(f"F1-score: {ensemble_metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
    
    print(f"Визуализации сохранены в директории {FIGURES_DIR}")
    print(f"Модели сохранены в директории {MODELS_DIR}")

if __name__ == "__main__":
    main() 