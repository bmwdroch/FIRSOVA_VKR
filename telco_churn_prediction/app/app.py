#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной модуль Flask-приложения для прогнозирования оттока клиентов телекоммуникационной компании.
Предоставляет веб-интерфейс для ввода данных о клиенте и получения прогноза.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Добавление корневой директории проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Импорт модуля для работы с моделью
sys.path.append(str(Path(__file__).resolve().parent))
from utils.model_utils import load_model_and_preprocessor, predict_churn

# Создание экземпляра Flask
app = Flask(__name__)

# Загрузка модели и препроцессора при запуске приложения
model, preprocessor = load_model_and_preprocessor()

@app.route('/')
def index():
    """
    Главная страница приложения с формой для ввода данных клиента.
    """
    # Словари для полей формы
    gender_options = ['Male', 'Female']
    yes_no_options = ['Yes', 'No']
    multiple_lines_options = ['No', 'Yes', 'No phone service']
    internet_service_options = ['DSL', 'Fiber optic', 'No']
    internet_service_addons_options = ['No', 'Yes', 'No internet service']
    contract_options = ['Month-to-month', 'One year', 'Two year']
    payment_method_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    
    return render_template('index.html', 
                           gender_options=gender_options,
                           yes_no_options=yes_no_options,
                           multiple_lines_options=multiple_lines_options,
                           internet_service_options=internet_service_options,
                           internet_service_addons_options=internet_service_addons_options,
                           contract_options=contract_options,
                           payment_method_options=payment_method_options)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Обработка данных формы и возвращение результата прогноза.
    """
    try:
        # Получение данных из формы
        data = {
            'gender': request.form['gender'],
            'SeniorCitizen': 1 if request.form['SeniorCitizen'] == 'Yes' else 0,
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }
        
        # Прогноз оттока клиента
        prediction, probability = predict_churn(data, model, preprocessor)
        
        # Подготовка результата
        churn_status = "Да, клиент уйдет" if prediction == 1 else "Нет, клиент не уйдет"
        churn_probability = f"{probability:.2%}"
        
        # Возвращение результата
        return render_template('result.html',
                              prediction=churn_status,
                              probability=churn_probability,
                              client_data=data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API-эндпоинт для получения прогноза в формате JSON.
    """
    try:
        # Получение данных из JSON запроса
        data = request.json
        
        # Преобразование SeniorCitizen в числовой формат, если он передан как строка
        if 'SeniorCitizen' in data and isinstance(data['SeniorCitizen'], str):
            data['SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
        
        # Прогноз оттока клиента
        prediction, probability = predict_churn(data, model, preprocessor)
        
        # Возвращение результата в формате JSON
        return jsonify({
            'prediction': int(prediction),
            'churn_status': "Yes" if prediction == 1 else "No",
            'probability': float(probability),
            'probability_percent': f"{probability:.2%}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 