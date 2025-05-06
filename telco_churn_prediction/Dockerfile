FROM python:3.9-slim

WORKDIR /app

# Копирование только файла с зависимостями для кэширования слоя установки
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода приложения
COPY app/ ./app/
COPY models/ ./models/

# Порт, который будет слушать приложение
EXPOSE 5000

# Переменные среды
ENV FLASK_APP=app/app.py
ENV PYTHONPATH=/app

# Команда запуска приложения с правильными настройками хоста и порта
CMD ["python", "app/app.py"] 