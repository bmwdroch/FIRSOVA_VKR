import os
import re
from pathlib import Path

# Порядок файлов для главы 2
chapter2_files = [
    'docs/chapters/chapter2/2.0_introduction.md',
    'docs/chapters/chapter2/2.1_ml_overview.md',
    'docs/chapters/chapter2/2.2_classification_algorithms.md',
    'docs/chapters/chapter2/2.3_model_evaluation.md',
    'docs/chapters/chapter2/2.4_model_interpretability.md',
    'docs/chapters/chapter2/2.5_conclusion.md'
]

# Порядок файлов для главы 3
chapter3_files = [
    'docs/chapters/chapter3/3.0_introduction.md',
    'docs/chapters/chapter3/3.1_exploratory_analysis.md',
    'docs/chapters/chapter3/3.2_data_preprocessing.md',
    'docs/chapters/chapter3/3.3_model_development.md',
    'docs/chapters/chapter3/3.4_model_optimization.md',
    'docs/chapters/chapter3/3.5_model_comparison.md',
    'docs/chapters/chapter3/3.6_web_application.md',
    'docs/chapters/chapter3/3.7_conclusion.md'
]

# Функция для обработки пути к изображению
def process_image_path(match):
    # Извлекаем описание и путь к изображению
    desc = match.group(1)
    path = match.group(2)
    
    # Преобразуем относительный путь в абсолютный для финального документа
    # Заменяем ../../images/ на images/
    new_path = path.replace("../../images/", "images/")
    
    # Корректировка пути для изображений на кириллице
    if 'случайный_лес' in new_path:
        new_path = new_path.replace('случайный_лес', 'random_forest')
    elif 'дерево_решений' in new_path:
        new_path = new_path.replace('дерево_решений', 'decision_tree')
    
    # Возвращаем ссылку с правильным путем
    return f'![{desc}]({new_path})'

# Функция для компоновки глав
def compile_chapter(files, output_file):
    content = ''
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # Обрабатываем пути к изображениям
                file_content = re.sub(r'!\[(.*?)\]\((.*?)\)', process_image_path, file_content)
                
                content += file_content + '\n\n'
        else:
            print(f'Файл не найден: {file_path}')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

# Создаем директорию output, если ее нет
os.makedirs('output', exist_ok=True)

# Компилируем главы
compile_chapter(chapter2_files, 'output/chapter2.md')
compile_chapter(chapter3_files, 'output/chapter3.md')

# Объединяем главы
with open('output/chapters2_3.md', 'w', encoding='utf-8') as f:
    with open('output/chapter2.md', 'r', encoding='utf-8') as ch2:
        f.write(ch2.read())
    f.write('\n\n')
    with open('output/chapter3.md', 'r', encoding='utf-8') as ch3:
        f.write(ch3.read())

print('Главы успешно объединены в файл output/chapters2_3.md')
print('Также доступны отдельные файлы для глав:')
print('- output/chapter2.md')
print('- output/chapter3.md') 