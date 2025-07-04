# Основной отчет по работе находится в проекте в pdf формате "Отчет по работе и наблюдения"
  
# Воспроизвести проект.
- Поставить зависимости
- Запустить файл train.py
# Разбор эксперимента с YOLOv11

## 1. Подготовка данных

### Разбивка видео
- Исходное видео: 30 FPS, 
- Интервал выборки: каждые 15 кадров 
- Всего извлечено: 600 кадров

### Разметка (аннотирование)
- Классы: 13 классов
- Время разметки: 1 час

## 2. Аугментация данных

Применяли 7 типа преобразований:
1. Геометрические:
   - Поворот:
   - Отражение: 
2. Цветовые:
   - Яркость: 
   - Контраст: 
3. Размытие: 
4. Преобразования размера: 

Результат:
- Увеличение датасета

## 3. Обучение моделей

## 4. Результаты

### Метрики качества
Средние значения
**YOLOv11s**:
- Precision: 0.99
- Recall: 0.54
- F1-score: 0.76

**YOLOv11n**:
- Precision: 0.97  
- Recall: 0.54
- F1-score: 0.77

### Примеры ошибок
1. **Ложные срабатывания** :
   
2. **Пропуски** :
   
3. **Неточные рамки**:

## Выводы и рекомендации

1. Для точности:
   - Выбирать YOLOv11s (mAP 0.78)
   - Увеличить датасет сложных случаев
   - Добавить аугментаций с тенями

2. Для скорости:
   - YOLOv11n (85 FPS)
   - Уменьшить размер изображения до 480x480
   - Использовать TensorRT для оптимизации

3. Общие улучшения:
   - Добавить больше примеров перекрытий
   - Разметить мелкие объекты отдельным классом
   - Попробовать другие аугментации
