# Домашнее задание
Оптимизация работы с Olivetti

Цель:
Цель этого домашнего задания попробовать использование разных классификаторов в качестве Feature Matching.

Описание/Пошаговая инструкция выполнения домашнего задания:
- Взять код из занятия.
- Добавить свою Сеть/Модель в конец.
- Обучить свой классификатор на PCA датасете

# Подход к решению задачи.
В рамках решения задачи я воспользуюсь тем же датасетом c фотографиями 31 селебрити, что и в задаче на использование [лицевых меток](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset).

![image](https://github.com/user-attachments/assets/07b3e31e-a419-44c4-a21a-e9b24f5f2427)


Основная задача - это классиифкация лиц, используемые метрики качества - Accuracy и F1.

Задача классификации лиц включает основные этапы:
- детекция лиц
- выраванивание и нормализация
- поиск в базе данных

Решать задачу буду в несколько подходов:
1. Использование готовой библиотеки для классификации лиц. Воспользуюсь библиотекой [DeepFace](https://github.com/serengil/deepface/tree/master) В этой библиотеке реализован весь пайплайн для классификации лиц.
2. Создание векторных представлений изображений с помощью алгоритма кластеризации/уменьшения размерности PCA и использование данных представлений классическими алгоритмами ML и небольщой полносвязной сетью.
3. Создание векторных представлений из модели ResNet с помощью обучения триплетов на Contrastive Loss.

## Сводные результаты экспериментов.

| Алгоритм | Accuracy   | F1    |Ссылка    |
| :---:   | :---: | :---: |:---: |
| Facial Landmarks as Features, CatBoost| 41.05%   | 37.5%| [ноутбук](https://github.com/shakhovak/CV_OTUS_course/blob/master/HW8/HW7_Face_Landmarks_Detectorv4.ipynb)|
| DeepFace, VGG-Face  | 84.6%   |85.97% |[ноутбук](https://github.com/shakhovak/CV_OTUS_course/blob/master/HW8/DeepFace_class.ipynb)|
| PCA, CatBoost  | 61.4%  | 60.58%  |[ноутбук](https://github.com/shakhovak/CV_OTUS_course/blob/master/HW8/HW8_Face_recognition.ipynb)|
| PCA, Dense Net  | 56.73%  | -  |[ноутбук](https://github.com/shakhovak/CV_OTUS_course/blob/master/HW8/HW8_Face_recognition.ipynb)|
