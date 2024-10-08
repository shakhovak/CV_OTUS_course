# Матчинг товаров на основе изображений и/или текстовых описаний

## Схема матчинга
Матчинг товаров представляет собой процесс сопоставления объектов (карточек товаров) на основе сравнения и расчета некоторой меры схожести, где первый объект представляет запрос, а другой - базу данных с объектами (см. рис ниже).

![image](https://github.com/user-attachments/assets/c7002f41-b044-4c4c-b24d-1e946b238032)

В рамках данного проекта будут исследованы варианты матчинга до стадии выдачи TOP N похожих объектов, так как попарное сравнение уже требует дополнительной разметки.

## Задачи проекта
Данный проект носит исследовательский характер и решает следующие задачи:
- изучить варианты эмбедингов для отбора похожих товаров на основе карточек товаров
- сравнить текстовые и картиночные эмбединги, а также объединение этих модальностей
- проверить, улучшает ли качество эмбедингов fine-tune моделей, используемых для создания эмбедингов

## Данные
Для проекта были собраны карточки товаров включая изображение у 6 продавцов спортивного снаряжения как представлено на схеме ниже. Для сбора использовались парсеры, собранные вот в этой [папке](https://github.com/shakhovak/CV_OTUS_course/tree/master/Fin_project/parcers). В качестве данных компании (т.е. объектов, для которых будут находиться top N) используются карточки товаров SportDoma. Все остальные карточки относятся к базе данных, в рамках которой производится поиск матчей для карточек компании.

![image](https://github.com/user-attachments/assets/997640bc-0c19-4a1c-b418-67f41eea99f6)

Собранные данные содержат следующую информацию:

- текст
   - **title** - название товара, указанное продавцом
   - **price** - цена товара, указанная продавцов
   - **cat_1, cat_2, cat_3** - категории иерархии товаров, выбранные продавцом для магазина
   - **carachteristics** - описание характеристик товара (свободный текстовый формат)
   - **img_ref** - путь к файлу с изображением товара
   - **target** - категория товара, используемая для матчинга (в большинстве случаев совпадает с  cat_2 или cat_3)
   - **dealer** - краткое название продавца
- изображение - изображение товара из карточки (выбранное продавцом в качестве основного/первого). Примеры изображений по категориям на рис ниже - видно, что изображения в разных размерах.

![image](https://github.com/user-attachments/assets/3241469d-4107-4be8-b8cb-429bfbfe7efb)

Собранная информация была предобработана для создания финальных файлов, используемых в проекте (см. [папку](https://github.com/shakhovak/CV_OTUS_course/tree/master/Fin_project/data)): файл ```company_data.csv ``` (итого ~ 1тыс позиций) включает информацию о данных товаров Sportdoma, файл ```comparable_data.csv ``` (итого ~ 13,5 тыс. позиций) включает информацию о карточках товаров, используемыз в качестве базы данных.

Предообработка включает ([файл предобработки](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/data/data_analysis_fin.ipynb) включает объединение данных из парсеров в один и обработку сводного файла) :
- удаление строк, где не скачался файл с изображением товара
- удаление лишних символов в текстах и нормализация по unicode



## Модели и подходы
Для создания картиночных эмбедингов используется модель ResNet18 (11.7M параметров). Эмбедингы берутся из слоя, предшествующего классификационному. Для этого классификационный слой заменяется на слой Identity, выходом которого является вектор, длиной 312:
 ```python
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```
Для создания текстовых эмбедингов используется модель [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) (29,4М параметров) для русского языка. Это модель специально обучена для создания векторных представлений текста (т.н. sentence-transformer). В качестве эмбединга используется вектор текста из последнего скрытого слоя, т.е. аналогично resnet до классификационного полносвязного слоя. Выходом данного слоя является вектор, длиной 512.

В качестве основного алгоритма поиска будет использована библиотека [**ANNOY (Approximate Nearest Neighbors Oh Yeah)**](https://github.com/spotify/annoy), разработанная Spotify для приближенного поиска (ANN). Прямой поиск knn является очень требовательным к ресурсам и достаточно длителен по времени исполнения, так как требует перебора всех данных в базе (O(N)). В отличие от него ANN алгоритм разбивает векторное пространство базы данных на несколько регионов и создает их представление в виде бинарных представлений. При обработке запроса алгоритм находит необходимое дерево и в рамках векторов этого дерева проводит классический knn поиск по косинусной близости (O(logn)).

Интересно, что при размере базы ~14 тыс векторов и размера леса в 50 деревьев, топ 10 выдача классического knn по косинусной близости не отличается от аналогичной выдаче ann из ANNOY. Ниже пример поиска топ 10 по заголовку (title) для артикула:

![image](https://github.com/user-attachments/assets/37332b23-3c25-4744-92b8-d3801c92691f)

**Выдача алгоритма knn (перебор всей базы)**
![image](https://github.com/user-attachments/assets/62fe034c-56c1-4bc7-9c01-2ebd998184a4)

**Выдача алгоритма ann (50 деревьев)**
![image](https://github.com/user-attachments/assets/a7b050ac-2f21-4b82-bbe3-8cb6b8c24c2b)

При этом knn обрабатывает данные компании в 1 тыс. позиций примерно 30 мин для выдачи топ 10, ANNOY менее 1 сек (2500 it/s). Поэтому в экспериментах буду использовать алгоритм ann как основной, а knn как дополнитльный, если будет переранжирование выдачи алгоритма.
## Эксперименты
В рамках проекта буду оценивать результат выдачи топ 10 на основе различных вариантов эмбедингов. При этом, если товар включенный в топ 10 принадлежит к той же группе товаров, что и искомый, то данный вариант выдачи получает 1, если принадлежит к другой категории товаров, то получает 0. В итоге по каждому товару в тестовой выборке (выборка товаров компании) можно получить точность выдачи в виде Accuracy@10, а по всему алгоритму можно подсчитать среднее значение данного показателя.

Помимо общей метрики посмотрю выдачу алгоритма для позиций товаров с id 500,626,621,0 для ручной оценки с точки зрения выбора марки товара и ценовой категории. При этом допускаю, что при обучении на категории может ухудшиться в рамках поиска аналогичных моделей.

Первая часть экспериментов будет использовать готовые модели ("из коробки"), вторая часть - те же самые подходы, как и в первой части, но уже на обученных моделях.

**Эксперименты с моделями "из коробки" без обучения**

| Эксперимент | Описание  | Файл с ноутбуком   | Acc@10, %   |Визуальная оценка  |
| :---:   | :---: | :---: |:---: |:---: |
| img |Использование только картиночных эмбедингов | [emb_comparison_img_v1.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_img_v1.ipynb)|85,21|3 из 10, находит только по 1 артикулу той же модели|
| title |Использование только title для текстового эмбединга | [emb_comparison_text.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_text.ipynb)|73,21|6 из 10, большинство артикулов в выдаче не только одной категории, но и марки кроме нескольких категорий |
| title+cat |Использование title+cat для текстового эмбединга  | [emb_comparison_text.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_text.ipynb)|85,11|4 из 10, хуже стал находить марки|
| text_all |Использование всей текстовой инфо с карточки для текстового эмбедингов| [emb_comparison_text.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_text.ipynb)|63,14|4 из 10, находит только по 1 артикулу той же модели|
| img+title |Конкатенация картиночного и текстового векторов | [emb_comparative_combined.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparative_combined.ipynb)|85,18|3 из 10, очень похоже на выдачу только img модели, text не особо помог|
| img+title+cat |Конкатенация картиночного и текстового векторов | [emb_comparative_combined.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparative_combined.ipynb)|84,91|3 из 10, очень похоже на выдачу только img модели, text не особо помог|
| img40+title_rerank |Поиск 40 похожих по картиночному вектору, переранжирование по текстовому | [emb_comparison_rerank.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_rerank.ipynb)|86,77|5 из 10|
| img40+title+cat_rerank |Поиск 40 похожих по картиночному вектору, переранжирование по текстовому | [emb_comparison_rerank.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_rerank.ipynb)|90,36|5 из 10|
| title40+img_rerank |Поиск 40 похожих по текстовому вектору, переранжирование по картиночному | [emb_comparison_rerank2.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_rerank2.ipynb)|71,00|2 из 10, еще долго работает|

Вторая часть экспериментов связана с обучением моделей для эмбединга.
ResNet18 будет обучаться на создания кастомных картиночных эмбедингов с помощью Contrastive Loss и Cosine Similarity Loss ([ноутбук здесь](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/models_training/resnet_training.ipynb)) , а также просто на классификацию картинок на категории ([ноутбук здесь](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/models_training/resnet_class.ipynb)). TinyBert будет обучаться также на создания кастомных текстовых эмбедингов на Contrastive Loss (формула для расчета из библиотеки Sentence Transformers) только из названия товара - title ([ноутбук здесь](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/models_training/sentence_transformers2.ipynb)) и из названия + все категории ([ноутбук здесь](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/models_training/sentence_transformers3.ipynb))

**Используемый Contrastive Loss для обучения ResNet**
 ```python
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive
```

**Используемый CosineSimilarityLoss для обучения ResNet**

 ```python
class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output1, output2, label):
        cosine_sim = F.cosine_similarity(output1, output2)
        loss_fn = nn.MSELoss()
        loss_similarity = loss_fn(cosine_sim, label)
        return loss_similarity
```


**Эксперименты с дообученными моделями**

| Эксперимент | Описание  | Файл с ноутбуком   | Acc@10, %   |Визуальная оценка  |
| :---:   | :---: | :---: |:---: |:---: |
| img_train_ContLoss |Использование только картиночных эмбедингов, ResNet, обученная на Contrastive Loss | [emb_comparison_img.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_img.ipynb)|55,03|1 из 10, адекватная выдача только у дорожек :(|
| img_train_class |Использование только картиночных эмбедингов, ResNet, обученная на классификацию изображений на категорию товара | [emb_comparison_img_v3.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_img_v3.ipynb)|94,55|8 из 10, хотя стал меньше предлагать вариантов той же модели|
| title_train_v1 |Только текстовые эмбединги от tiny bert, обученного на Contrastive Loss на основе title | [emb_comarison_text2.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comarison_text2.ipynb)|87,03|8 из 10, хорошо также подбирает модели|
| title+cat_train_v1 |Только текстовые эмбединги от tiny bert, обученного на Contrastive Loss на основе title| [emb_comarison_text2.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comarison_text2.ipynb)|98,37|8 из 10, хорошо также подбирает модели|
| title_train_v2 |Только текстовые эмбединги от tiny bert, обученного на Contrastive Loss на основе title+cat | [emb_comarison_text_3.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comarison_text_3.ipynb)|85,54|8 из 10, хорошо также подбирает модели|
| title+cat_train_v2 |Только текстовые эмбединги от tiny bert, обученного на Contrastive Loss на основе title+cat| [emb_comarison_text_3.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comarison_text_3.ipynb)|99,53|8 из 10, хорошо также подбирает модели|
| img+title_train_v1 |Конкатенация картиночного и текстового векторов, ResNet из коробки, tinybert обучен на Contrastive Loss на title | [emb_comparative_combined_v2.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparative_combined_v2.ipynb)|85,19|3 из 10, обучение особо не помогло|
| img+title+cat_train_v1 |Конкатенация картиночного и текстового векторов, ResNet из коробки, tinybert обучен на Contrastive Loss на title | [emb_comparative_combined_v2.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparative_combined_v2.ipynb)|85,01|3 из 10, очень похоже на выдачу только img модели, text не особо помог|
| img40+title_rerank_train_v1 |Поиск 40 похожих по картиночному вектору, переранжирование по текстовому, ResNet из коробки, tinybert обучен на Contrastive Loss на title| [emb_comparison_rerank 3.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_rerank%203.ipynb)|90,26|6 из 10|
| img40+title+cat_rerank_train_v1 |Поиск 40 похожих по картиночному вектору, переранжирование по текстовому, ResNet из коробки, tinybert обучен на Contrastive Loss на title | [emb_comparison_rerank 3.ipynb](https://github.com/shakhovak/CV_OTUS_course/blob/master/Fin_project/experiments_notebooks/emb_comparison_rerank%203.ipynb)|93,45|6 из 10|

**Сводная диаграмма по всем экспериментам:**

![image](https://github.com/user-attachments/assets/dbdcc529-6ca3-4a9c-983e-a6aa7b81541e)



## Структура репозитория
```bash
│   README.md - отчет для ДЗ
│   models.py - файл со структурой для обучения модели ResNet
│   utils.py - вспомогательные функции 
|   
├───parcers
├───models_training 
├───experiment_notebooks -ноутбуки с экспериментами
├───data
│       comparable_data.csv - база данных 
│       сompany_data.csv - данные для запросов 
│       data_analysis_fin.ipynb - обработка данных
```
## Выводы

1. При использовании моделей из коробки лучший результат для выдачи топ 10 показали эмбединги из ResNet при переранжировании на текстовые вектора, состоящие из названия и категории.
2. Качество выдачи на эмбедингах из текста (название + категории), изображения и их конкатенации при использовании моделей из коробки не отличается.
3. Обучение на Contrastive Loss дало бустинг в метрике только для текстовой модели, для картинок - качество ухудшилось. Лучшая выдача для картиночных эмбедингов получается при обучении на простую классификацию.
4. Обучение дало следующие приросты в Acc@10:
   - img only -> +9%
   - text only (title + cat) -> +14%
   - concat ->+0%
   - rerank ->+3%
