# ggwt
## Как запустить: 
Установить Python 3.11.15

Перейти в папку проекта в командной строке.

Создать среду:
`python3.11 -m venv venv`

Активировать:
`venv\Scripts\activate`

Установить зависимости:
`pip install -r requirements.txt`

Запуск для предсказания выработки следующего периода с 01.01.2026 по 31.03.2026:
`python red.py --train_path train_dataset.csv --valid_path valid_features.csv --output_path submission_q1.csv`

Запуск только для предсказания выработки для 18 мая 2026:
`python red.py --train_path train_dataset.csv --valid_path test_dataset.csv --filter_date 2026-05-18 --output_path submission_may18.csv`
