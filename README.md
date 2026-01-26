# OpenWebUI Spreadsheet Analyst

Цей репозиторій додає до OpenWebUI можливість аналізувати табличні файли (CSV/XLSX/TSV/JSON/Parquet) через пайплайн і виконувати pandas‑код у безпечному sandbox‑сервісі. Логіка розділена на три компоненти:

- OpenWebUI — інтерфейс і бекенд чату.
- pipelines — сервіс, який приймає OpenAI‑compatible запити та виконує пайплайни.
- sandbox_service — ізольований FastAPI сервіс для завантаження таблиць і виконання коду.

## Архітектура та потік даних

```
Користувач → OpenWebUI → pipelines → sandbox_service → pipelines → OpenWebUI
                    ↘ (Pipe) → pipelines або → зовнішній API
```

Основний сценарій: файл завантажується в OpenWebUI, pipelines витягує файл і надсилає у sandbox, а результат повертається у чат.

## Структура проєкту

```
.
├── config/
│   └── user_params.env
├── docker-compose.yml
├── README.md
├── sandbox_service/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── pipelines/
    ├── pipeline-requirements.txt
    ├── spreadsheet_analyst_pipeline.py
    ├── spreadsheet_analyst_pipeline/
    │   └── valves.json
    └── pipe/
        ├── pipe.py
        └── valves.json
```

## Налаштування

Всі змінні параметри зібрані в одному файлі: `config/user_params.env`. Там можна задати:

- URL‑и сервісів, API‑ключі, модель LLM.
- Ліміти пайплайна (кількість рядків, таймаути, розміри виводу).
- Ліміти sandbox (рядки, кеш, CPU/памʼять, розмір результату).
- Параметри Pipe та fallback‑API.

Рекомендовані значення для локального запуску:

- `PIPELINES_BASE_URL=http://pipelines:9099/v1`
- `SANDBOX_URL=http://sandbox:8081`
- `WEBUI_BASE_URL=http://openwebui:8080`

### Приклад підключення `user_params.env` у Docker Compose

Додайте `env_file` до сервісів `pipelines` і `sandbox`:

```yaml
services:
  sandbox:
    env_file:
      - ./config/user_params.env
  pipelines:
    env_file:
      - ./config/user_params.env
```

Після цього значення з `config/user_params.env` будуть використовуватись автоматично.

## Запуск через Docker Compose

```bash
docker compose up -d --build
```

Після запуску у OpenWebUI:
1) Admin → Settings → Connections.
2) Додати OpenAI‑compatible connection:
   - URL: `http://localhost:9099/v1`
   - API key: `0p3n-w3bu!`
3) Обрати модель `Spreadsheet Analyst`.

## Збірка з нуля (без кешу)

Увага: команда `--volumes` видалить дані OpenWebUI.

```bash
# зупинити і видалити контейнери та томи
docker compose down --volumes --remove-orphans

# (опційно) підтягнути свіжі базові образи
docker compose pull

# збірка без кешу
docker compose build --no-cache

# запустити з нуля
docker compose up -d --force-recreate
```

## Приклади запитів

- `Покажи топ‑10 продуктів за виручкою.`
- `Порахуй середній чек по місяцях і покажи тренд.`
- `Знайди 5 клієнтів з найбільшим обсягом покупок.`
- `Побудуй зведену таблицю: регіон × категорія з сумою продажів.`

## Поради

- Для великих файлів зменшіть `PIPELINE_PREVIEW_ROWS`, щоб пришвидшити відповідь.
- Якщо отримуєте таймаути — підвищіть `PIPELINE_CODE_TIMEOUT_S` або `CPU_TIME_S`.
- Якщо відповідь надто коротка — збільшіть `MAX_RESULT_CHARS` і `PIPELINE_MAX_STDOUT_CHARS`.
- Якщо не бачите файл у чаті — перевірте `WEBUI_BASE_URL` і `WEBUI_API_KEY`.

## Діагностика

- `Attach a CSV/XLSX file and ask a question.` — файл не прикріпився до запиту.
- `Failed to load the table in the sandbox.` — sandbox не зміг прочитати файл або перевищено ліміти.
- `Pipeline error` — перевірте доступність `BASE_LLM_BASE_URL` та `SANDBOX_URL`.

## Компоненти

### `pipelines/spreadsheet_analyst_pipeline.py`

Основний пайплайн:
- Витягує файл з OpenWebUI.
- Завантажує таблицю у sandbox.
- Генерує pandas‑код через LLM.
- Виконує код у sandbox і повертає відповідь українською.

Ключові параметри читаються з середовища (див. `config/user_params.env`).

### `pipelines/pipe/pipe.py`

Pipe‑проксі між OpenWebUI та pipelines (з fallback на зовнішній API). Параметри теж беруться з середовища.

### `sandbox_service/main.py`

Безпечне виконання pandas‑коду в ізольованому сервісі з лімітами CPU/памʼяті та обмеженнями API.

## Примітки

- `pipelines/*/valves.json` залишені як пусті, дефолти беруться з коду та/або env.
- Для зміни лімітів достатньо відредагувати `config/user_params.env` і перезапустити сервіси.
