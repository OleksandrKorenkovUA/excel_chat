# OpenWebUI Spreadsheet Analyst

## Зміст

1. [Концептуальний огляд](#концептуальний-огляд)
2. [Архітектура системи](#архітектура-системи)
3. [Встановлення та налаштування](#встановлення-та-налаштування)
4. [Використання](#використання)
5. [API документація](#api-документація)
6. [Розробка](#розробка)
7. [Приклади запитів](#приклади-запитів)
8. [Діагностика](#діагностика)

---

## Концептуальний огляд

### Проблема, яку вирішує система

Сучасні аналітичні завдання вимагають частої роботи з табличними даними (CSV, XLSX, TSV). Традиційні підходи вимагають:

- Експорту даних з CRM/ERP систем
- Імпорту у Excel або Python середовище
- Написання скриптів аналізу
- Повторення циклу для кожного нового запиту

**OpenWebUI Spreadsheet Analyst** вирішує ці проблеми шляхом:

1. **Надання інтерактивного інтерфейсу** — розмовний стиль взаємодії через чат
2. **Автоматична генерація коду** — переклад природної мови у pandas-код
3. **Безпечне виконання** — ізольоване середовище (sandbox) для виконання користувацького коду
4. **Кешування та швидкість** — повторне використання завантажених даних та шаблонних запитів

### Основні концепції

#### DataFrame як перший клас об'єктів

У цій системі `DataFrame` є основним об'єктом роботи:

- Завантажується один раз з файлу
- Зберігається у сесійному кеші
- Може бути модифікований операціями edit
- Історія змін підтримується для операцій undo

#### Двопроходова обробка запитів

1. **Швидкий шлях (shortcut)** — pattern matching для поширених запитів (min, max, groupby, head/tail)
2. **Повний шлях (LLM)** — генерація коду через мовну модель для унікальних запитів

#### Сесійна модель

Кожна сесія об'єднує:
- Файл користувача
- DataFrame у sandbox
- Історію запитів
- Проміжні результати

---

## Архітектура системи

### Компоненти системи

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Архітектура системи                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   OpenWebUI      │──────>│   Pipelines      │──────>│  Sandbox Service │
│  (FastAPI)       │      │  (FastAPI)       │      │  (FastAPI)       │
│                  │      │                  │      │                  │
│  - Web UI        │      │  - Pipeline      │      │  - DF Store      │
│  - Chat History  │      │    Orchestrator  │      │  - Code Runner   │
│  - File Upload   │      │  - Pipe Proxy    │      │  - Security      │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                        │                        │
         │                        │                        │
         │                  ┌─────┴──────┐                 │
         │                  │  Pipe      │                 │
         │                  │  Connector │                 │
         │                  └────────────┘                 │
```

### Сервіси

#### OpenWebUI

Головний інтерфейс системи. Надає:
- Веб-інтерфейс чату
- Завантаження файлів
- Історію повідомлень
- Відображення результатів

#### Pipelines (spreadsheet_analyst_pipeline.py)

Orchestrator системи. Відповідає за:
- Обробку запитів користувачів
- Координацію між сервісами
- Генерацію та валідацію коду
- Формування відповідей українською

#### Sandbox Service (sandbox_service/main.py)

Безпечне середовище для виконання коду. Надає:
- Взаємодію з DataFrame
- AST-валідацію коду
- Мультипроцесне виконання з лімітами
- Мутаційний跟踪 змін

---

## Встановлення та налаштування

### Передумови

- Docker 20.10+
- Docker Compose 2.0+
- 4+ GB оперативної пам'яті
- 2+ ядра CPU

### Базове встановлення

1. **Клонування репозиторію**

```bash
cd /Users/home/excel_chat-main
```

2. **Налаштування змінних середовища**

Створіть файл `config/user_params.env`:

```bash
# OpenWebUI
WEBUI_API_KEY=your-webui-api-key

# LLM API (для генерації коду)
BASE_LLM_API_KEY=your-llm-api-key
BASE_LLM_BASE_URL=http://your-llm-server:8031/v1
BASE_LLM_MODEL=chat-model

# Embeddings API (для шорткатів)
EMB_API_KEY=your-embed-api-key
EMB_BASE_URL=http://your-embed-server:8022/v1
EMB_MODEL=multilingual-embeddings

# Sandbox
SANDBOX_API_KEY=your-sandbox-api-key

# URLs (для Docker Compose)
WEBUI_BASE_URL=http://openwebui:8080
SANDBOX_URL=http://sandbox:8081
PIPELINES_BASE_URL=http://pipelines:9099/v1
```

3. **Запуск контейнерів**

```bash
docker compose up -d --build
```

4. **Перевірка статусу**

```bash
docker compose ps
docker compose logs -f
```

### Детальна конфігурація

#### Змінні OpenWebUI

| Змінна | Опис | Значення за замовч. |
|--------|------|---------------------|
| `WEBUI_API_KEY` | API-ключ для доступу до OpenWebUI | (опціонально) |
| `PIPELINES_API_KEY` | API-ключ для доступу до Pipelines | (опціонально) |

#### Змінні LLM

| Змінна | Опис | Значення за замовч. |
|--------|------|---------------------|
| `BASE_LLM_API_KEY` | API-ключ для LLM API | (обов'язково) |
| `BASE_LLM_BASE_URL` | URL LLM API | `http://alph-gpu.silly.billy:8031/v1` |
| `BASE_LLM_MODEL` | ID моделі LLM | `chat-model` |

#### Змінні Embeddings

| Змінна | Опис | Значення за замовч. |
|--------|------|---------------------|
| `EMB_API_KEY` | API-ключ для embeddings | (опціонально) |
| `EMB_BASE_URL` | URL embeddings API | `http://alph-gpu.silly.billy:8022/v1` |
| `EMB_MODEL` | Модель embeddings | `multilingual-embeddings` |

#### Змінні Sandbox

| Змінна | Опис | Значення за замовч. |
|--------|------|---------------------|
| `SANDBOX_API_KEY` | API-ключ для sandbox | (опціонально) |
| `MAX_ROWS` | Макс. рядків у DataFrame | `200000` |
| `PREVIEW_ROWS` | Рядків у прев'ю | `200000` |
| `MAX_CELL_CHARS` | Макс. довжина клітинки | `200` |
| `MAX_STDOUT_CHARS` | Макс. stdout | `8000` |
| `MAX_RESULT_CHARS` | Макс. розмір результату | `20000` |
| `CPU_TIME_S` | Таймаут CPU | `120` |
| `MAX_MEMORY_MB` | Ліміт пам'яті | `1024` |

#### Змінні Pipeline

| Змінна | Опис | Значення за замовч. |
|--------|------|---------------------|
| `PIPELINE_ID` | ID пайплайну | `spreadsheet-analyst` |
| `PIPELINE_NAME` | Назва пайплайну | `Spreadsheet Analyst` |
| `PIPELINE_DEBUG` | Режим налагодження | `false` |
| `PIPELINE_MAX_ROWS` | Макс. рядків | `200000` |
| `PIPELINE_PREVIEW_ROWS` | Рядків прев'ю | `200000` |
| `PIPELINE_SESSION_CACHE_TTL_S` | TTL кешу (сек) | `1800` |
| `PIPELINE_CODE_TIMEOUT_S` | Таймаут коду (сек) | `120` |
| `SHORTCUT_ENABLED` | Увімкнення шорткатів | `true` |
| `SHORTCUT_TOP_K` | K результатів retrieval | `5` |
| `SHORTCUT_THRESHOLD` | Поріг схожості | `0.35` |

---

## Використання

### Налаштування в OpenWebUI

1. Перейдіть до **Admin → Settings → Connections**

2. Додайте нове **OpenAI-compatible connection**:
   - **URL**: `http://localhost:9099/v1`
   - **API Key**: `0p3n-w3bu!` (або ваш `PIPELINES_API_KEY`)
   - **Model Name**: `Spreadsheet Analyst`

3. Збережіть налаштування

### Завантаження файлу

1. У чаті натисніть кнопку **Attach files** (або `+`)
2. Оберіть файл: **CSV, XLSX, TSV, JSON, Parquet**
3. Натисніть **Send**

### Формулювання запитів

#### Прості статистичні запити

```
Скільки рядків у таблиці?
Покажи перші 10 рядків.
Знайди мінімальне значення в колонці price.
Порахуй середнє по колонці quantity.
```

#### Групування

```
Покажи топ-10 продуктів за виручкою.
Скільки продажів по кожній категорії?
Знайди середній чек по місяцях.
```

#### Фільтрація

```
Покажи тільки продукти з категорії Electronics.
Знайди клієнтів з сумою замовлення > 1000.
Покажи записи за останній місяць.
```

#### Редагування даних

```
Видали рядок 5.
Додай новий рядок: product_id=100, name=Test.
Зміни значення в рядку 10, колонці price на 99.99.
```

### Структура запиту

**Мінімальний запит:**
```
Запит + прикріплений файл
```

**Запит з контекстом:**
```
У файлі orders.csv знайди топ-5 клієнтів за витратами.
```

**Спеціальні конструкції:**
```
### Task: Analyze data
<user_query>Покажи статистику</user_query>
```

---

## API документація

### Pipelines API

#### POST /v1/chat/completions

Головний endpoint для обробки запитів.

**Request:**
```json
{
  "model": "spreadsheet-analyst",
  "messages": [
    {"role": "system", "content": "You are a data analyst..."},
    {"role": "user", "content": "Покажи топ-10..."}
  ],
  "file": {
    "id": "file-uuid",
    "filename": "data.csv",
    "content_type": "text/csv"
  },
  "stream": true
}
```

**Response (stream):**
```
data: {"type": "status", "data": {"description": "Стартую обробку запиту.", "done": false}}

data: {"type": "status", "data": {"description": "Файл знайдено: file-uuid", "done": false}}

data: {"type": "status", "data": {"description": "Завантажую таблицю в sandbox.", "done": false}}

data: {"type": "status", "data": {"description": "Генерую план та код аналізу.", "done": false}}

data: {"type": "status", "data": {"description": "Виконую аналіз у sandbox.", "done": false}}

data: {"type": "status", "data": {"description": "Готово. Відповідь сформована.", "done": true}}

data: {"type": "response", "data": "Топ-10 клієнтів:\n| customer_id | total_spent |\n|-------------|-------------|\n| 1 | 15000 |\n..."}
```

### Sandbox API

#### POST /v1/dataframe/load

Завантажує DataFrame у sandbox.

**Request:**
```json
{
  "file_id": "file-uuid",
  "filename": "data.csv",
  "content_type": "text/csv",
  "data_b64": "base64encodedcontent",
  "max_rows": 200000,
  "preview_rows": 200000
}
```

**Response:**
```json
{
  "df_id": "uuid-123",
  "profile": {
    "rows": 10000,
    "cols": 15,
    "columns": ["col1", "col2"],
    "dtypes": {"col1": "int64"},
    "nulls_top": {"col1": 0},
    "preview": [...]
  }
}
```

#### POST /v1/dataframe/run

Виконує код над DataFrame.

**Request:**
```json
{
  "df_id": "uuid-123",
  "code": "result = df.groupby('category')['price'].sum()",
  "timeout_s": 120,
  "preview_rows": 200000,
  "max_cell_chars": 200,
  "max_stdout_chars": 8000
}
```

**Response:**
```json
{
  "status": "ok",
  "stdout": "",
  "result_text": "| category | sum |\n|----------|-----|\n| A | 100 |\n...",
  "result_meta": {"rows": 5, "cols": 2},
  "mutation_summary": {...},
  "committed": true,
  "auto_committed": false,
  "structure_changed": false,
  "profile": {...}
}
```

#### GET /v1/dataframe/{df_id}/profile

Отримує поточний профіль DataFrame.

**Response:**
```json
{
  "df_id": "uuid-123",
  "profile": {...},
  "ts": 1234567890.123
}
```

---

## Розробка

### Локальна розробка

#### Структура проекту

```
excel_chat-main/
├── config/
│   └── user_params.env           # Конфігурація
├── docker-compose.yml             # Визначення сервісів
├── README.md                      # Цей файл
├── pipelines/
│   ├── spreadsheet_analyst_pipeline.py   # Основний пайплайн
│   ├── pipe/
│   │   └── pipe.py                      # Pipe connector
│   └── prompts.txt                      # Templates
├── sandbox_service/
│   ├── main.py                          # Sandbox API
│   ├── Dockerfile                       # Image definition
│   └── requirements.txt                 # Dependencies
└── docs/
    ├── README.md                        # Цей файл
    ├── FUNCTIONS.md                     # Детальна документація функцій
    └── DATA_FLOW.md                     # Карта потоків даних
```

#### Запуск локально (без Docker)

**Sandbox Service:**
```bash
cd sandbox_service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8081
```

**Pipelines:**
```bash
cd pipelines
pip install -r pipeline-requirements.txt
uvicorn spreadsheet_analyst_pipeline:pipeline --host 0.0.0.0 --port 9099
```

#### Додавання нових шаблонів

1. Відредагуйте `pipelines/prompts.txt`:

```txt
[custom_template]
You write custom pandas code...
Rules:
- Rule 1
- Rule 2
[/custom_template]
```

2. Використовуйте у pipeline:

```python
custom_prompt = _read_prompts(PROMPTS_PATH).get("custom_template", DEFAULT_PROMPT)
```

#### Додавання нових шорткатів

1. Додайте intent у `sandbox_service/catalog.json`:

```json
{
  "intents": [{
    "id": "my_custom_intent",
    "text": "мій запит",
    "slots": {...},
    "plan": [...]
  }]
}
```

2. Обов'язково оновіть FAISS індекс:

```bash
python -m scripts.build_index
```

---

## Приклади запитів

### Економічний аналіз

| Запит | Пояснення |
|-------|-----------|
| `Покажи топ-10 продуктів за виручкою` | GroupBy + Sum + Sort + Head |
| `Яка середня вартість замовлення?` | Mean aggregation |
| `Скільки унікальних клієнтів?` | nunique |
| `Знайди клієнта з найбільшою сумою` | Max + idxmax |

### Операційний аналіз

| Запит | Пояснення |
|-------|-----------|
| `Покажи останні 50 записів` | Tail |
| `Порахуй кількість записів за місяць` | GroupBy + Size |
| `Знайди продукти з низьким запасом` | Filter + Low threshold |

### Фінансовий аналіз

| Запит | Пояснення |
|-------|-----------|
| `Яка загальна виручка за рік?` | Sum |
| `Порахуй середнє, мін, макс по цінам` | Multiple aggregations |
| `Знайди місяць з найвищою виручкою` | GroupBy + Argmax |

### Редагування даних

| Запит | Пояснення |
|-------|-----------|
| `Видали рядок 5` | Drop by index |
| `Додай рядок: A, 100, 50` | Concat |
| `Зміни колонку price: 10→9.99` | At assignment |

---

## Діагностика

### Поширені помилки

#### 1. File not found

```
Будь ласка, прикріпіть файл CSV/XLSX для аналізу.
```

**Рішення:**
- Перевірте, чи файл прикріплений до запиту
- Перевірте `WEBUI_BASE_URL` та `WEBUI_API_KEY`
- Перевірте доступність OpenWebUI

#### 2. Sandbox connection failed

```
Не вдалося завантажити таблицю в пісочницю (sandbox).
```

**Рішення:**
- Перевірте `SANDBOX_URL`
- Перевірте `SANDBOX_API_KEY`
- Перевірте логи sandbox service

#### 3. Code generation failed

```
Я не зміг згенерувати код для цього запиту.
```

**Рішення:**
- Спробуйте переформулювати запит
- Перевірте `BASE_LLM_BASE_URL` та `BASE_LLM_API_KEY`
- Перевірте логи pipelines

#### 4. Edit operation not committed

```
Зміни не були зафіксовані. Спробуйте ще раз або вкажіть COMMIT_DF = True явно.
```

**Рішення:**
- Для edit-операцій використовуйте `df = df.copy()` та `COMMIT_DF = True`
- Перевірте синтаксис модифікації

### Логування

#### Pipeline logs

```bash
# Docker
docker compose logs -f pipelines

# Логування подій
event=pipe_sync_query
event=shortcut_router status=hit/miss
event=llm_json_request/response
event=analysis_code
event=commit_result
event=final_answer
```

#### Sandbox logs

```bash
# Docker
docker compose logs -f sandbox

# Логування подій
event=df_store_loaded
event=df_run_code
event=df_committed
```

### Налагодження

#### Увімкнення debug режиму

```bash
PIPELINE_DEBUG=true
```

#### Вивід детальних логів

```
event=query_selection source=effective_user_query preview="..."
event=shortcut_router status=hit intent_id=... score=0.85
event=llm_json_request system_preview="..." user_preview="..."
event=analysis_code preview="..."
event=final_answer mode=deterministic/llm preview="..."
```

### Системні перевірки

#### Перевірка доступності сервісів

```bash
# OpenWebUI
curl http://localhost:3000

# Pipelines
curl http://localhost:9099/v1/models

# Sandbox
curl http://localhost:8081/health
```

#### Перевірка конфігурації

```bash
# Поточні змінні
docker compose exec pipelines env | grep PIPELINE
docker compose exec sandbox env | grep -E "MAX|TIMEOUT"
```

---

## Додаткові ресурси

- [Functions Documentation](./FUNCTIONS.md) — Детальна документація всіх функцій
- [Data Flow Documentation](./DATA_FLOW.md) — Карта потоків даних та життєвий цикл
- [Shortcut Debug Dashboard](./SHORTCUT_DEBUG_DASHBOARD.md) — Як запустити локальний дашборд трейсів

---

**Автор:** Claude Code (Qwen 2.5 Coder / Qwen 3)
**Версія документації:** 1.0.0
**Останнє оновлення:** 2026-02-08
