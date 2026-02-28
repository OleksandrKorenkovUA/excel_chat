# Shortcut Debug Dashboard

Цей дашборд показує трейс-логи роутера shortcuts з файлу `debug_trace.jsonl` у браузері.

## Передумови

- Робоча директорія: корінь репозиторію (`/Users/home/excel_chat-main`)
- Python 3
- Наявний файл трейсів (за замовчуванням: `pipelines/shortcut_router/learning/debug_trace.jsonl`)

## Швидкий запуск

```bash
python3 tools/shortcut_debug_dashboard.py
```

Після запуску відкрий:

```text
http://127.0.0.1:8765
```

## Параметри запуску

```bash
python3 tools/shortcut_debug_dashboard.py \
  --host 127.0.0.1 \
  --port 8765 \
  --trace-path pipelines/shortcut_router/learning/debug_trace.jsonl \
  --limit 150
```

- `--host`: адреса bind (default: `127.0.0.1`)
- `--port`: порт HTTP-сервера (default: `8765`)
- `--trace-path`: шлях до jsonl з трейсами
- `--limit`: дефолтний ліміт рядків у таблиці

## Через змінну середовища

Замість `--trace-path` можна задати:

```bash
SHORTCUT_DEBUG_TRACE_PATH=pipelines/shortcut_router/learning/debug_trace.jsonl \
python3 tools/shortcut_debug_dashboard.py
```

## Перевірка, що сервіс працює

```bash
curl http://127.0.0.1:8765/api/health
```

Трейси API:

```bash
curl "http://127.0.0.1:8765/api/traces?limit=50"
```

## Типові проблеми

- `Couldn't connect to server`: дашборд не запущений або зайнятий порт.
- Порожня таблиця: файл трейсів порожній або неправильний `--trace-path`.
- Для доступу з іншої машини/контейнера запусти з `--host 0.0.0.0` і відкрий `http://<host-ip>:8765`.
