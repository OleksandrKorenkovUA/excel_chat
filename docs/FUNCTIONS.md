# Детальна документація функцій проекту

## Підсистема: Spreadsheet Analyst Pipeline

---

### Функції утилітного модуля (spreadsheet_analyst_pipeline.py)

---

#### ` _safe_trunc(text: str, limit: int) -> str`

**Призначення:** Обмежує довжину тексту до вказаного ліміту символів, додавши три крапки в кінці, якщо текст перевищує ліміт.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Вхідний рядок для обрізання |
| limit | `int` | Максимальна довжина результуючого рядка |

**Результат:** `str` — Обрізаний рядок з додатковим префіксом "..." якщо довжина перевищувала ліміт.

**Приклади використання:**
```python
# Приклад 1: Обрізання довгого тексту
text = "Це дуже довге повідомлення, яке потрібно скоротити для логування"
result = _safe_trunc(text, 20)
# Результат: "Це дуже довге пові..."

# Приклад 2: Короткий текст (без змін)
short = "Привіт"
result = _safe_trunc(short, 100)
# Результат: "Привіт"
```

---

#### `_extract_request_trace_ids(body: dict) -> Tuple[Optional[str], Optional[str]]`

**Призначення:** Витягує ідентифікатори запиту та трасування з тіла HTTP-запиту для логування.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| body | `dict` | Тіло HTTP-запиту з метаданими |

**Результат:** `Tuple[Optional[str], Optional[str]]` — Кортеж (request_id, trace_id)

**Приклади використання:**
```python
body = {"request_id": "req-123", "trace_id": "trace-456"}
req_id, trace_id = _extract_request_trace_ids(body)
# req_id = "req-123"
# trace_id = "trace-456"
```

---

#### `_session_key(body: dict) -> str`

**Призначення:** Генерує унікальний ключ для кешу сесії на основі metadata запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| body | `dict` | Тіло HTTP-запиту |

**Результат:** `str` — Унікальний рядок ключа сесії

**Приклади використання:**
```python
body = {"chat_id": "chat-123", "user_id": "user-456"}
key = _session_key(body)
# key = "chat-123:user-456"
```

---

#### `_read_prompts(path: str) -> dict`

**Призначення:** Зчитує шаблони запитів (prompts) з текстового файлу у форматі `[section_name] content`.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| path | `str` | Шлях до файлу з prompts |

**Результат:** `dict` — Словник з назвами секцій та їх контентом

**Приклади використання:**
```python
prompts = _read_prompts("pipelines/prompts.txt")
# prompts = {
#     "plan_code_system": "You write pandas code...",
#     "final_answer_system": "You are a data analysis assistant..."
# }
```

---

#### `_effective_user_query(user_message: str, messages: List[dict]) -> str`

**Призначення:** Витягує реальний запит користувача з історії повідомлень, враховуючи мета-завдання.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| user_message | `str` | Останнє повідомлення користувача |
| messages | `List[dict]` | Історія чату |

**Результат:** `str` — Очищений текст запиту або порожній рядок

**Приклади використання:**
```python
messages = [
    {"role": "system", "content": "### Task: Analyze data"},
    {"role": "user", "content": "Покажи топ-10 продуктів"}
]
query = _effective_user_query("Покажи топ-10 продуктів", messages)
# query = "Покажи топ-10 продуктів"
```

---

#### `_normalize_query_text(text: str) -> str`

**Призначення:** Нормалізує текст запиту — видаляє HTML-теги та видаляє зайві пробіли.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Вхідний текст запиту |

**Результат:** `str` — Нормалізований текст

**Приклади використання:**
```python
text = "<p>Скільки рядків у таблиці?</p>"
normalized = _normalize_query_text(text)
# normalized = "Скільки рядків у таблиці?"
```

---

#### `_is_meta_task_text(text: str) -> bool`

**Призначення:** Визначає, чи є запит мета-завданням (директива OpenWebUI для системних дій).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |

**Результат:** `bool` — `True`, якщо це мета-завдання

**Приклади використання:**
```python
_is_meta_task_text("### Task: Analyze data")  # True
_is_meta_task_text("Скільки рядків?")  # False
```

---

#### `_is_search_query_meta_task(text: str) -> bool`

**Призначення:** Перевіряє, чи є запит пошуковим мета-завданням (не для аналізу даних).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |

**Результат:** `bool` — `True`, якщо це пошуковий запит

---

#### `_pick_file_ref(body: dict, messages: List[dict]) -> Tuple[Optional[str], Optional[dict]]`

**Призначення:** Знаходить посилання на файл у тілі запиту або повідомленнях.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| body | `dict` | Тіло HTTP-запиту |
| messages | `List[dict]` | Історія повідомлень |

**Результат:** `Tuple[Optional[str], Optional[dict]]` — (file_id, file_object) або (None, None)

---

#### `_query_selection_debug(messages: List[dict]) -> str`

**Призначення:** Генерує зручний для логування рядок з інформацією про вибір запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| messages | `List[dict]` | Історія повідомлень |

**Результат:** `str` — Рядок з деталями вибору запиту

---

#### `_has_edit_triggers(text: str) -> bool`

**Призначення:** Перевіряє наявність ключових слів, що вказують на намір редагувати дані.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |

**Результат:** `bool` — `True`, якщо запит містить тригери для редагування

**Приклади використання:**
```python
_has_edit_triggers("Видали рядок 5")  # True
_has_edit_triggers("Знайди максимум")  # False
```

---

#### `_infer_op_from_question(question: str) -> str`

**Призначення:** Визначає тип операції (read/edit) на основі аналізу запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |

**Результат:** `str` — "read" або "edit"

---

#### `_is_total_value_scalar_question(question: str, profile: Optional[dict]) -> bool`

**Призначення:** Перевіряє, чи запит стосується загальної вартості/суми.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `Optional[dict]` | Метадані таблиці |

**Результат:** `bool` — `True`, якщо запит про загальну вартість

---

#### `_detect_metrics(question: str) -> List[str]`

**Призначення:** Виявляє статистичні метрики, про які йде мова у запиті.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |

**Результат:** `List[str]` — Список виявлених метрик (min, max, mean, sum, count, median)

**Приклади використання:**
```python
metrics = _detect_metrics("Знайди мінімум і середнє значення")
# metrics = ["min", "mean"]
```

---

#### `_pick_relevant_column(question: str, columns: List[str]) -> Optional[str]`

**Призначення:** Вибирає найбільш релевантну колонку на основі семантичного збігу з запитом.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| columns | `List[str]` | Список доступних колонок |

**Результат:** `Optional[str]` — Ім'я колонки або `None`

---

#### `_is_availability_count_intent(question: str) -> bool`

**Призначення:** Визначає намір користувача порахувати наявність/доступність товарів.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |

**Результат:** `bool` — `True`, якщо запит про наявність

---

#### `_availability_target_mode(question: str) -> str`

**Призначення:** Визначає режим відповіді для запитів про наявність.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |

**Результат:** `str` — "boolean" або "detailed"

---

#### `_pick_availability_column(question: str, profile: dict) -> Optional[str]`

**Призначення:** Вибирає колонку, що містить дані про наявність товарів.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Optional[str]` — Ім'я колонки або `None`

---

#### `_extract_top_n_from_question(question: str, default: int = 10) -> int`

**Призначення:** Витягує параметр top-N (кількість результатів) з запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| default | `int` | Значення за замовчуванням |

**Результат:** `int` — Кількість результатів

**Приклади використання:**
```python
_extract_top_n_from_question("Покажи топ-5 продуктів")  # 5
_extract_top_n_from_question("Покажи найкращі 10")  # 10
_extract_top_n_from_question("Покажи результати")  # 10 (default)
```

---

#### `_find_columns_in_text(text: str, columns: List[str]) -> List[str]`

**Призначення:** Знаходить імена колонок у тексті запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |
| columns | `List[str]` | Список доступних колонок |

**Результат:** `List[str]` — Список знайдених колонок

---

#### `_find_column_in_text(text: str, columns: List[str]) -> Optional[str]`

**Призначення:** Знаходить одну найбільш релевантну колонку в тексті.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |
| columns | `List[str] | Список доступних колонок |

**Результат:** `Optional[str]` — Ім'я колонки або `None`

---

#### `_find_column_by_index(text: str, columns: List[str]) -> Optional[str]`

**Призначення:** Витягує ім'я колонки за її порядковим номером (1-based індекс).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |
| columns | `List[str]` | Список доступних колонок |

**Результат:** `Optional[str]` — Ім'я колонки або `None`

---

#### `_parse_literal(value: str) -> Any`

**Призначення:** Парсе літерал (число, рядок, boolean, null) з тексту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| value | `str` | Текст літерала |

**Результат:** `Any` — Python об'єкт відповідного типу

---

#### `_parse_number(text: str) -> Optional[float]`

**Призначення:** Парсе числове значення з тексту (підтримує коми як десятковий розділювач).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст з числом |

**Результат:** `Optional[float]` — Число або `None`

---

#### `_parse_row_index(text: str) -> Optional[int]`

**Призначення:** Витягує номер рядка (1-based) з тексту запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |

**Результат:** `Optional[int]` — Номер рядка або `None`

---

#### `_parse_condition(text: str, columns: List[str]) -> Optional[Tuple[str, str, Optional[str], Optional[float]]]`

**Призначення:** Парсе умову фільтрації з тексту запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |
| columns | `List[str]` | Список доступних колонок |

**Результат:** `Optional[Tuple[str, str, Optional[str], Optional[float]]]` — (column, operator, value, numeric_value)

---

#### `_parse_set_value(text: str) -> Optional[str]`

**Призначення:** Витягує значення для встановлення в операції редагування клітинки.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| text | `str` | Текст запиту |

**Результат:** `Optional[str]` — Значення як рядок або `None`

---

#### `_classify_columns_by_role(question: str, found: List[str], profile: dict) -> Dict[str, Any]`

**Призначення:** Класифікує знайдені колонки за їх роллю (group_by, aggregate, top_n).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| found | `List[str]` | Знайдені колонки |
| profile | `dict` | Метадані таблиці |

**Результат:** `Dict[str, Any]` — Словник з ролями колонок

---

#### `_template_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]`

**Призначення:** Генерує оптимізований pandas-код для поширених шаблонів запитів.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Optional[Tuple[str, str]]` — (code, plan) або `None`

**Приклади використання:**
```python
# Запит: "Покажи перші 5 рядків"
code, plan = _template_shortcut_code("Покажи перші 5 рядків", profile)
# code = "result = df.head(5)\n"
# plan = "Отримати перші 5 рядків таблиці."
```

---

#### `_edit_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]`

**Призначення:** Генерує код для операцій редагування даних (видалення/додавання рядків/колонок).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Optional[Tuple[str, str]]` — (code, plan) або `None`

---

#### `_stats_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]`

**Призначення:** Генерує код для статистичних обчислень (min, max, mean, sum, count, median).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Optional[Tuple[str, str]]` — (code, plan) або `None`

---

#### `_choose_column_from_question(question: str, profile: dict) -> Optional[str]`

**Призначення:** Вибирає релевантну колонку для статистичних операцій.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Optional[str]` — Ім'я колонки або `None`

---

#### `_emit_status(event_emitter: Any, description: str, done: bool = False, hidden: bool = False) -> None`

**Призначення:** Відправляє статусні повідомлення через event emitter.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| event_emitter | `Any` | Функція/об'єкт для відправки подій |
| description | `str` | Опис події |
| done | `bool` | Прапорець завершення |
| hidden | `bool` | Прапорець прихованості |

**Результат:** `None` — Процедура (без повернення)

---

#### `_status_marker(description: str, done: bool = False, hidden: bool = False) -> str`

**Призначення:** Генерує маркер статусу для вставки в потік відповіді.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| description | `str` | Опис події |
| done | `bool` | Прапорець завершення |
| hidden | `bool` | Прапорець прихованості |

**Результат:** `str` — JSON-рядок маркера статусу

---

#### `_guess_filename(meta: dict) -> str`

**Призначення:** Витягує ім'я файлу з метаданих.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| meta | `dict` | Метадані файлу |

**Результат:** `str` — Ім'я файлу або порожній рядок

---

#### `_status_message(event: str, payload: Optional[dict]) -> str`

**Призначення:** Генерує українське повідомлення статусу на основі типу події.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| event | `str` | Тип події (start, file_id, codegen, sandbox_run, тощо) |
| payload | `Optional[dict]` | Додаткові дані події |

**Результат:** `str` — Українське повідомлення статусу

---

#### `_normalize_generated_code(code: str) -> str`

**Призначення:** Нормалізує згенерований код — виправляє помилки із відступами та форматуванням.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| code | `str` | Згенерований Python код |

**Результат:** `str` — Нормалізований код

---

#### `_finalize_code_for_sandbox(question: str, code: str, op: str, commit_df: Optional[bool], df_profile: dict) -> Tuple[str, Optional[bool], Optional[str]]`

**Призначення:** Фінальна обробка коду перед виконанням у sandbox.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| code | `str` | Згенерований код |
| op | `str` | Тип операції (read/edit) |
| commit_df | `Optional[bool]` | Флаг commit |
| df_profile | `dict` | Метадані таблиці |

**Результат:** `Tuple[str, Optional[bool], Optional[str]]` — (code, commit_df, error)

---

#### `_validate_edit_code(code: str, op: str) -> Tuple[bool, Optional[str]]`

**Призначення:** Валідує код операції редагування.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| code | `str` | Код для валідації |
| op | `str` | Тип операції |

**Результат:** `Tuple[bool, Optional[str]]` — (is_valid, error_message)

---

#### `_enforce_count_code(question: str, code: str) -> Tuple[str, Optional[str]]`

**Призначення:** Додає валідацію для операцій підрахунку.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| code | `str` | Код для модифікації |

**Результат:** `Tuple[str, Optional[str]]` — (modified_code, error)

---

#### `_enforce_entity_nunique_code(question: str, code: str, profile: dict) -> str`

**Призначення:** Додає валідацію для підрахунку унікальних сутностей.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| code | `str` | Код для модифікації |
| profile | `dict` | Метадані таблиці |

**Результат:** `str` — Модифікований код

---

#### `_has_forbidden_import_nodes(code: str) -> bool`

**Призначення:** Перевіряє наявність заборонених import-нод у коді (AST аналіз).

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| code | `str` | Python код для перевірки |

**Результат:** `bool` — `True`, якщо знайдено заборонені import

---

#### `_profile_fingerprint(profile: dict) -> Optional[str]`

**Призначення:** Генерує хеш-відбиток профілю таблиці для порівняння змін.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| profile | `dict` | Метадані таблиці |

**Результат:** `Optional[str]` — SHA256 хеш або `None`

---

### Клас Pipeline

---

#### `class Pipeline`

**Призначення:** Головний клас пайплайну, що обробляє запити користувачів та оркеструє роботу з таблицями.

**Конфігурація (Valves):**
| Параметр | Тип | Значення за замовч. | Опис |
|----------|-----|-------------------|------|
| id | `str` | "spreadsheet-analyst" | Ідентифікатор пайплайну |
| name | `str` | "Spreadsheet Analyst" | Назва пайплайну |
| description | `str` | - | Опис функціональності |
| debug | `bool` | False | Режим налагодження |
| webui_base_url | `str` | "http://host.docker.internal:3000" | URL OpenWebUI |
| webui_api_key | `str` | "" | API-ключ для OpenWebUI |
| base_llm_base_url | `str` | - | URL LLM API |
| base_llm_api_key | `str` | "" | API-ключ для LLM |
| base_llm_model | `str` | "chat-model" | Модель LLM |
| sandbox_url | `str` | "http://sandbox:8081" | URL sandbox service |
| sandbox_api_key | `str` | "" | API-ключ для sandbox |
| max_rows | `int` | 200000 | Макс. кількість рядків |
| preview_rows | `int` | 200000 | Кількість рядків прев'ю |
| max_cell_chars | `int` | 200 | Макс. довжина клітинки |
| code_timeout_s | `int` | 120 | Таймаут виконання коду |
| session_cache_ttl_s | `int` | 1800 | Час життя кешу сесії |
| shortcut_enabled | `bool` | True | Увімкнення шорткатів |
| shortcut_catalog_path | `str` | - | Шлях до каталогу шорткатів |
| shortcut_index_path | `str` | - | Шлях до FAISS індексу |
| shortcut_meta_path | `str` | - | Шлях до мета-даних |
| shortcut_top_k | `int` | 5 | Кількість результатів retrieval |
| shortcut_threshold | `float` | 0.35 | Поріг схожості retrieval |
| vllm_base_url | `str` | - | URL для embeddings |
| vllm_embed_model | `str` | - | Модель embeddings |
| vllm_api_key | `str` | "" | API-ключ для embeddings |

---

#### `Pipeline.pipe(...)` / `Pipeline._pipe_sync(...)` / `Pipeline._pipe_stream(...)`

**Призначення:** Основні методи обробки запитів — синхронний та потоковий.

**Аргументи (pipe):**
| Параметр | Тип | Опис |
|----------|-----|------|
| user_message | `str` | Повідомлення користувача |
| model_id | `str` | ID використаної моделі |
| messages | `List[dict]` | Історія чату |
| body | `dict` | Тіло HTTP-запиту |
| __event_emitter__ | `Any` | Функція для emit-подій |

**Результат:** `Union[str, Iterator, Generator]` — Рядок або потік для streaming

**Опис:** Метод координує весь процес обробки:
1. Витягує запит та файл
2. Завантажує таблицю в sandbox (або використовує кеш)
3. Спробує знайти шорткат через retrieval
4. Якщо не вдалося — генерує код через LLM
5. Виконує код у sandbox
6. Формує відповідь українською мовою

---

#### `Pipeline._sandbox_load(file_id: str, meta: dict, data: bytes) -> Dict[str, Any]`

**Призначення:** Завантажує файл у sandbox service.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| file_id | `str` | Ідентифікатор файлу в OpenWebUI |
| meta | `dict` | Метадані файлу |
| data | `bytes` | Бінарні дані файлу (base64-decoded) |

**Результат:** `Dict[str, Any]` — Відповідь sandbox: {"df_id": "...", "profile": {...}}

---

#### `Pipeline._sandbox_run(df_id: str, code: str) -> Dict[str, Any]`

**Призначення:** Виконує Python-код у sandbox service.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| df_id | `str` | Ідентифікатор DataFrame у sandbox |
| code | `str` | Python-код для виконання |

**Результат:** `Dict[str, Any]` — Відповідь sandbox з результатами

---

#### `Pipeline._llm_json(system: str, user: str) -> dict`

**Призначення:** Відправляє запит до LLM та парсе JSON-відповідь.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| system | `str` | Системний prompt |
| user | `str` | Користувацьке повідомлення |

**Результат:** `dict` — Розпарсений JSON з LLM відповіді

---

#### `Pipeline._plan_code(question: str, profile: dict) -> Tuple[str, str, str, Optional[bool]]`

**Призначення:** Генерує план аналізу та код через LLM.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Tuple[str, str, str, Optional[bool]]` — (code, plan, op, commit_df)

---

#### `Pipeline._plan_code_retry_missing_result(...)`

**Призначення:** Повторна генерація коду з явним вказівкою на необхідність result.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |
| previous_code | `str` | Попередній код |
| reason | `str` | Причина повтору |

**Результат:** `Tuple[str, str, str, Optional[bool]]` — (code, plan, op, commit_df)

---

#### `Pipeline._resolve_shortcut_placeholders(analysis_code: str, plan: str, question: str, profile: dict) -> Tuple[str, str]`

**Призначення:** Замінює плейсхолдери в шорткатах на реальні значення.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| analysis_code | `str` | Код шортката з плейсхолдерами |
| plan | `str` | План з плейсхолдерами |
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |

**Результат:** `Tuple[str, str]` — (code, plan) з заміненими значеннями

---

#### `Pipeline._deterministic_answer(question: str, result_text: str, profile: Optional[dict]) -> Optional[str]`

**Призначення:** Формує детерміновану відповідь без LLM для простих запитів.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| result_text | `str` | Текст результату виконання |
| profile | `Optional[dict]` | Метадані таблиці |

**Результат:** `Optional[str]` — Відповідь або `None` (якщо потрібен LLM)

---

#### `Pipeline._final_answer(...)`

**Призначення:** Формує фінальну відповідь користувачу через LLM.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| question | `str` | Текст запиту |
| profile | `dict` | Метадані таблиці |
| plan | `str` | План виконання |
| code | `str` | Виконаний код |
| edit_expected | `bool` | Ч очікувалося редагування |
| run_status | `str` | Статус виконання |
| stdout | `str` | Вивід stdout |
| result_text | `str` | Текст результату |
| result_meta | `dict` | Мета даних результату |
| mutation_summary | `Optional[dict]` | Звіт про зміни |
| mutation_flags | `Optional[dict]` | Прапорці коміту |
| error | `str` | Повідомлення про помилку |

**Результат:** `str` — Фінальна відповідь українською мовою

---

## Підсистема: Sandbox Service

---

### Функції безпеки та виконання коду

---

#### `_ast_guard(code: str) -> None`

**Призначення:** Валідує Python-код на заборонені конструкції через AST.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| code | `str` | Python код для перевірки |

**Результат:** `None` — Викидає `ValueError` при знаходженні заборонених елементів

**Заборонені елементи:**
- Import/ImportFrom
- Global/Nonlocal
- With/AsyncWith
- Try/Raise
- Lambda
- FunctionDef/AsyncFunctionDef/ClassDef
- Delete

**Заборонені виклики:**
- eval, exec, compile
- open, input, __import__
- globals, locals, vars, dir
- getattr, setattr, delattr
- pd.read_*, pd.to_*
- df.to_*

---

#### `_run_code(...)`

**Призначення:** Безпечне виконання Python-коду в ізольованому процесі.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| code | `str` | Python код для виконання |
| df | `pd.DataFrame` | DataFrame для обробки |
| timeout_s | `int` | Таймаут виконання |
| preview_rows | `int` | Кількість рядків прев'ю |
| max_cell_chars | `int` | Макс. довжина клітинки |
| max_stdout_chars | `int` | Макс. довжина stdout |
| max_result_chars | `int` | Макс. довжина результату |

**Результат:** `Tuple` — (status, stdout, result_text, result_meta, error, mutation_summary, df_out, commit_flag, undo_flag, auto_commit_flag, structure_changed)

---

### API endpoints (FastAPI)

---

#### `POST /v1/dataframe/load`

**Призначення:** Завантажує файл у sandbox та створює DataFrame.

**Тіло запиту:**
| Параметр | Тип | Обов'язковий | Опис |
|----------|-----|--------------|------|
| file_id | `str` | Ні | Ідентифікатор файлу |
| filename | `str` | Ні | Ім'я файлу |
| content_type | `str` | Ні | MIME тип |
| data_b64 | `str` | Так | Бінарні дані base64 |
| max_rows | `int` | Ні | Макс. рядків |
| preview_rows | `int` | Ні | Рядків прев'ю |

**Відповідь:**
```json
{
  "df_id": "uuid-str",
  "profile": {
    "rows": 1000,
    "cols": 10,
    "columns": ["col1", "col2"],
    "dtypes": {"col1": "int64"},
    "nulls_top": {"col1": 0},
    "preview": [...]
  }
}
```

---

#### `POST /v1/dataframe/run`

**Призначення:** Виконує Python-код над DataFrame.

**Тіло запиту:**
| Параметр | Тип | Обов'язковий | Опис |
|----------|-----|--------------|------|
| df_id | `str` | Так | ID DataFrame |
| code | `str` | Так | Python код |
| timeout_s | `int` | Ні | Таймаут |
| preview_rows | `int` | Ні | Рядків прев'ю |

**Відповідь:**
```json
{
  "status": "ok",
  "stdout": "",
  "result_text": "...",
  "result_meta": {...},
  "mutation_summary": {...},
  "error": "",
  "profile": {...},
  "committed": true,
  "auto_committed": false,
  "structure_changed": false
}
```

---

## Підсистема: Shortcut Router

---

#### `class ShortcutRouterConfig`

**Призначення:** Конфігурація ShortcutRouter.

---

#### `class ShortcutRouter`

**Призначення:** Система retrieval-пошуку для пошуку шаблонів запитів через FAISS.

---

#### `ShortcutRouter.shortcut_to_sandbox_code(query: str, profile: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]`

**Призначення:** Знаходить та компілює шорткат для запиту.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| query | `str` | Текст запиту |
| profile | `Dict[str, Any]` | Метадані таблиці |

**Результат:** `Optional[Tuple[str, Dict[str, Any]]]` — (code, meta) або `None`

---

#### `ShortcutRouter._embed_query(query: str) -> Optional[np.ndarray]`

**Призначення:** Генерує embedding-вектор для запиту через vLLM.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| query | `str` | Текст запиту |

**Результат:** `Optional[np.ndarray]` — Вектор або `None`

---

#### `ShortcutRouter._retrieve(query: str) -> Optional[Tuple[str, float, str]]`

**Призначення:** Виконує пошук у FAISS індексі.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| query | `str` | Текст запиту |

**Результат:** `Optional[Tuple[str, float, str]]` — (intent_id, score, example)

---

#### `ShortcutRouter._llm_groupby_slots(query: str, columns: List[str], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]`

**Призначення:** Використовує LLM для розпізнавання колонок у group-by запитах.

**Аргументи:**
| Параметр | Тип | Опис |
|----------|-----|------|
| query | `str` | Текст запиту |
| columns | `List[str]` | Список колонок |
| profile | `Dict[str, Any]` | Метадані |

**Результат:** `Optional[Dict[str, Any]]` — {group_col, target_col, agg, top_n}

---

#### `ShortcutRouter._fill_slots(...)` / `_slot_from_text(...)` / `_slot_from_llm(...)`

**Призначення:** Заповнює плейсхолдери шаблону значеннями.

**Типи слотів:**
- `column` — ім'я колонки
- `columns` — список колонок
- `row_indices` — список індексів рядків
- `int/float` — числа
- `bool` — boolean значення
- `enum` — вибір з переліку

---

## Підсистема: Pipe (OpenWebUI Connector)

---

#### `pipe.py`

**Призначення:** Коннектор між OpenWebUI та pipelines service.

**Основні функції:**
- Перенаправлення запитів з OpenWebUI
- Fallback на зовнішній API якщо pipelines недоступний
- Логування статусів обробки
