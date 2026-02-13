# Карта потоків даних та життєвий цикл запитів

## 1. Огляд системи

Система обробляє природні запити користувачів щодо табличних даних (CSV/XLSX/TSV/JSON/Parquet) та повертає результати аналізу українською мовою.

### Архітектурні компоненти

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  OpenWebUI      │────>│  Pipelines      │────>│  Sandbox Service │
│  (Frontend)     │     │  (Orchestrator) │     │  (Executor)      │
└─────────────────┘     └─────────────────┘     └──────────────────┘
        ▲                       │                        │
        │                       │                        │
        └───────────────────────┴────────────────────────┘
                    Повернення результатів
```

---

## 2. Життєвий цикл запиту користувача

### Етап 1: Отримання запиту

```
Користувач → OpenWebUI → Pipe (pipe.py) → Pipelines
```

**Опис процесу:**
1. Користувач формулює запит у чаті разом із прикріпленим файлом
2. OpenWebUI перенаправляє запит через Pipe connector
3. Pipe витягує останній текстовий запит та інформацію про прикріплені файли
4. Запит надсилається у pipelines service

**Структура запиту:**
```json
{
  "chat_id": "uuid",
  "user_message": "Покажи топ-10 продуктів за виручкою",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Покажи топ-10..."}
  ],
  "files": [
    {
      "id": "file-uuid",
      "filename": "sales_2024.csv",
      "content_type": "text/csv"
    }
  ],
  "stream": true
}
```

---

### Етап 2: Витягування файлу та завантаження в Sandbox

```
Pipelines → OpenWebUI API → Sandbox Service
```

**Опис процесу:**
1. Pipelines витягує `file_id` з запиту або кешу сесії
2. Запит до OpenWebUI API для отримання метаданих файлу
3. Завантаження бінарних даних файлу
4. Кодування в base64 та відправка в Sandbox
5. Sandbox створює DataFrame та генерує профіль

**Профіль таблиці (profile):**
```json
{
  "rows": 10000,
  "cols": 15,
  "columns": ["product_id", "name", "category", "price", "quantity", ...],
  "dtypes": {
    "product_id": "int64",
    "name": "object",
    "price": "float64",
    "quantity": "int64"
  },
  "nulls_top": {"price": 12, "quantity": 5},
  "preview": [
    {"product_id": 1, "name": "Product A", "price": 100.5, ...},
    ...
  ]
}
```

**Кешування:**
- Сесія кешується за ключем `{chat_id}:{user_id}`
- TTL: 1800 секунд (30 хвилин)
- Включає: `file_id`, `df_id`, `profile`, `profile_fp`

---

### Етап 3: Вибір методу обробки

```
Запит + Профіль → Логіка вибору

┌─────────────────────────────────────────────────────────────┐
│                        Вибір методу                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Перевірка на edit-операції (видалити, додати, змінити)   │
│    └─> Якщо є → Пропуск retrieval, пряме генерування коду  │
│                                                             │
│ 2. Спроба retrieval через ShortcutRouter                    │
│    └─> Embed запит → FAISS search → Intent matching        │
│    └─> Якщо score > threshold → Компіляція шортката        │
│                                                             │
│ 3. Rule-based шорткати (pattern matching)                   │
│    └─> _template_shortcut_code (head/tail/preview)         │
│    └─> _edit_shortcut_code (add/drop/replace)              │
│    └─> _stats_shortcut_code (min/max/sum/mean/count)       │
│                                                             │
│ 4. LLM-based генерація коду                                 │
│    └─> _plan_code → LLM запит → JSON response              │
└─────────────────────────────────────────────────────────────┘
```

---

### Етап 4: Генерація/обробка коду

```
┌─────────────────────────────────────────────────────────────┐
│                        Модулі генерації                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Шорткат (retrieval):                                        │
│   Query → Embedding → FAISS Index → Intent ID             │
│   ↓                                                         │
│   Intent (plan) + Slots → Template substitution            │
│   ↓                                                         │
│   Final pandas code                                        │
│                                                             │
│ Rule-based:                                                 │
│   Query → Pattern matching → Template selection            │
│   ↓                                                         │
│   Slot filling → Final pandas code                         │
│                                                             │
│ LLM-based:                                                  │
│   Query + Profile → System Prompt → LLM                    │
│   ↓                                                         │
│   JSON (code, plan, op, commit_df)                        │
│   ↓                                                         │
│   Validation & normalization                               │
└─────────────────────────────────────────────────────────────┘
```

**Приклад LLM prompts:**

**System prompt (plan_code_system):**
```
You write pandas code to answer questions about a DataFrame named df.
Return ONLY valid JSON object with keys: analysis_code, short_plan, op, commit_df.
Rules:
- op must be "edit" when user asks to modify data; otherwise "read".
- commit_df must be true only when DataFrame is modified.
- analysis_code must be pure Python statements, no imports, no file/network access.
- CRITICAL: The FINAL value MUST be assigned to variable named `result`.
- Do not use try/except, with, def, class.
- For op="edit", always persist changes in df and include: COMMIT_DF = True
```

**User payload:**
```json
{
  "question": "Покажи топ-10 продуктів за виручкою",
  "df_profile": {
    "columns": ["product_id", "name", "category", "price", "quantity", "revenue"],
    "dtypes": {"revenue": "float64"}
  }
}
```

---

### Етап 5: Валідація та нормалізація

```
Згенерований код → Валідація → Нормалізація → Sandbox code
```

**Етапи валідації:**

1. **ASTGuard** — Перевірка AST-дерева на заборонені конструкції:
   - Import/ImportFrom
   - eval/exec/compile
   - open/input
   - pd.read_*/to_*, df.to_*

2. **Result assignment** — Перевірка наявності `result = ...`:
   - Якщо відсутній → Retry з підказкою

3. **Operation type check** — Співставлення `op`:
   - `read` — не повинен містити COMMIT_DF
   - `edit` — повинен містити COMMIT_DF

4. **Format normalization**:
   - `textwrap.dedent` для виправлення відступів
   - Додавання `df_profile = {...}` якщо використовується

**Приклад нормалізованого коду:**
```python
result = df.groupby("category")["revenue"].sum().reset_index(name="total_revenue")
result = result.sort_values("total_revenue", ascending=False).head(10)
```

---

### Етап 6: Виконання в Sandbox

```
Pipelines → Sandbox /v1/dataframe/run
```

**Запит до Sandbox:**
```json
{
  "df_id": "uuid",
  "code": "result = df.groupby...",
  "timeout_s": 120,
  "preview_rows": 200000,
  "max_cell_chars": 200,
  "max_stdout_chars": 8000
}
```

**Виконання в Sandbox:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Sandbox Execution                        │
├─────────────────────────────────────────────────────────────┤
│ 1. _ast_guard(code) — повторна перевірка безпеки           │
│ 2. Multiprocessing worker з resource limits:               │
│    - CPU time: 120s                                        │
│    - Memory: 1024MB                                        │
│ 3. exec(code, env, env) в ізольованому середовищі          │
│    - df, pd, np, re доступні                               │
│    - safe_builtins без опасних функцій                     │
│ 4. Зчитування змінних з env:                               │
│    - result, COMMIT_DF, UNDO, df                           │
│ 5. Обчислення mutation_summary (якщо зміни)                │
└─────────────────────────────────────────────────────────────┘
```

**Відповідь Sandbox:**
```json
{
  "status": "ok",
  "stdout": "num_col source=revenue dtype=float64 nan_count=12\n",
  "result_text": "| category | total_revenue |\n|----------|---------------|\n| Electronics | 150000 |\n| ...",
  "result_meta": {"rows": 10, "cols": 2},
  "mutation_summary": {
    "rows_before": 10000,
    "rows_after": 10,
    "added_rows": 0,
    "removed_rows": 9990,
    "added_columns": [],
    "removed_columns": [],
    "changed_cells_count": 0,
    "changed_cells": []
  },
  "committed": true,
  "auto_committed": false,
  "structure_changed": true,
  "profile": {...}
}
```

---

### Етап 7: Формування відповіді

```
Результат → Вибір методу формування → Відповідь користувачу
```

**Логіка формування:**

```python
def _deterministic_answer(question, result_text, profile):
    # 1. Edit operations → статусна відповідь
    if "видали", "додай", "зміни" in question:
        return "Зміни успішно внесено"

    # 2. Scalar values → прямі відповіді
    if not "\n" in result_text:
        return f"{column} — {result_text}"

    # 3. Tables → markdown formatting
    if is_table:
        return format_table(result_text)

    # 4. Availability questions → special handling
    if "наявн" in question:
        return format_availability(result_text)

    # 5. Grouped data → top pairs formatting
    if "груп" in question or "кожн" in question:
        return format_top_pairs(result_text)

    # 6. Fallback → LLM
    return None  # use _final_answer
```

**LLM-based final answer:**

**System prompt (final_answer_system):**
```
You are a data analysis assistant. Answer in Ukrainian.
Use only the execution results provided.
If you mention any numbers, they must appear in result_text.
Do not mention model numbers or SKUs unless they appear in result_text.
If there is an error, explain what to change.
If stdout includes price conversion diagnostics, include them in the answer.
```

**User payload:**
```json
{
  "question": "Покажи топ-10 продуктів за виручкою",
  "df_profile": {...},
  "plan": "Групувати по категоріях, підрахувати суму виручки, відсортувати",
  "analysis_code": "result = df.groupby...",
  "exec_status": "ok",
  "stdout": "...",
  "result_text": "| category | total_revenue |\n...",
  "result_meta": {...},
  "error": ""
}
```

---

## 3. Діаграма потоків даних

### Повний потік (Sync)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Повний потік обробки запиту                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────┐     ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│  User   │────>│  Pipe   │────>│ Pipeline │────>│ Sandbox │────>│  User   │
└─────────┘     └─────────┘     └──────────┘     └─────────┘     └─────────┘
     │               │              │               │               │
     │ 1. POST       │ 2. Extract   │ 3. Load       │ 4. Execute    │
     │  /chat        │  file_id     │  to sandbox   │  code         │
     │               │              │               │               │
     │               │ 3. Get       │ 5. Run code   │ 6. Result     │
     │               │  profile     │  to sandbox   │  to pipeline  │
     │               │              │               │               │
     │               │ 4. Load      │ 7. Generate   │               │
     │               │  file        │  answer       │               │
     │               │              │               │               │
     │ 8. Stream     │              │               │               │
     │  response     │              │               │               │
```

### Потік даних DataFrame

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Життєвий цикл DataFrame                                 │
└─────────────────────────────────────────────────────────────────────────────┘

User Upload
    │
    ▼
┌─────────────────┐
│ File (CSV/XLSX) │
│ base64 encoded  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sandbox /load   │
│ pd.read_csv()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DataFrame      │
│  stored in      │
│  DF_STORE[df_id]│
└────────┬────────┘
         │
         │  Code execution
         ▼
┌─────────────────┐
│  Modified DF    │
│  (if COMMIT_DF) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Session Cache  │
│  profile updated│
└─────────────────┘
```

---

## 4. Система кешування

### Сесійний кеш (Pipeline)

```
Ключ: {chat_id}:{user_id}
Значення:
{
  "file_id": "...",
  "df_id": "...",
  "profile": {...},
  "profile_fp": "sha256hash",
  "ts": 1234567890.123
}

Термін дії: 1800 секунд (30 хв)
Оновлення при:
  - Співпадіння file_id + новий profile
  - Зміна структури після edit
```

### Кеш DataFrame (Sandbox)

```
DF_STORE[df_id] = {
  "df": DataFrame,
  "profile": {...},
  "ts": timestamp,
  "file_id": "...",
  "history": [df_v1, df_v2, ...]  # для UNDO
}

Макс. елементів: 32
TTL: 1800 секунд
MAX_DF_HISTORY: 5
```

---

## 5. Система шорткатів

### Shortcut Router Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Shortcut Router Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘

Query
    │
    ▼
┌─────────────────┐
│ _embed_query()  │
│ vLLM /embeddings│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FAISS Index     │
│ L2 normalized   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Search (top_k)  │
│ scores, idxs    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Threshold check │
│ score > threshold│
│ margin check    │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Hit    │    │  Miss
    ▼         │    ▼
┌────────┐   │  ┌────────────┐
│ Intent │   │  │ LLM        │
│ Slots  │   │  │ Generation │
│ Plan   │   │  └────────────┘
└────┬───┘   │
     │       │
     └───────┘
         │
         ▼
┌─────────────────┐
│ _compile_plan() │
│ Template substs │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ pandas code     │
└─────────────────┘
```

---

## 6. Безпека та валідація

### AST Guard

```
Python Code
    │
    ▼
┌─────────────────┐
│ ast.parse()     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Walk AST nodes  │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Forbidden│    │  OK
    ▼         │
┌────────┐   │
│ Raise  │   │
│ Error  │   │
└────────┘   │
             ▼
┌─────────────────┐
│ Safe to exec()  │
└─────────────────┘
```

### Мультипроцесне виконання

```
Parent Process
    │
    ├─> fork/spawn
    │
    ▼
┌─────────────────┐
│ Worker Process  │
│ resource.limits │
│ - CPU: 120s     │
│ - Memory: 1GB   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ exec(code, env) │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Queue  │
    │  Result │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Parent reads    │
│ timeout check   │
└─────────────────┘
```

---

## 7. Журналювання

### Структура логів

```
event=pipe_sync_query question="..."
event=shortcut_router status=hit intent_id=... score=0.85
event=llm_json_request system_preview="..." user_preview="..."
event=llm_json_response preview="..."
event=analysis_code preview="..."
event=commit_result edit_expected=true committed=true
event=final_answer mode=deterministic preview="..."
```

### Співвідношення подій

```
Start → Query → File → Sandbox Load →
Codegen → Shortcut/Llm → Sandbox Run →
Final Answer → End

Або помилки на будь-якому етапі
```
