# Освітня Документація: Recursive Language Model (RLM)

**Мова:** Українська
**Рівень:** Для студентів та початківців
**Мета:** Детальне роз'яснення кожної функції, методу та класу проекту RLM

---

## Зміст

1. [Вступ до проекту RLM](#вступ-до-проекту-rlm)
   - [Карта промптів у pipeline (excel_chat-main)](#карта-промптів-у-pipeline-excel_chat-main)
2. [Головні компоненти](#головні-компоненти)
3. [Детальна документація по функціям](#детальна-документація-по-функціям)
   - [Core RLM](#core-rlm)
   - [LM Handler та Comms](#lm-handler-та-comms)
   - [Environments](#environments)
   - [Clients](#clients)
   - [Parsing Utilities](#parsing-utilities)
   - [Prompts Utilities](#prompts-utilities)
   - [Logger та Verbose](#logger-та-verbose)
   - [Types & Dataclasses](#types--dataclasses)

---

## Вступ до проекту RLM

**Що таке RLM?**
RLM (Recursive Language Model) — це фреймворк, який дозволяє мовним моделям (LLM) виконувати код у захищених середовищах (REPL) та рекурсивно викликати інші LLM для обробки великих обсягів даних.

**Як це працює?**
1. Користувач надає запит (prompt)
2. RLM створює REPL-середовище з даними контексту
3. Модель генерує Python-код для аналізу контексту
4. Код виконується в REPL, результат повертається
5. Якщо відповідь готова — фінальна, інакше процес повторюється

**Сценарій використання:**
```python
from rlm import RLM

# Створюємо екземпляр RLM
rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-4"})

# Виконуємо запит
result = rlm.completion("Аналогія: Літній сезон це як...")
print(result.response)
```

### Адаптація для `excel_chat-main` (code generation tool)

У цьому проекті RLM-підхід варто застосовувати як **інструмент генерації коду з ітеративною перевіркою**, а не як одноразовий JSON-планер:

1. `Generate`: модель генерує `analysis_code` для `df`.
2. `Validate`: пайплайн перевіряє guardrails (без `import`, обов'язковий `result =`, read/edit правила).
3. `Repair`: якщо код невалідний або впав у sandbox, модель отримує `previous_code + runtime_error` і генерує виправлену версію.
4. `Execute`: виконується останній валідний код; відповідь формується тільки з `result_text`.

Це знижує залежність від "strict JSON only" відповідей і краще працює з моделями, які додають `<think>`/reasoning у вивід.

### Поточна інтеграція в пайплайні

У `spreadsheet_analyst_pipeline.py` для RLM-ремонту виконання використовується `SandboxREPL` (REPL-обгортка над sandbox service). Після невдалого `sandbox_run` цей REPL-шар застосовується як пріоритетний шлях відновлення в `RLMCore`, щоб модель могла ітеративно виправити код через цикл `validate -> execute -> repair`.

### Карта промптів у pipeline (excel_chat-main)

Нижче карта, де саме формується `system`/`user` prompt і де він відправляється в модель.

| Stage | Джерело system prompt | Формування payload/messages | Виклик моделі |
| --- | --- | --- | --- |
| Планування коду (`plan_code`) | `DEFAULT_PLAN_CODE_SYSTEM` або `self._prompts["plan_code_system"]` | `_plan_code()` готує `{"question","schema",...}` | `_llm_json()` |
| Retry планування (`missing_result`, `missing_filter`, `read_mutation`, `runtime_error`) | той самий `plan_code_system` + `retry_constraints` | `_plan_code_retry_*()` | `_llm_json()` |
| RLM codegen tool | `DEFAULT_RLM_CODEGEN_SYSTEM` або `self._prompts["rlm_codegen_system"]` | `_plan_code_with_rlm_tool()` -> `RLMCore.completion(payload_obj)` | `LMHandler.completion()` |
| RLM core REPL repair | `DEFAULT_RLM_CORE_REPL_SYSTEM` або `self._prompts["rlm_core_repl_system"]` | `_repair_code_with_rlm_core_repl()` -> `RLMCore.completion(payload)` | `LMHandler.completion()` |
| Lookup/ranking/column slots | Inline `system` у `_llm_pick_*` методах | `_llm_pick_column_for_shortcut`, `_llm_pick_lookup_slots`, `_llm_pick_ranking_slots`, etc. | `_llm_json()` |
| Final rewrite | `DEFAULT_FINAL_REWRITE_SYSTEM` або `self._prompts["final_rewrite_system"]` | `_rewrite_from_result_text()` -> `rewrite_payload` | `chat.completions.create()` |
| Final answer | `DEFAULT_FINAL_ANSWER_SYSTEM` або `self._prompts["final_answer_system"]` | `_final_answer()` -> `payload` | `chat.completions.create()` |

Ключові точки в коді:

1. Завантаження prompt-файлу: `spreadsheet_analyst_pipeline.py::_read_prompts()` та `self._prompts = _read_prompts(PROMPTS_PATH)`.
2. Базові шаблони: `pipelines/lib/pipeline_prompts.py` (`DEFAULT_PLAN_CODE_SYSTEM`, `DEFAULT_RLM_*`, `DEFAULT_FINAL_*`).
3. Файловий prompt: `pipelines/prompts.txt` (`[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`).
4. Інжекція spreadsheet-skill у system prompt: `_with_spreadsheet_skill_prompt()`.
5. Уніфікований JSON gateway для більшості промптів: `_llm_json(system, user)`, де додається `STRICT OUTPUT CONTRACT`, далі формується `messages=[{"role":"system"}, {"role":"user"}]`.
6. RLM message-builder: `pipelines/lib/rlm_core.py::build_rlm_system_prompt()` + `build_user_prompt()`, далі `LMHandler.completion()` відправляє `messages` в `chat.completions.create()`.

---

## Головні компоненти

| Компонент          | Призначення                                                   |
| ------------------ | ------------------------------------------------------------- |
| **RLM**            | Головний клас, який керує всім процесом                       |
| **LMHandler**      | Слухає запити до LLM через socket сервер                      |
| **Environment**    | Середовище для виконання коду (LocalREPL, Docker, Modal тощо) |
| **Client**         | Драйвер для з'єднання з LLM API (OpenAI, Anthropic тощо)      |
| **Logger**         | Записує історію ітерацій у JSON                               |
| **VerbosePrinter** | Приємний вивід у консоль                                      |

---

## Детальна документація по функціям

---

### Core RLM

#### `class RLM`

**Назва та суть:** Головний клас RLM, який є точкою входу для всіх користувачів. Він керує всім процесом рекурсивного виклику моделей.

**Призначення:** Цей клас необхідний, щоб спростити використання RLM. Замість того, щоб налаштовувати окремо сервер, середовище виконання та логіку, користувач створює один об'єкт і викликає `completion()`.

**Технічні специфікації:**

| Параметр | Тип | Значення |
|----------|-----|----------|
| `backend` | `ClientBackend` | Назва бекенду (openai, anthropic тощо) |
| `backend_kwargs` | `dict[str, Any] | None` | Аргументи для бекенду (model_name, api_key тощо) |
| `environment` | `EnvironmentType` | Тип середовища (local, docker тощо) |
| `environment_kwargs` | `dict[str, Any] | None` | Аргументи для середовища |
| `depth` | `int` | Глибина рекурсії (поточна, 0-індексована) |
| `max_depth` | `int` | Максимальна глибина (за замовчуванням 1) |
| `max_iterations` | `int` | Максимум ітерацій (за замовчуванням 30) |
| `custom_system_prompt` | `str | None` | Користувацький системний промпт |
| `other_backends` | `list[ClientBackend] | None` | Додаткові бекенди для під-викликів |
| `persistent` | `bool` | Використовувати одне середовище між викликами |

**Як працює:**
- При ініціалізації створюється конфігурація для кожного виклику
- Під час `completion()` створюється окремий LMHandler та Environment
- Об'єкт підтримує context manager протокол (`with`)

**Ключові методи:**

---

#### `__init__(self, ...) -> None`

**Назва та суть:** Ініціалізує екземпляр RLM з заданими параметрами конфігурації.

**Призначення:** Налаштовує всі компоненти RLM перед початком роботи. Під час ініціалізації відбувається перевірка підтримки persistent режиму та логування метаданих.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `backend` | `ClientBackend` | Назва LLM бекенду |
| `backend_kwargs` | `dict | None` | Ключові аргументи для бекенду |
| `environment` | `EnvironmentType` | Тип середовища виконання |
| `environment_kwargs` | `dict | None` | Аргументи для середовища |
| `depth` | `int` | Поточна глибина рекурсії |
| `max_depth` | `int` | Максимальна глибина рекурсії |
| `max_iterations` | `int` | Максимальна кількість ітерацій |
| `custom_system_prompt` | `str | None` | Користувацький системний промпт |
| `other_backends` | `list[ClientBackend] | None` | Додаткові бекенди для під-викликів |
| `other_backend_kwargs` | `list[dict] | None` | Аргументи для додаткових бекендів |
| `logger` | `RLMLogger | None` | Логер для запису результатів |
| `verbose` | `bool` | Увімкнути/вимкнути розширений вивід |
| `persistent` | `bool` | Використовувати persistent режим |

**Повертає:** Нічого (None)

**Механіка та підфункції:**

1. **Збереження конфігурації:**
```python
self.backend = backend
self.backend_kwargs = backend_kwargs
self.environment_type = environment
self.environment_kwargs = environment_kwargs.copy() if environment_kwargs is not None else {}
```

2. **Перевірка додаткових бекендів:**
```python
if other_backends is not None:
    if len(other_backends) != 1:
        raise ValueError(
            "We currently only support one additional backend for the recursive sub-calls!"
        )
```
Це важлива перевірка: наразі RLM підтримує лише один додатковий бекенд для під-викликів.

3. **Створення системного промпту:**
```python
self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
```
Якщо користувач не надає свій промпт, використовується вбудований `RLM_SYSTEM_PROMPT`.

4. **Відображення метаданих:**
```python
if self.logger or verbose:
    metadata = RLMMetadata(
        root_model=backend_kwargs.get("model_name", "unknown"),
        max_depth=max_depth,
        max_iterations=max_iterations,
        ...
    )
    if self.logger:
        self.logger.log_metadata(metadata)
```

**Приклад використання:**

```python
# Сценарій 1: Стандартне використання
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4-turbo"},
    environment="local",
    max_iterations=20,
    verbose=True
)
# Результат: RLM екземпляр готовий до роботи з OpenAI та локальним REPL

# Сценарій 2: З рекурсивним під-бекендом
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4"},
    other_backends=["anthropic"],
    other_backend_kwargs=[{"model_name": "claude-3-5-sonnet"}],
    max_depth=1
)
# Результат: Глибина 0 використовує GPT-4, глибина 1 використовує Claude
```

---

#### `completion(self, prompt: str | dict[str, Any], root_prompt: str | None = None) -> RLMChatCompletion`

**Назва та суть:** Головний метод для виконання запиту до RLM. Це основна точка входу для отримання відповіді.

**Призначення:** Цей метод виконує повний цикл: створення середовища, генерацію промптів, ітеративне виконання коду та повернення фінальної відповіді.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `prompt` | `str | dict[str, Any]` | Текстовий запит або список повідомлень |
| `root_prompt` | `str | None` | Додатковий початковий запит для кореневої моделі |

**Повертає:** `RLMChatCompletion` — об'єкт з:
- `root_model`: назва моделі
- `prompt`: вихідний запит
- `response`: фінальна відповідь
- `usage_summary`: статистика використання (tokens)
- `execution_time`: час виконання у секундах

**Механіка та підфункції:**

```python
def completion(self, prompt: str | dict[str, Any], root_prompt: str | None = None) -> RLMChatCompletion:
    time_start = time.perf_counter()  # 1. Запуск таймера

    # 2. Якщо досягнуто максимальної глибини, використовуємо звичайний LM
    if self.depth >= self.max_depth:
        return self._fallback_answer(prompt)

    # 3. Створення контексту (LMHandler + Environment)
    with self._spawn_completion_context(prompt) as (lm_handler, environment):
        message_history = self._setup_prompt(prompt)

        # 4. Ітераційний процес
        for i in range(self.max_iterations):
            # Формування поточного промпту
            current_prompt = message_history + [
                build_user_prompt(root_prompt, i, context_count, history_count)
            ]

            # 5. Виконання одного "обороту"
            iteration: RLMIteration = self._completion_turn(
                prompt=current_prompt,
                lm_handler=lm_handler,
                environment=environment,
            )

            # 6. Пошук фінальної відповіді
            final_answer = find_final_answer(iteration.response, environment=environment)
            iteration.final_answer = final_answer

            # 7. Логування та відображення
            if self.logger:
                self.logger.log(iteration)
            self.verbose.print_iteration(iteration, i + 1)

            # 8. Якщо відповідь знайдено — вихід
            if final_answer is not None:
                return RLMChatCompletion(...)

            # 9. Оновлення історії для наступної ітерації
            new_messages = format_iteration(iteration)
            message_history.extend(new_messages)

    # 10. Якщо ітерації закінчилися без відповіді — підсумувати
    final_answer = self._default_answer(message_history, lm_handler)
    return RLMChatCompletion(...)
```

**Приклад використання:**

```python
# Сценарій 1: Успішний запит
result = rlm.completion(
    prompt="Як знайти суму чисел від 1 до 100?",
    root_prompt="Вирахуй суму арифметичної прогресії"
)
print(result.response)  # "Сума чисел від 1 до 100 дорівнює 5050"
print(result.execution_time)  # 2.35

# Сценарій 2: Рекурсивний запит з контекстом
context = "Дуже великий текст з багатьма розділами..."
result = rlm.completion(prompt=context)
```

---

#### `_completion_turn(...) -> RLMIteration`

**Назва та суть:** Виконує одну ітерацію (один "оборот") процесу RLM.

**Призначення:** Цей метод об'єднує три ключові дії: запит LLM, знаходження коду у відповіді, виконання коду в REPL.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `prompt` | `str | dict[str, Any]` | Промпт для LLM |
| `lm_handler` | `LMHandler` | Обробник LLM запитів |
| `environment` | `BaseEnv` | Середовище виконання коду |

**Повертає:** `RLMIteration` — об'єкт з:
- `prompt`: промпт який був переданий
- `response`: відповідь від LLM
- `code_blocks`: список знайдених блоків коду
- `final_answer`: знайдена фінальна відповідь (якщо є)
- `iteration_time`: час виконання

**Механіка та підфункції:**

```python
def _completion_turn(
    self,
    prompt: str | dict[str, Any],
    lm_handler: LMHandler,
    environment: BaseEnv,
) -> RLMIteration:
    iter_start = time.perf_counter()  # Початок таймера

    # 1. Запит LLM
    response = lm_handler.completion(prompt)

    # 2. Знайти всі блоки коду ```repl```
    code_block_strs = find_code_blocks(response)

    # 3. Виконати кожен блок коду
    code_blocks = []
    for code_block_str in code_block_strs:
        code_result: REPLResult = environment.execute_code(code_block_str)
        code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

    # 4. Обчислити час виконання
    iteration_time = time.perf_counter() - iter_start

    # 5. Повернути результат
    return RLMIteration(
        prompt=prompt,
        response=response,
        code_blocks=code_blocks,
        iteration_time=iteration_time,
    )
```

**Приклад:**
```
Запит → "Як знайти суму чисел?"
LLM відповідає:
```
Знайдіть суму арифметичної прогресії. Використайте формулу S = n*(a1+an)/2.

```repl
n = 100
a1 = 1
an = 100
s = n * (a1 + an) / 2
print(f"Сума: {s}")
```
```

Метод `_completion_turn`:
1. Відправляє запит LLM
2. Отримує відповідь з блоком коду ```repl```
3. Виконує код: `n = 100`, `s = n * (a1 + an) / 2`, `print(...)`
4. Повертає `RLMIteration` з відповіддю та результатом виконання

---

#### `_default_answer(...) -> str`

**Назва та суть:** Повертає фінальну відповідь, коли вичерпано кількість ітерацій.

**Призначення:** Якщо протягом `max_iterations` не знайдено фінальну відповідь, цей метод "попрошує" модель зробити остаточний висновок на основі історії.

**Механіка:**
```python
def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
    # Додаємо додатковий запит для фінальної відповіді
    current_prompt = message_history + [
        {
            "role": "assistant",
            "content": "Please provide a final answer to the user's question based on the information provided.",
        }
    ]
    response = lm_handler.completion(current_prompt)
    return response  # Повертаємо відповідь моделі
```

**Приклад:**
```
Після 30 ітерацій без FINAL(...) відповіді:
-system: "Будь ласка, дай фінальну відповідь..."
-model: "З усієї історії обговорення можна зробити висновок, що..."
```

---

#### `_fallback_answer(...) -> str`

**Назва та суть:** Коли досягнуто максимальної глибини рекурсії, RLM стає звичайним LM.

**Призначення:** Це "кінцева точки" рекурсії. Коли глибина `depth >= max_depth`, більше немає під-рівнів, тому використовується звичайний LM без REPL.

**Механіка:**
```python
def _fallback_answer(self, message: str | dict[str, Any]) -> str:
    client: BaseLM = get_client(self.backend, self.backend_kwargs)
    response = client.completion(message)
    return response
```

**Приклад сценарію:**
```
Глибина 0: RLM(backend=openai, depth=0)
  → генерує код, виконує в REPL
  → якщо треба, викликає під-рівень

Глибина 1: RLM(backend=anthropic, depth=1, max_depth=1)
  → depth >= max_depth (1 >= 1) → TRUE
  → _fallback_answer() → звичайнийAnthropic LM без REPL
```

---

#### `_spawn_completion_context(prompt) -> (lm_handler, environment)`

**Назва та суть:** Контекстний менеджер, який створює LMHandler та Environment для одного виклику completion.

**Призначення:** Забезпечує чисте виділення та звільнення ресурсів. Коли `persistent=True`, середовище повторно використовується.

**Механіка:**
```python
@contextmanager
def _spawn_completion_context(self, prompt: str | dict[str, Any]):
    # 1. Створити client та обгорнути в handler
    client: BaseLM = get_client(self.backend, self.backend_kwargs)

    # 2. Створити other_backend_client якщо потрібно
    other_backend_client: BaseLM | None = None
    if self.other_backends and self.other_backend_kwargs:
        other_backend_client = get_client(self.other_backends[0], self.other_backend_kwargs[0])

    # 3. Створити LMHandler
    lm_handler = LMHandler(client, other_backend_client=other_backend_client)

    # 4. Запустити сервер
    lm_handler.start()

    # 5. Створити Environment (persistent або новий)
    if self.persistent and self._persistent_env is not None:
        environment = self._persistent_env
        environment.update_handler_address((lm_handler.host, lm_handler.port))
        environment.add_context(prompt)
    else:
        env_kwargs = self.environment_kwargs.copy()
        env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
        env_kwargs["context_payload"] = prompt
        env_kwargs["depth"] = self.depth + 1
        environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

    try:
        yield lm_handler, environment  # Виконується блок with
    finally:
        lm_handler.stop()
        if not self.persistent and hasattr(environment, "cleanup"):
            environment.cleanup()  # Очищення ресурсів
```

**Приклад:**
```python
# Неперсистентний режим (default)
with rlm._spawn_completion_context("prompt") as (lh, env):
    # Локальний REPL створюється
    # Після with він автоматично знищується

# Персистентний режим
rlm = RLM(persistent=True, environment="local")
# Перший виклик створює REPL
# Другий виклик використовує той самий REPL
```

---

#### `close(self) -> None`

**Назва та суть:** Очищує persistent середовище при завершенні роботи.

**Призначення:** Забезпечує коректне звільнення ресурсів для persistent режиму.

**Механіка:**
```python
def close(self) -> None:
    if self._persistent_env is not None:
        if hasattr(self._persistent_env, "cleanup"):
            self._persistent_env.cleanup()  # Викликає cleanup середовища
        self._persistent_env = None
```

---

#### `_validate_persistent_environment_support(self) -> None`

**Назва та суть:** Перевіряє чи підтримує обране середовище persistent режим.

**Призначення:** Запобігає спробі використовувати persistent режим з непідтримуваними середовищами.

**Механіка:**
```python
def _validate_persistent_environment_support(self) -> None:
    # Відомі середовища, що підтримують persistence
    persistent_supported_environments = {"local"}

    if self.environment_type not in persistent_supported_environments:
        raise ValueError(
            f"persistent=True is not supported for environment type '{self.environment_type}'. "
            f"Persistent mode requires environments that implement update_handler_address(), "
            f"add_context(), and get_context_count(). "
            f"Supported environments: {sorted(persistent_supported_environments)}"
        )
```

**Приклад:**
```
RLM(persistent=True, environment="docker")
  → ValueError: persistent=True не підтримується для Docker

RLM(persistent=True, environment="local")
  → OK: LocalREPL підтримує persistence
```

---

### LM Handler та Comms

#### `class LMHandler`

**Назва та суть:** Обробник усіх LLM запитів з RLM процесу та підпроцесів середовищ.

**Призначення:** LMHandler створює socket сервер, який слухає запити від Environment та виконує LLM виклики.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `client` | `BaseLM` | Клієнт для основних запитів |
| `host` | `str` | Хост для сервера (за замовчуванням 127.0.0.1) |
| `port` | `int` | Порт для сервера (0 = автоматичний) |
| `other_backend_client` | `BaseLM | None` | Клієнт для під-викликів |

**Ключові методи:**
- `start()` — запуск socket сервера у фоновому потоці
- `stop()` — зупинка сервера
- `completion(prompt, model)` — прямий виклик LLM
- `register_client(model_name, client)` — реєстрація додаткових клієнтів
- `get_client(model, depth)` — отримання правильного клієнта
- `get_usage_summary()` — отримання статистики використання

---

#### `start(self) -> tuple[str, int]`

**Назва та суть:** Запускає socket сервер у фоновому потоці.

**Призначення:** Дозволяє середовищам (REPL) виконувати LLM запити паралельно.

**Механіка:**
```python
def start(self) -> tuple[str, int]:
    if self._server is not None:
        return self.address  # Вже запущено

    # Створити сервер
    self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
    self._server.lm_handler = self  # Передати посилання на себе

    # Запустити у фоновому потоці
    self._thread = Thread(target=self._server.serve_forever, daemon=True)
    self._thread.start()

    return self.address  # Повернути (host, port)
```

**Приклад:**
```python
handler = LMHandler(openai_client, host="127.0.0.1", port=0)
host, port = handler.start()
# Повертає: ("127.0.0.1", 54321) — порт автоматично вибраний
```

---

#### `get_client(self, model: str | None = None, depth: int = 0) -> BaseLM`

**Назва та суть:** Отримує правильний клієнт для запиту залежно від model та depth.

**Призначення:** Реалізує маршрутизацію запитів до різних моделей.

**Логіка маршрутизації:**
```python
def get_client(self, model: str | None = None, depth: int = 0) -> BaseLM:
    # 1. Якщо model заданий і існує в зареєстрованих — використовувати його
    if model and model in self.clients:
        return self.clients[model]

    # 2. Маршрутизація на основі глибини
    if depth == 1 and self.other_backend_client is not None:
        return self.other_backend_client  # Під-рівень

    # 3. За замовчуванням — основний клієнт
    return self.default_client
```

**Приклад:**
```
Глибина 0 (main RLM):
  get_client(model=None, depth=0)
  → повертає default_client (OpenAI GPT-4)

Глибина 1 (sub-call):
  get_client(model=None, depth=1)
  → повертає other_backend_client (Anthropic Claude)

Прямий виклик:
  get_client(model="claude-3-5-sonnet", depth=0)
  → повертає зареєстрований клієнт з ім'ям "claude-3-5-sonnet"
```

---

#### `get_usage_summary(self) -> UsageSummary`

**Назва та суть:** Збирає статистику використання усіх зареєстрованих клієнтів.

**Призначення:** Дає загальну картину витрат на всі LLM виклики.

**Механіка:**
```python
def get_usage_summary(self) -> UsageSummary:
    merged = {}

    # 1. Основний клієнт
    default_summary = self.default_client.get_usage_summary()
    merged.update(default_summary.model_usage_summaries)

    # 2. Під-від клієнт якщо існує
    if self.other_backend_client is not None:
        other_summary = self.other_backend_client.get_usage_summary()
        merged.update(other_summary.model_usage_summaries)

    # 3. Усі зареєстровані клієнти
    for client in self.clients.values():
        client_summary = client.get_usage_summary()
        merged.update(client_summary.model_usage_summaries)

    return UsageSummary(model_usage_summaries=merged)
```

**Приклад результату:**
```json
{
  "model_usage_summaries": {
    "gpt-4-turbo": {
      "total_calls": 5,
      "total_input_tokens": 1250,
      "total_output_tokens": 450
    },
    "claude-3-5-sonnet": {
      "total_calls": 3,
      "total_input_tokens": 800,
      "total_output_tokens": 300
    }
  }
}
```

---

#### `class LMRequestHandler(StreamRequestHandler)`

**Назва та суть:** Оброблює socket запити до LM.

**Призначення:** Кожен раз коли Environment хоче зробити LLM запит, він підключається до цього handler.

**Механіка:**
```python
class LMRequestHandler(StreamRequestHandler):
    def handle(self):
        try:
            # 1. Отримати запит (4-byte length prefix + JSON)
            request_data = socket_recv(self.connection)

            # 2. Перетворити на LMRequest
            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler

            # 3. Обробити batched або single
            if request.is_batched:
                response = self._handle_batched(request, handler)
            elif request.prompt:
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts'")

            # 4. Відправити відповідь
            self._safe_send(response)
        except Exception as e:
            response = LMResponse.error_response(str(e))
            self._safe_send(response)
```

---

#### `socket_send(sock, data: dict) -> None`

**Назва та суть:** Відправляє JSON повідомлення через socket з length prefix.

**Призначення:** Реалізує протокол комунікації між Environment та LMHandler.

**Механіка:**
```python
def socket_send(sock: socket.socket, data: dict) -> None:
    # 1. Перетворити на JSON
    payload = json.dumps(data).encode("utf-8")

    # 2. Додати 4-byte length prefix (big-endian)
    sock.sendall(struct.pack(">I", len(payload)) + payload)
```

**Приклад:**
```
Дані: {"prompt": "Hello", "model": "gpt-4"}
JSON: {"prompt": "Hello", "model": "gpt-4"} (35 bytes)
Відправлено: [00 00 00 23] + "{\"prompt\": \"Hello\", \"model\": \"gpt-4\"}"
         [length]     [payload]
```

---

#### `socket_recv(sock) -> dict`

**Назва та суть:** Отримує length-prefixed JSON повідомлення з socket.

**Призначення:** Читає повідомлення у відповідному протоколі.

**Механіка:**
```python
def socket_recv(sock: socket.socket) -> dict:
    # 1. Прочитати перші 4 байти (довжина)
    raw_len = sock.recv(4)
    if not raw_len:
        return {}  # З'єднання закрито

    # 2. Розпакувати довжину (big-endian)
    length = struct.unpack(">I", raw_len)[0]

    # 3. Прочитати payload
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("Connection closed before message complete")
        payload += chunk

    # 4. Розпакувати JSON
    return json.loads(payload.decode("utf-8"))
```

---

### Environments

#### `class BaseEnv`

**Назва та суть:** Абстрактний базовий клас для всіх середовищ виконання.

**Призначення:** Визначає загальний інтерфейс для всіх середовищ.

**Обов'язкові методи:**
- `setup()` — ініціалізація середовища
- `load_context(context_payload)` — завантаження контексту
- `execute_code(code)` — виконання коду

```python
class BaseEnv(ABC):
    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError
```

---

#### `class LocalREPL(NonIsolatedEnv)`

**Назва та суть:** Локальне Python REPL середовище з постійним namespace.

**Призначення:** Це **головне** середовище RLM. Воно виконує Python код у захищеному середовищі з доступом до контексту.

**Ключові особливості:**
- Постійний namespace (змінні зберігаються між викликами)
- Безпечні builtins (blocks eval/exec/input)
- Доступ до LLM через llm_query/llm_query_batched
- Підтримка persistent режиму

**Механіка:**

```python
class LocalREPL(NonIsolatedEnv):
    def __init__(self, lm_handler_address, context_payload, persistent, depth, **kwargs):
        # 1. Створити безпечний namespace
        self.setup()

        # 2. Завантажити контекст
        if context_payload is not None:
            self.load_context(context_payload)

        # 3. Запустити setup_code якщо є
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        # Безпечні builtins (без eval/exec/input)
        self.globals = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }

        # Додати допоміжні функції
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
```

---

#### `execute_code(self, code: str) -> REPLResult`

**Назва та суть:** Виконує Python код у захищеному namespace та повертає результат.

**Призначення:** Це **найважливіша функція** в середовищі. Вона виконує весь код, який генерує модель.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `code` | `str` | Python код для виконання |

**Повертає:** `REPLResult` з:
- `stdout`: стандартний вивід (print)
- `stderr`: помилки
- `locals`: змінні після виконання
- `execution_time`: час виконання
- `rlm_calls`: список LLM викликів зроблених під час виконання

**Механіка:**
```python
def execute_code(self, code: str) -> REPLResult:
    start_time = time.perf_counter()

    # 1. Очистити pending LLM calls
    self._pending_llm_calls = []

    # 2. Захопити stdout/stderr
    with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
        try:
            # 3. Об'єднати globals та locals
            combined = {**self.globals, **self.locals}

            # 4. Виконати код
            exec(code, combined, combined)

            # 5. Оновити locals новими змінними
            for key, value in combined.items():
                if key not in self.globals and not key.startswith("_"):
                    self.locals[key] = value

            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
        except Exception as e:
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

    # 6. Повернути результат
    return REPLResult(
        stdout=stdout,
        stderr=stderr,
        locals=self.locals.copy(),
        execution_time=time.perf_counter() - start_time,
        rlm_calls=self._pending_llm_calls.copy(),
    )
```

**Приклад:**
```
Код:
```repl
# Створити список чисел
numbers = list(range(1, 11))

# Знайти суму
total = sum(numbers)

# Вивести результат
print(f"Сума чисел 1-10: {total}")
```

Виклик execute_code(code):
  → exec(code, combined, combined)  # Виконується код
  → self.locals["numbers"] = [1,2,3,4,5,6,7,8,9,10]
  → self.locals["total"] = 55
  → stdout = "Сума чисел 1-10: 55\n"

Повертає:
REPLResult(
  stdout="Сума чисел 1-10: 55\n",
  stderr="",
  locals={"numbers": [1,2,3,4,5,6,7,8,9,10], "total": 55},
  execution_time=0.015,
  rlm_calls=[]
)
```

---

#### `llm_query(self, prompt: str, model: str | None = None) -> str`

**Назва та суть:** Функція, яка дозволяє зробити LLM запит зсередини REPL.

**Призначення:** Це **ключова функція** RLM. Вона дозволяє моделі використовувати інші LLM для аналізу великих обсягів даних.

**Механіка:**
```python
def _llm_query(self, prompt: str, model: str | None = None) -> str:
    if not self.lm_handler_address:
        return "Error: No LM handler configured"

    try:
        # 1. Створити запит
        request = LMRequest(prompt=prompt, model=model, depth=self.depth)

        # 2. Відправити через socket
        response = send_lm_request(self.lm_handler_address, request)

        # 3. Перевірити успіх
        if not response.success:
            return f"Error: {response.error}"

        # 4. Відстежити виклик
        self._pending_llm_calls.append(response.chat_completion)

        # 5. Повернути відповідь
        return response.chat_completion.response
    except Exception as e:
        return f"Error: LM query failed - {e}"
```

**Приклад:**
```
Код у REPL:
```repl
# Великий текст, який не влазить в контекст
large_text = context

# Розбити на частини
chunks = [large_text[i:i+50000] for i in range(0, len(large_text), 50000)]

# Запитати по частинам
answers = []
for chunk in chunks:
    answer = llm_query(f"Який головний зміст цього тексту? {chunk}")
    answers.append(answer)

# Об'єднати відповіді
final_summary = llm_query(f"Об'єднати наступні відповіді: {answers}")
```

Коли llm_query() викликається:
  → send_lm_request(handler_address, request)
  → socket server отримує запит
  → LMHandler.completion() викликає LLM API
  → Відповідь повертається назад
  → Виклик зареєстрований у _pending_llm_calls
```

---

#### `add_context(self, context_payload, context_index) -> int`

**Назва та суть:** Додає контекст як змінну з versioned іменем.

**Призначення:** Дозволяє мати кілька контекстів одночасно в persistent режимі.

**Механіка:**
```python
def add_context(self, context_payload: dict | list | str, context_index: int | None = None) -> int:
    if context_index is None:
        context_index = self._context_count  # Auto-increment

    var_name = f"context_{context_index}"

    # Зберегти контекст у файл
    if isinstance(context_payload, str):
        context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
        with open(context_path, "w") as f:
            f.write(context_payload)
        # Завантажити у Python
        self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
    else:
        context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
        with open(context_path, "w") as f:
            json.dump(context_payload, f)
        self.execute_code(
            f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
        )

    # Створити alias context_0 -> context
    if context_index == 0:
        self.execute_code(f"context = {var_name}")

    self._context_count = max(self._context_count, context_index + 1)
    return context_index
```

**Приклад:**
```
add_context("Перший текст")  # повертає 0, створює context_0 та context
add_context("Другий текст")  # повертає 1, створює context_1
add_context("Третій текст")  # повертає 2, створює context_2

Код у REPL може використовувати:
  context, context_0, context_1, context_2
```

---

#### `class DockerREPL(NonIsolatedEnv)`

**Назва та суть:** Середовище, яке виконує код у Docker контейнері.

**Призначення:** Надає ізоляцію від основної системи через Docker.

**Механіка:**
```python
class DockerREPL(NonIsolatedEnv):
    def setup(self):
        # 1. Запустити LLM proxy server
        self.proxy_server = HTTPServer(("127.0.0.1", 0), handler)
        self.proxy_port = self.proxy_server.server_address[1]
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()

        # 2. Запустити Docker контейнер
        result = subprocess.run([
            "docker", "run", "-d", "--rm",
            "-v", f"{self.temp_dir}:/workspace",
            "--add-host", "host.docker.internal:host-gateway",
            self.image, "tail", "-f", "/dev/null"
        ])
        self.container_id = result.stdout.strip()

        # 3. Встановити залежності
        subprocess.run([
            "docker", "exec", self.container_id, "pip", "install", "-q", "dill", "requests"
        ])

    def execute_code(self, code: str) -> REPLResult:
        # 1. Закодувати код в base64
        code_b64 = base64.b64encode(code.encode()).decode()

        # 2. Створити execution script
        script = _build_exec_script(code, self.proxy_port, self.depth)

        # 3. Виконати у контейнері
        result = subprocess.run([
            "docker", "exec", self.container_id, "python", "-c", script
        ])

        # 4. Розпакувати результат
        data = json.loads(result.stdout.strip().split("\n")[-1])
        return REPLResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            locals=data.get("locals", {}),
            ...
        )
```

**Script, який виконується у контейнері:**
```python
# Завантажити state (persistent)
state = load_state()  # з /workspace/state.dill

# Виконати код
exec(code, combined, combined)

# Зберегти змінні
save_state(combined)

# Вивести результат
print(json.dumps({"stdout": ..., "stderr": ..., "locals": ...}))
```

---

#### `class ModalREPL(IsolatedEnv)`

**Назва та суть:** Середовище на базі Modal Sandbox з broker сервером.

**Призначення:** Надає повну ізоляцію у хмарному середовищі Modal.

**Механіка:**
```python
class ModalREPL(IsolatedEnv):
    def setup(self):
        # 1. Створити Modal app
        self.app = modal.App.lookup(self.app_name, create_if_missing=True)

        # 2. Створити sandbox
        self.sandbox = modal.Sandbox.create(
            app=self.app,
            image=self.image,
            encrypted_ports=[8080]  # Для broker
        )

        # 3. Запустити broker server
        self.broker_process = self.sandbox.exec("python", "-c", _BROKER_SCRIPT)

        # 4. Отримати tunnel URL
        self.broker_url = tunnels[8080].url

        # 5. Запустити polling thread
        self.poller_thread = threading.Thread(target=self._poll_broker, daemon=True)
        self.poller_thread.start()

    def _poll_broker(self):
        """Poll broker for LLM requests and handle them."""
        while not self.poller_stop.is_set():
            # 1. Отримати pending запити
            pending = requests.get(f"{self.broker_url}/pending").json().get("pending", [])

            for item in pending:
                request_id = item["id"]
                req_data = item["request"]

                # 2. Виконати запит
                response = self._handle_llm_request(req_data)

                # 3. Відправити відповідь
                requests.post(f"{self.broker_url}/respond", json={
                    "id": request_id,
                    "response": response
                })
```

**Broker протокол:**
```
Запит від sandbox:
POST /enqueue
{"type": "single", "prompt": "...", "model": "...", "depth": 1}

Відповідь від ModalREPL:
POST /respond
{"id": "uuid", "response": {"response": "...", "error": null}}
```

---

### Clients

#### `class BaseLM`

**Назва та суть:** Абстрактний базовий клас для всіх LLM клієнтів.

**Призначення:** Визначає загальний інтерфейс для всіх LLM API.

**Обов'язкові методи:**
- `completion(prompt)` — синхронний виклик
- `acompletion(prompt)` — асинхронний виклик
- `get_usage_summary()` — статистика використання
- `get_last_usage()` — останнє використання

---

#### `class OpenAIClient(BaseLM)`

**Назва та суть:** Клієнт для OpenAI API (та сумісних API як vLLM, OpenRouter).

**Призначення:** Найпоширеніший клієнт для RLM.

**Ключові методи:**

**`completion(self, prompt) -> str`**
```python
def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
    # 1. Перетворити prompt на messages
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        messages = prompt

    # 2. Викликати OpenAI API
    response = self.client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body
    )

    # 3. Відстежити витрати
    self._track_cost(response, model)

    # 4. Повернути відповідь
    return response.choices[0].message.content
```

**`_track_cost(self, response, model)`**
```python
def _track_cost(self, response: openai.ChatCompletion, model: str):
    # Відстежити кількість викликів
    self.model_call_counts[model] += 1

    # Відстежити tokens
    usage = response.usage
    self.model_input_tokens[model] += usage.prompt_tokens
    self.model_output_tokens[model] += usage.completion_tokens
    self.model_total_tokens[model] += usage.total_tokens

    # Зберегти для get_last_usage
    self.last_prompt_tokens = usage.prompt_tokens
    self.last_completion_tokens = usage.completion_tokens
```

**Приклад використання:**
```python
client = OpenAIClient(
    api_key="sk-...",
    model_name="gpt-4-turbo"
)

response = client.completion("Яка столиця Франції?")
print(response)  # "Париж"

usage = client.get_usage_summary()
# {"gpt-4-turbo": {"total_calls": 1, "total_input_tokens": 10, "total_output_tokens": 5}}
```

---

#### `get_client(backend, backend_kwargs) -> BaseLM`

**Назва та суть:** Фабрична функція для створення правильного клієнта.

**Призначення:** Дозволяє використовувати різні LLM API без зміни коду RLM.

**Механіка:**
```python
def get_client(backend: ClientBackend, backend_kwargs: dict[str, Any]) -> BaseLM:
    if backend == "openai":
        from rlm.clients.openai import OpenAIClient
        return OpenAIClient(**backend_kwargs)

    elif backend == "anthropic":
        from rlm.clients.anthropic import AnthropicClient
        return AnthropicClient(**backend_kwargs)

    elif backend == "vllm":
        from rlm.clients.openai import OpenAIClient
        backend_kwargs.setdefault("base_url", "http://localhost:8000/v1")
        return OpenAIClient(**backend_kwargs)

    elif backend == "openrouter":
        from rlm.clients.openai import OpenAIClient
        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)

    elif backend == "gemini":
        from rlm.clients.gemini import GeminiClient
        return GeminiClient(**backend_kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}")
```

**Приклад:**
```python
# OpenAI
client = get_client("openai", {"model_name": "gpt-4"})

# vLLM (локальний)
client = get_client("vllm", {"base_url": "http://localhost:8000/v1"})

# Anthropic
client = get_client("anthropic", {"model_name": "claude-3-5-sonnet"})

# Gemini
client = get_client("gemini", {"model_name": "gemini-1.5-pro"})
```

---

#### `get_environment(environment, environment_kwargs) -> BaseEnv`

**Назва та суть:** Фабрична функція для створення правильного середовища.

**Призначення:** Дозволяє використовувати різні середовища виконання.

**Механіка:**
```python
def get_environment(
    environment: Literal["local", "modal", "docker", "daytona", "prime", "e2b"],
    environment_kwargs: dict[str, Any],
) -> BaseEnv:
    if environment == "local":
        return LocalREPL(**environment_kwargs)
    elif environment == "modal":
        from rlm.environments.modal_repl import ModalREPL
        return ModalREPL(**environment_kwargs)
    elif environment == "docker":
        from rlm.environments.docker_repl import DockerREPL
        return DockerREPL(**environment_kwargs)
    # ... інші
```

---

### Parsing Utilities

#### `find_code_blocks(text: str) -> list[str]`

**Назва та суть:** Знаходить всі блоки коду ```repl``` у відповіді від LLM.

**Призначення:** Після того як модель генерує відповідь, ця функція витягує всі блоки коду для виконання.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `text` | `str` | Відповідь від LLM |

**Повертає:** `list[str]` — список знайдених блоків коду

**Механіка:**
```python
def find_code_blocks(text: str) -> list[str]:
    # Шаблон для ```repl ... ```
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    # Знайти всі збіги
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results
```

**Приклад:**
```python
text = """Я знайду суму чисел.

```repl
numbers = list(range(1, 11))
total = sum(numbers)
print(f"Сума: {total}")
```

Це дасть нам результат."""

blocks = find_code_blocks(text)
# ["numbers = list(range(1, 11))\ntotal = sum(numbers)\nprint(f\"Сума: {total}\")"]

# Якщо є кілька блоків:
text2 = """```repl
a = 1
```

```repl
b = 2
```
"""
blocks2 = find_code_blocks(text2)
# ["a = 1", "b = 2"]
```

---

#### `find_final_answer(text: str, environment: BaseEnv | None) -> str | None`

**Назва та суть:** Знаходить фінальну відповідь з `FINAL(...)` або `FINAL_VAR(...)`.

**Призначення:** Ця функція **ключова** для роботи RLM. Вона визначає, коли модель завершила роботу.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `text` | `str` | Відповідь від LLM |
| `environment` | `BaseEnv | None` | Опціонально для виконання FINAL_VAR |

**Повертає:** `str | None` — фінальна відповідь або `None`

**Механіка:**
```python
def find_final_answer(text: str, environment: BaseEnv | None = None) -> str | None:
    # 1. Спочатку перевірити FINAL_VAR (змінна)
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is not None:
            # Виконати print(FINAL_VAR(...)) щоб отримати значення
            result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            final_answer = result.stdout.strip()
            if final_answer == "":
                final_answer = result.stderr.strip() or ""
            return final_answer
        return None  # Без environment не можна виконати

    # 2. Потім перевірити FINAL (прямий текст)
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None
```

**Приклад:**
```python
# Сценарій 1: FINAL з прямим текстом
text = """Аналогія: Літній сезон це як ...
FINAL(Відкриті вікна, морожене, відпочинок на пляжі)"""

answer = find_final_answer(text)
# "Відкриті вікна, морожене, відпочинок на пляжі"

# Сценарій 2: FINAL_VAR з змінною
text2 = """Я створив багато змінних, але фінальний результат зберіг у buffer.

FINAL_VAR(buffer)"""

# Без environment:
find_final_answer(text2)  # None

# З environment:
# 1. execute_code("print(FINAL_VAR(buffer))")
# 2. buffer = "головна відповідь"
# 3. stdout = "головна відповідь\n"
find_final_answer(text2, environment)  # "головна відповідь"
```

---







#### `format_iteration(iteration, max_character_length) -> list[dict[str, str]]`

**Назва та суть:** Форматує ітерацію для додавання до історії повідомлень.

**Призначення:** Після виконання ітерації, результат потрібно форматувати для наступного запиту.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `iteration` | `RLMIteration` | Результат ітерації |
| `max_character_length` | `int` | Максимальна довжина результату (default 20000) |

**Повертає:** `list[dict[str, str]]` — список повідомлень для наступного промпту

**Механіка:**
```python
def format_iteration(
    iteration: RLMIteration,
    max_character_length: int = 20000
) -> list[dict[str, str]]:
    # 1. Додати відповідь моделі
    messages = [{"role": "assistant", "content": iteration.response}]

    # 2. Для кожного блоку коду
    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)

        # 3. Обрізати якщо занадто довго
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        # 4. Додати повідомлення про виконання
        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)

    return messages
```

**Приклад:**
```python
# Input RLMIteration:
iteration = RLMIteration(
    response="Сума дорівнює 55",
    code_blocks=[
        CodeBlock(
            code="numbers = list(range(1, 11))\nprint(sum(numbers))",
            result=REPLResult(
                stdout="55\n",
                stderr="",
                locals={"numbers": [1,2,3,4,5,6,7,8,9,10]},
                ...
            )
        )
    ]
)

# Output:
[
    {"role": "assistant", "content": "Сума дорівнює 55"},
    {
        "role": "user",
        "content": "Code executed:\n```python\nnumbers = list(range(1, 11))\nprint(sum(numbers))\n```\n\nREPL output:\n55\nREPL variables: ['numbers']"
    }
]

# Наступний промпт отримує ці повідомлення + новий user prompt
```

---

#### `format_execution_result(result: REPLResult) -> str`

**Назва та суть:** Форматує результат виконання для відображення.

**Призначення:** Перетворює `REPLResult` у зручний для читання рядок.

**Механіка:**
```python
def format_execution_result(result: REPLResult) -> str:
    result_parts = []

    # 1. stdout
    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    # 2. stderr
    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # 3. Важливі змінні (не внутрішні)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in ["__builtins__", "__name__", "__doc__"]:
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"
```

**Приклад:**
```python
result = REPLResult(
    stdout="55\n",
    stderr="",
    locals={"numbers": [1,2,3,4,5,6,7,8,9,10], "total": 55},
    execution_time=0.01,
    rlm_calls=[]
)

formatted = format_execution_result(result)
# "\n55\n\nREPL variables: ['numbers', 'total']"
```

---

### Prompts Utilities

#### `build_rlm_system_prompt(system_prompt, query_metadata) -> list[dict[str, str]]`

**Назва та суть:** Створює початковий системний промпт з метаданими про контекст.

**Призначення:** Надає моделі інформацію про структуру контексту.

**Механіка:**
```python
def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
) -> list[dict[str, str]]:
    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # Якщо більше 100 частин, обрізати
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    # Створити мета-промпт
    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]
```

**Приклад:**
```python
metadata = QueryMetadata("Дуже довгий текст...")
# context_lengths = [10000], context_total_length = 10000, context_type = "str"

prompt = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, metadata)
# [
#   {"role": "system", "content": "You are tasked with answering a query..."},
#   {"role": "assistant", "content": "Your context is a str with 10000 total characters, and is broken up into chunks of char lengths: [10000]."}
# ]
```

---

#### `build_user_prompt(...) -> dict[str, str]`

**Назва та суть:** Створює user prompt для поточної ітерації.

**Призначення:** Надає модель інструкції щодо наступних дій.

**Механіка:**
```python
def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        # Перша ітерація
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        # Наступні ітерації
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Інформувати про кількість контекстів
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Інформувати про історію
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}
```

**Приклад:**
```python
# Перша ітерація
build_user_prompt(root_prompt="Яка сума?", iteration=0, context_count=1, history_count=0)
# {
#   "role": "user",
#   "content": "You have not interacted with the REPL environment yet...\n\nThink step-by-step... original prompt: "Яка сума?"..."
# }

# Друга ітерація
build_user_prompt(root_prompt="Яка сума?", iteration=1, context_count=1, history_count=0)
# {
#   "role": "user",
#   "content": "The history before is your previous interactions...\n\nThink step-by-step... original prompt: "Яка сума?"..."
# }

# З кількома контекстами
build_user_prompt(iteration=0, context_count=3, history_count=0)
# "...\n\nNote: You have 3 contexts available (context_0 through context_2)."
```

---

### Logger та Verbose

#### `class RLMLogger`

**Назва та суть:** Логер для запису ітерацій у JSON-lines файл.

**Призначення:** Дозволяє аналізувати роботу RLM пізніше.

**Ключові методи:**

**`log_metadata(self, metadata: RLMMetadata)`**
```python
def log_metadata(self, metadata: RLMMetadata):
    """Log RLM metadata as the first entry in the file."""
    if self._metadata_logged:
        return  # Вже записано

    entry = {
        "type": "metadata",
        "timestamp": datetime.now().isoformat(),
        **metadata.to_dict(),
    }

    with open(self.log_file_path, "a") as f:
        json.dump(entry, f)
        f.write("\n")

    self._metadata_logged = True
```

**`log(self, iteration: RLMIteration)`**
```python
def log(self, iteration: RLMIteration):
    """Log an RLMIteration to the file."""
    self._iteration_count += 1

    entry = {
        "type": "iteration",
        "iteration": self._iteration_count,
        "timestamp": datetime.now().isoformat(),
        **iteration.to_dict(),
    }

    with open(self.log_file_path, "a") as f:
        json.dump(entry, f)
        f.write("\n")
```

**Приклад файлу:**
```json
{"type": "metadata", "timestamp": "2024-01-01T12:00:00", "root_model": "gpt-4", "max_depth": 1, ...}
{"type": "iteration", "iteration": 1, "timestamp": "2024-01-01T12:00:01", "prompt": "...", "response": "...", ...}
{"type": "iteration", "iteration": 2, "timestamp": "2024-01-01T12:00:02", "prompt": "...", "response": "...", ...}
```

---

#### `class VerbosePrinter`

**Назва та суть:** Приємний вивід у консоль з використанням rich.

**Призначення:** Допомагає відслідковувати процес в реальному часі.

**Ключові методи:**

**`print_iteration(self, iteration: RLMIteration, iteration_num: int)`**
```python
def print_iteration(self, iteration: RLMIteration, iteration_num: int):
    """Print a complete iteration including response and code executions."""
    if not self.enabled:
        return

    # 1. Вивести заголовок ітерації
    self.print_iteration_start(iteration_num)

    # 2. Вивести відповідь LLM
    self.print_completion(iteration.response, iteration.iteration_time)

    # 3. Вивести кожен блок коду
    for code_block in iteration.code_blocks:
        self.print_code_execution(code_block)

        # 4. Вивести під-виклики
        for call in code_block.result.rlm_calls:
            self.print_subcall(
                model=call.root_model,
                prompt_preview=call.prompt,
                response_preview=call.response,
                execution_time=call.execution_time,
            )
```

**`print_final_answer(self, answer)`**
```python
def print_final_answer(self, answer: Any):
    """Print the final answer."""
    # Вивести відповідь у красивій рамці
    title = Text()
    title.append("★ ", style=STYLE_WARNING)
    title.append("Final Answer", style=Style(color=COLORS["warning"], bold=True))

    answer_text = Text(_to_str(answer), style=STYLE_TEXT)

    panel = Panel(answer_text, title=title, ...)
    self.console.print(panel)
```

---

### Types & Dataclasses

#### `class ModelUsageSummary`

**Назва та суть:** Записує використання моделі (кількість викликів та tokens).

**Механіка:**
```python
@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int

    def to_dict(self):
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
```

---

#### `class UsageSummary`

**Назва та суть:** Загальна статистика по усіх моделях.

**Механіка:**
```python
@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    def to_dict(self):
        return {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }
```

---

#### `class REPLResult`

**Назва та суть:** Результат виконання коду в REPL.

**Механіка:**
```python
@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    rlm_calls: list[RLMChatCompletion]

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, ...)"
```

---

#### `class RLMIteration`

**Назва та суть:** Результат однієї ітерації RLM.

**Механіка:**
```python
@dataclass
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final_answer: str | None = None
    iteration_time: float | None = None
```

---

#### `class RLMChatCompletion`

**Назва та суть:** Записує один LLM виклик з метаданими.

**Механіка:**
```python
@dataclass
class RLMChatCompletion:
    root_model: str
    prompt: str | dict[str, Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float
```

---

#### `class QueryMetadata`

**Назва та суть:** Метадані про запит (довжина контексту тощо).

**Механіка:**
```python
@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
        # ... інші типи
        self.context_total_length = sum(self.context_lengths)
```

---

## Додаткові приклади

### Повний приклад роботи RLM

```python
from rlm import RLM

# 1. Створити екземпляр
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4-turbo"},
    environment="local",
    max_iterations=30,
    verbose=True,
)

# 2. Виконати запит
result = rlm.completion(
    prompt="Як обчислити факторіал числа 10? Надішліть код для розрахунку.",
    root_prompt="Обчислити факторіал 10"
)

# 3. Перевірити результат
print(f"Відповідь: {result.response}")
print(f"Час виконання: {result.execution_time:.2f}s")
print(f"Використані моделі: {result.usage_summary.to_dict()}")

# 4. Прибрати ресурси
rlm.close()
```

### Приклад із рекурсією

```python
from rlm import RLM

# Основна модель: GPT-4
# Під-модель: Claude для під-викликів
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4"},
    other_backends=["anthropic"],
    other_backend_kwargs=[{"model_name": "claude-3-5-sonnet"}],
    max_depth=1,
    max_iterations=20,
)

result = rlm.completion(
    prompt="Аналогія: Літній сезон це як...",
    root_prompt="Створи креативну аналогію для літнього сезону"
)

print(result.response)
```

### Приклад із persistent режимом

```python
from rlm import RLM

# Persistent режим для багатоходових розмов
rlm = RLM(
    backend="openai",
    environment="local",
    persistent=True,
)

# Перший запит
result1 = rlm.completion("Як змінити пароль у Linux?")
print(result1.response)

# Другий запит - контекст зберігається
result2 = rlm.completion("А якщо я забув пароль?")
print(result2.response)

# Прибрати ресурси
rlm.close()
```

---

## Корисні посилання

- **Головний клас:** `rlm/core/rlm.py` - основна логіка RLM
- **LM Handler:** `rlm/core/lm_handler.py` - socket сервер для LLM запитів
- **Local REPL:** `rlm/environments/local_repl.py` - локальне середовище
- **Clients:** `rlm/clients/` - API клієнти
- **Parsing:** `rlm/utils/parsing.py` - парсинг відповідей

---

## Висновок

RLM — це потужний фреймворк для рекурсивного використання LLM. Основні концепції:

1. **RLM** - головний клас для роботи
2. **LMHandler** - socket сервер для LLM запитів
3. **Environment** - середовище виконання коду (LocalREPL, Docker, Modal)
4. **Client** - API клієнт для LLM (OpenAI, Anthropic тощо)
5. **Parsing** - знаходження коду та фінальних відповідей

Кожен компонент має чітке призначення та легко розширюється.

---

**Кінець документації**

---

## Додаткові Clients та Environments

### Додаткові Client класи

---

#### `class AnthropicClient(BaseLM)`

**Назва та суть:** Клієнт для Anthropic API (Claude моделі).

**Призначення:** Дозволяє використовувати моделі Claude від Anthropic як бекенд для RLM.

**Ключові особливості:**
- Використовує `anthropic.Anthropic` client
- Підтримує system prompt через загальну структуру запиту
- Відстежує використання tokens (input, output)

**Механіка:**
```python
def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = self.client.messages.create(
        model=model,
        messages=messages,
        system=self.system_prompt or "",
        extra_headers=self.extra_headers,
    )

    self._track_cost(response, model)
    return response.content[0].text
```

**Приклад:**
```python
client = AnthropicClient(
    api_key="sk-ant-...",
    model_name="claude-3-5-sonnet-20240620"
)

response = client.completion("Що таке Python?")
print(response)
```

---

#### `class GeminiClient(BaseLM)`

**Назва та суть:** Клієнт для Google Gemini API.

**Призначення:** Дозволяє використовувати Gemini моделі від Google.

**Механіка:**
```python
def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = self.client.chat.send_message(prompt)
    self._track_cost(response, model)
    return response.text
```

---

#### `class LiteLLMClient(BaseLM)`

**Назва та суть:** Універсальний клієнт через LiteLLM для підключення до будь-якого LLM API.

**Призначення:** LiteLLM - це універсальний API, який об'єднує багато LLM провайдерів.

**Механіка:**
```python
def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=self.api_key,
        base_url=self.base_url,
    )

    self._track_cost(response, model)
    return response.choices[0].message.content
```

---

#### `class AzureOpenAIClient(BaseLM)`

**Назва та суть:** Клієнт для Azure OpenAI API.

**Призначення:** Дозволяє використовувати Azure OpenAI сервіси.

**Механіка:**
```python
def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = self.client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"api-version": "2023-03-15-preview"},
    )

    self._track_cost(response, model)
    return response.choices[0].message.content
```

---

#### `class PortkeyClient(BaseLM)`

**Назва та суть:** Клієнт для Portkey API - універсального gateway для LLM.

**Призначення:** Portkey дозволяє централизовано керувати доступом до різних LLM провайдерів.

**Механіка:**
```python
def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = self.client.chat.completions.create(
        model=model,
        messages=messages,
    )

    self._track_cost(response, model)
    return response.choices[0].message.content
```

---

### Додаткові Environment класи

---

#### `class DaytonaREPL(IsolatedEnv)`

**Назва та суть:** Середовище на базі Daytona Sandbox з broker сервером.

**Призначення:** Daytona — це open-source платформа для керування sandbox середовищами.

**Ключові особливості:**
- Використовує Daytona API для створення sandbox
- Підтримує preview URLs для broker комунікації
- Використовує Flask broker server у sandbox
- Підтримує state persistence через dill

**Механіка:**
```python
class DaytonaREPL(IsolatedEnv):
    def setup(self):
        config = DaytonaConfig(api_key=self.api_key, target=self.target)
        self.daytona = Daytona(config)

        resources = Resources(cpu=self.cpu, memory=self.memory, disk=self.disk)
        params = CreateSandboxFromImageParams(
            name=self.name,
            image=self.image,
            resources=resources,
        )
        self.sandbox = self.daytona.create(params)

        self.sandbox.fs.upload_file(_BROKER_SCRIPT.encode(), "broker_server.py")
        self.sandbox.process.create_session(self.broker_session_id)
        self.sandbox.process.execute_session_command(
            self.broker_session_id,
            SessionExecuteRequest(command="python broker_server.py", var_async=True),
        )

        preview_info = self.sandbox.get_preview_link(self.BROKER_PORT)
        self.broker_url = preview_info.url

        self.poller_thread = threading.Thread(target=self._poll_broker, daemon=True)
        self.poller_thread.start()
```

---

#### `class PrimeREPL(IsolatedEnv)`

**Назва та суть:** Середовище на базі Prime Intellect Sandboxes.

**Призначення:** Prime Intellect надає ізольовані sandbox середовища для виконання коду.

**Ключові особливості:**
- Використовує Prime Sandbox SDK
- Підтримує port exposure через `sandboxes.expose()`
- Встановлює залежності через apt/pip
- Слідує тому ж broker pattern як ModalREPL

---

#### `class E2BREPL(IsolatedEnv)`

**Назва та суть:** Середовище на базі E2B Code Interpreter Sandboxes.

**Призначення:** E2B — це хмарна платформа для безпечного виконання коду Python в ізольованих sandbox.

**Ключові особливості:**
- Використовує `e2b_code_interpreter.Sandbox`
- Підтримує background processes
- Використовує public URL для broker комунікації
- Port 8889 для broker (8888 зайнятий Jupyter)

---

## Детальніша документація по ключовим функціям

### `execute_code(self, code: str) -> REPLResult` (LocalREPL)

**Назва та суть:** Це **найважливіша функція** в усьому проекті RLM. Вона виконує Python-код у захищеному середовищі та повертає результат.

**Призначення:** Ця функція дозволяє моделі виконувати код безпечно, зберігаючи стан між викликами. Без неї RLM не зміг би працювати.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `code` | `str` | Python-код, який потрібно виконати |

**Повертає:** `REPLResult` з наступними полями:
- `stdout`: звичайний вивід (від `print()`)
- `stderr`: помилки (від `traceback.print_exc()`)
- `locals`: словник змінних після виконання
- `execution_time`: час виконання в секундах
- `rlm_calls`: список LLM-викликів зроблених під час виконання

**Механіка (крок за кроком):**

```python
def execute_code(self, code: str) -> REPLResult:
    start_time = time.perf_counter()  # 1. Початок таймера

    # 2. Очистити виклики LLM з попереднього виконання
    self._pending_llm_calls = []

    # 3. Захопити stdout та stderr
    with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
        try:
            # 4. Об'єднати глобальні та локальні змінні
            combined = {**self.globals, **self.locals}

            # 5. ВИКОНАТИ КОД - найважливіший момент!
            exec(code, combined, combined)

            # 6. Оновити локальні змінні новими значеннями
            for key, value in combined.items():
                if key not in self.globals and not key.startswith("_"):
                    self.locals[key] = value

            # 7. Отримати вивід
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()

        except Exception as e:
            # 8. Обробити помилку
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

    # 9. Повернути результат
    return REPLResult(
        stdout=stdout,
        stderr=stderr,
        locals=self.locals.copy(),
        execution_time=time.perf_counter() - start_time,
        rlm_calls=self._pending_llm_calls.copy(),
    )
```

**Чому це складно:**
1. Потрібно захопити вивід `print()` без зміни коду моделі
2. Потрібно зберегти стан (змінні) між викликами
3. Потрібно безпечно виконати довільний код (без eval/exec/input)
4. Потрібно відстежити виклики `llm_query()` зсередини коду

**Приклад 1 - Успішне виконання:**
```
Вхідний код:
```python
numbers = list(range(1, 11))
total = sum(numbers)
print(f"Сума: {total}")
```

Виклик: execute_code(code)

Повертає:
REPLResult(
    stdout="Сума: 55\n",
    stderr="",
    locals={"numbers": [1,2,3,4,5,6,7,8,9,10], "total": 55},
    execution_time=0.012,
    rlm_calls=[]
)
```

**Приклад 2 - З помилкою:**
```
Вхідний код:
```python
result = 10 / 0
```

Виклик: execute_code(code)

Повертає:
REPLResult(
    stdout="",
    stderr="ZeroDivisionError: division by zero",
    locals={},
    execution_time=0.008,
    rlm_calls=[]
)
```

**Приклад 3 - З llm_query():**
```
Вхідний код:
```python
answer = llm_query(f"Який зміст цього тексту? {large_text[:5000]}")
```

Виклик: execute_code(code)

Повертає: REPLResult(...) з rlm_calls з деталями виклику
```

---

### `find_final_answer(text: str, environment: BaseEnv | None) -> str | None`

**Назва та суть:** Ця функція **визначає, коли модель завершила роботу**. Вона шукає маркери `FINAL(...)` або `FINAL_VAR(...)` у відповіді моделі.

**Призначення:** Без цієї функції RLM не знатиме, коли зупинити ітераційний процес.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `text` | `str` | Відповідь від LLM |
| `environment` | `BaseEnv | None` | Опціонально для `FINAL_VAR` |

**Повертає:** `str | None` — фінальна відповідь або `None`

**Механіка:**
```python
def find_final_answer(text: str, environment: BaseEnv | None = None) -> str | None:
    # 1. Спочатку перевірити FINAL_VAR (змінна, яку потрібно виконати)
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is not None:
            # Виконати print(FINAL_VAR(...))
            result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            final_answer = result.stdout.strip()
            if final_answer == "":
                final_answer = result.stderr.strip() or ""
            return final_answer
        return None  # Без environment не можна виконати

    # 2. Потім перевірити FINAL (прямий текст)
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None
```

**Приклад 1 - FINAL з прямим текстом:**
```
Вхідний текст:
Аналогія: Літній сезон це як відкриті вікна, морожене та відпочинок на пляжі.
FINAL(Відкриті вікна, морожене, відпочинок на пляжі)

Виклик: find_final_answer(text)

Повертає: "Відкриті вікна, морожене, відпочинок на пляжі"
```

**Приклад 2 - FINAL_VAR з змінною:**
```
Вхідний текст:
Я створив багато змінних, але фінальний результат зберіг у buffer.
FINAL_VAR(buffer)

Без environment: find_final_answer(text)  # Повертає None
З environment: environment.execute_code("buffer = 'головна відповідь'")
find_final_answer(text, environment)  # Повертає "головна відповідь"
```

**Приклад 3 - Немає фінальної відповіді:**
```
Вхідний текст: Я знайшов суму чисел 1-10: 55
Виклик: find_final_answer(text)
Повертає: None
```

---

### `socket_send(sock, data: dict) -> None`

**Назва та суть:** Ця функція реалізує протокол комунікації між Environment та LMHandler через socket.

**Призначення:** Без цієї функції sandbox-середовища (Docker, Modal, Daytona тощо) не змогли би відправляти LLM-запити.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `sock` | `socket.socket` | Socket для відправки |
| `data` | `dict` | JSON-дані для відправки |

**Повертає:** Нічого (None)

**Механіка (протокол length-prefix):**
```python
def socket_send(sock: socket.socket, data: dict) -> None:
    # 1. Перетворити dict на JSON-рядок в UTF-8
    payload = json.dumps(data).encode("utf-8")
    # 2. Додати 4-byte length prefix (big-endian)
    sock.sendall(struct.pack(">I", len(payload)) + payload)
```

**Чому length-prefix?**
Без цього протоколу socket не знає, коли повідомлення закінчилося.

**Приклад:**
```
Дані: {"prompt": "Hello", "model": "gpt-4"}
JSON: 35 байт
Відправлено на socket: [00 00 00 23] + "{"prompt": "Hello", "model": "gpt-4"}"
 [length]              [payload]
```

---

### `socket_recv(sock) -> dict`

**Назва та суть:** Ця функція читає length-prefixed JSON повідомлення з socket.

**Призначення:** Це протилежна функція до `socket_send()`. Вона дозволяє LMHandler читати запити від Environment.

**Технічні специфікації:**

| Аргумент | Тип | Опис |
|----------|-----|------|
| `sock` | `socket.socket` | Socket для читання |

**Повертає:** `dict` — розпакований JSON об'єкт

**Механіка:**
```python
def socket_recv(sock: socket.socket) -> dict:
    # 1. Прочитати перші 4 байти (довжина повідомлення)
    raw_len = sock.recv(4)
    if not raw_len:
        return {}
    # 2. Розпакувати довжину (big-endian)
    length = struct.unpack(">I", raw_len)[0]
    # 3. Прочитати payload (може прийти частинами)
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("Connection closed before message complete")
        payload += chunk
    # 4. Розпакувати JSON
    return json.loads(payload.decode("utf-8"))
```

**Приклад:**
```
Socket отримує байти: [00 00 00 23] + "{"prompt": "Hello", "model": "gpt-4"}"
socket_recv() читає:
1. raw_len = sock.recv(4) → [00 00 00 23]
2. length = struct.unpack(">I", raw_len)[0] → 35
3. payload = sock.recv(35) → b'{"prompt": "Hello", "model": "gpt-4"}'
4. return json.loads(payload.decode("utf-8")) → {"prompt": "Hello", "model": "gpt-4"}
```

---

## Заключення

RLM — це потужний фреймворк для рекурсивного використання LLM. Основні концепції:

1. **RLM** - головний клас для роботи
2. **LMHandler** - socket сервер для LLM запитів
3. **Environment** - середовище виконання коду (LocalREPL, Docker, Modal, Daytona, Prime, E2B)
4. **Client** - API клієнт для LLM (OpenAI, Anthropic, Gemini, LiteLLM, Azure, Portkey)
5. **Parsing** - знаходження коду та фінальних відповідей
6. **Socket Protocol** - length-prefix протокол для міжпроцесної комунікації

Кожен компонент має чітке призначення та легко розширюється.

---

**Кінець документації**
