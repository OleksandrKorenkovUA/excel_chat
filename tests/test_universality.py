from pipelines.spreadsheet_analyst_pipeline import (
    AGG_COL_PLACEHOLDER,
    GROUP_COL_PLACEHOLDER,
    SUM_COL_PLACEHOLDER,
    TOP_N_PLACEHOLDER,
    Pipeline,
    _choose_column_from_question,
    _classify_columns_by_role,
    _find_column_in_text,
    _find_columns_in_text,
    _template_shortcut_code,
)


def test_choose_column_from_question_semantic_not_price_only() -> None:
    profile = {
        "columns": ["ID", "Кількість", "Вага_кг", "Стан"],
        "dtypes": {"ID": "int64", "Кількість": "int64", "Вага_кг": "float64", "Стан": "object"},
    }
    col = _choose_column_from_question("Яка вага в рядку 7?", profile)
    assert col == "Вага_кг"


def test_template_shortcut_generic_row_value_lookup() -> None:
    profile = {"columns": ["ID", "Кількість", "Вага_кг"]}
    q = "Яке значення ваги в рядку 7?"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_row_num = 7" in code
    assert "_col = 'Вага_кг'" in code


def test_template_shortcut_availability_count() -> None:
    profile = {"columns": ["ID", "Назва", "Статус наявності"]}
    q = "Перерахуй уважно кількість товарів зі статусом в наявності"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_col = '_SHORTCUT_COL_'" in code
    assert "_status = df[_col].astype(str).str.strip().str.lower()" in code
    assert "result = int(df.loc[_in & ~_out].shape[0])" in code


def test_template_shortcut_availability_count_without_status_column_name() -> None:
    profile = {
        "columns": ["ID", "Назва", "Склад"],
        "preview": [
            {"ID": 1, "Назва": "A", "Склад": "в наявності"},
            {"ID": 2, "Назва": "B", "Склад": "немає"},
        ],
    }
    q = "Скільки товарів доступно на складі?"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_col = '_SHORTCUT_COL_'" in code
    assert "_mode = 'in'" in code
    assert "result = int(df.loc[_in & ~_out].shape[0])" in code


def test_template_shortcut_availability_out_of_stock_count() -> None:
    profile = {"columns": ["ID", "Наявність"]}
    q = "Скільки товарів немає в наявності?"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_mode = 'out'" in code
    assert "result = int(df.loc[_out & ~_in].shape[0])" in code


def test_template_shortcut_availability_has_no_lambda() -> None:
    profile = {"columns": ["ID", "Статус"]}
    q = "Порахуй кількість товарів, які є в наявності"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "lambda" not in code


def test_template_shortcut_lowercase_columns_has_no_lambda() -> None:
    profile = {"columns": ["ID", "Назва"]}
    q = "Зроби назви колонок у нижній регістр"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "rename(columns=lambda" not in code
    assert "df.columns = [str(x).lower() for x in df.columns]" in code


def test_template_shortcut_fill_numeric_mean_has_no_lambda() -> None:
    profile = {"columns": ["ID", "Ціна", "Кількість"]}
    q = "замін усіх числових на mean"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "apply(lambda" not in code
    assert "fillna(df[num_cols].mean())" in code


def test_find_column_preserves_original_spacing() -> None:
    cols = ["import ", "status"]
    picked = _find_column_in_text("покажи import", cols)
    assert picked == "import "


def test_find_columns_in_text_semantic_multiple() -> None:
    cols = ["Категорія товару", "Кількість_шт", "Ціна_UAH"]
    found = _find_columns_in_text("Покажи топ категорій за загальною кількістю", cols)
    assert "Категорія товару" in found
    assert "Кількість_шт" in found


def test_classify_columns_by_role_prefers_categorical_plus_numeric() -> None:
    roles = _classify_columns_by_role(
        "Які топ-3 категорії мають найбільшу загальну кількість товарів?",
        ["Категорія", "Кількість"],
        {"dtypes": {"Категорія": "object", "Кількість": "int64"}},
    )
    assert roles["group_by"] == "Категорія"
    assert roles["aggregate"] == "Кількість"


def test_resolve_shortcut_placeholders_groupby_sum_topn_llm_first() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_columns_by_role_for_shortcut = lambda _q, _p: {  # type: ignore[method-assign]
        "group_by": "Категорія",
        "aggregate": "Кількість",
        "top_n": 3,
    }
    pipeline._llm_pick_column_for_shortcut = lambda _q, _p: None  # type: ignore[method-assign]
    profile = {"columns": ["ID", "Категорія", "Кількість"], "dtypes": {"Кількість": "int64"}}
    code = (
        f"result = df.groupby('{GROUP_COL_PLACEHOLDER}')['{SUM_COL_PLACEHOLDER}'].sum()"
        f".sort_values(ascending=False).head({TOP_N_PLACEHOLDER})\n"
    )
    plan = f"Топ по {GROUP_COL_PLACEHOLDER} і {AGG_COL_PLACEHOLDER}"
    resolved_code, resolved_plan = pipeline._resolve_shortcut_placeholders(
        code, plan, "Які топ-3 категорії за кількістю?", profile
    )
    assert "groupby('Категорія')" in resolved_code
    assert "['Кількість']" in resolved_code
    assert ".head(3)" in resolved_code
    assert "Категорія" in resolved_plan
    assert "Кількість" in resolved_plan
