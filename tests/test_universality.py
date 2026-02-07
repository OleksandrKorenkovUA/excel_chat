from pipelines.spreadsheet_analyst_pipeline import _choose_column_from_question, _template_shortcut_code


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
