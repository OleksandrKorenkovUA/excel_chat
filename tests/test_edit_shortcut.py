from pipelines.spreadsheet_analyst_pipeline import _edit_shortcut_code, _parse_set_value


def test_parse_set_value_extracts_numeric_with_currency_and_tail() -> None:
    text = "Зміни ціну в рядку 77 на 10 000 грн (якщо треба) та онови таблицю."
    assert _parse_set_value(text) == "10 000"


def test_parse_set_value_handles_status_with_typographic_quotes() -> None:
    text = "Змін статус товару ID 1003 на статус “На складі”"
    assert _parse_set_value(text) == "На складі"


def test_edit_shortcut_handles_row_update_without_cell_word() -> None:
    profile = {"columns": ["ID", "Ціна_UAH", "Назва"]}
    q = "Зміни значення ціни в рядку 77 на 10 000 грн та онови таблицю."
    shortcut = _edit_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "df.at[76, 'Ціна_UAH'] = 10000.0" in code
    assert "COMMIT_DF = True" in code


def test_edit_shortcut_id_update_handles_float_id_and_clean_status_value() -> None:
    profile = {"columns": ["ID", "Статус", "Назва"]}
    q = "Змін статус товару ID 1003 на статус “На складі”"
    shortcut = _edit_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_id_col_norm = _id_col.str.replace(r'\\.0+$', '', regex=True)" in code
    assert "_mask = (_id_col == _id_target) | (_id_col_norm == _id_target)" in code
    assert "df.loc[_mask, 'Статус'] = 'На складі'" in code
