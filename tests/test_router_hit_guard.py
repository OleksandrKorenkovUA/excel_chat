from pipelines.spreadsheet_analyst_pipeline import _should_reject_router_hit_for_read


def test_router_hit_guard_rejects_mutating_intent_on_read() -> None:
    code = "df = df.drop(index=[0])\nCOMMIT_DF = True\nresult = {'status': 'updated'}\n"
    meta = {"intent_id": "drop_rows_by_position"}
    assert _should_reject_router_hit_for_read(False, code, meta) is True


def test_router_hit_guard_accepts_read_intent() -> None:
    code = "result = df.at[76, 'Ціна_UAH']\n"
    meta = {"intent_id": "read_cell_value"}
    assert _should_reject_router_hit_for_read(False, code, meta) is False


def test_router_hit_guard_accepts_filter_result_for_filter_context_question() -> None:
    code = "result = df[df['Статус'].astype(str).str.lower() == str('закінчується').lower()]\n"
    meta = {"intent_id": "filter_equals"}
    assert _should_reject_router_hit_for_read(False, code, meta, 'Товари зі статусом "Закінчується"') is False


def test_router_hit_guard_rejects_total_inventory_value_when_code_sums_unit_price_only() -> None:
    code = (
        "_col_name = 'Ціна_UAH'\n"
        "_metric = 'sum'\n"
        "if _col_name not in df.columns:\n"
        "    result = None\n"
        "else:\n"
        "    _col = df[_col_name]\n"
        "    _num = _col.astype(float)\n"
        "    _v = _num.sum()\n"
        "    result = None if pd.isna(_v) else float(_v)\n"
    )
    meta = {"intent_id": "stats_aggregation"}
    profile = {
        "columns": ["Ціна_UAH", "Кількість", "Статус"],
        "dtypes": {"Ціна_UAH": "float64", "Кількість": "int64", "Статус": "str"},
    }
    question = "Яка загальна вартість усіх товарів на складі?"
    assert _should_reject_router_hit_for_read(False, code, meta, question, profile) is True
