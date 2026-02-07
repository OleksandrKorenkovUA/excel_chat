from pipelines.spreadsheet_analyst_pipeline import _should_reject_router_hit_for_read


def test_router_hit_guard_rejects_mutating_intent_on_read() -> None:
    code = "df = df.drop(index=[0])\nCOMMIT_DF = True\nresult = {'status': 'updated'}\n"
    meta = {"intent_id": "drop_rows_by_position"}
    assert _should_reject_router_hit_for_read(False, code, meta) is True


def test_router_hit_guard_accepts_read_intent() -> None:
    code = "result = df.at[76, 'Ціна_UAH']\n"
    meta = {"intent_id": "read_cell_value"}
    assert _should_reject_router_hit_for_read(False, code, meta) is False
