import pytest
import pandas as pd

from sandbox_service.main import _ast_guard, _run_code


def test_ast_guard_allows_safe_pd_converters() -> None:
    _ast_guard("result = pd.to_numeric(df['x'], errors='coerce')")
    _ast_guard("result = pd.to_datetime(df['ts'], errors='coerce')")
    _ast_guard("result = pd.to_timedelta(df['dur'], errors='coerce')")


def test_ast_guard_blocks_pd_to_pickle() -> None:
    with pytest.raises(ValueError, match="forbidden_pandas_io"):
        _ast_guard("pd.to_pickle(df, '/tmp/out.pkl')")


def test_run_code_does_not_fail_with_keyerror_import() -> None:
    df = pd.DataFrame({"Кількість": [1, 2, 3]})
    code = (
        "df = df.copy(deep=False)\n"
        "_col = df['Кількість']\n"
        "_dtype = str(_col.dtype).lower()\n"
        "if _dtype.startswith(('int', 'float', 'uint')):\n"
        "    _num = _col.astype(float)\n"
        "else:\n"
        "    _num = pd.to_numeric(_col, errors='coerce')\n"
        "result = int(_num.count())\n"
    )
    status, stdout, result_text, _meta, err, *_rest = _run_code(
        code,
        df,
        timeout_s=10,
        preview_rows=10,
        max_cell_chars=120,
        max_stdout_chars=1000,
        max_result_chars=1000,
    )
    assert status == "ok"
    assert err == ""
    assert result_text.strip() == "3"
