from pipelines.spreadsheet_analyst_pipeline import (
    Pipeline,
    _auto_detect_commit,
    _detect_result_sum_of_product_columns,
    _detect_result_single_column_sum,
    _ensure_result_variable,
    _finalize_code_for_sandbox,
    _is_total_value_scalar_question,
    _rewrite_single_column_sum_code,
    _rewrite_sum_of_product_code,
)
import numpy as np
import pandas as pd


def test_auto_detect_commit_df_assignment_mutating_method() -> None:
    code = "df = df.drop(columns=['A'])\nresult = {'ok': True}\n"
    assert _auto_detect_commit(code) is True


def test_auto_detect_commit_cell_assignment() -> None:
    code = "df.loc[df['A'] > 10, 'B'] = 0\nresult = 'done'\n"
    assert _auto_detect_commit(code) is True


def test_auto_detect_commit_inplace_true() -> None:
    code = "df.rename(columns={'A': 'a'}, inplace=True)\nresult = 'done'\n"
    assert _auto_detect_commit(code) is True


def test_auto_detect_commit_read_only_code() -> None:
    code = "result = int(df['A'].mean())\n"
    assert _auto_detect_commit(code) is False


def test_finalize_blocks_commit_for_read_with_mutation() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="покажи середнє значення",
        analysis_code="df = df.drop(columns=['A'])\nresult = 'ok'\n",
        op="read",
        commit_df=False,
    )
    assert committed is False
    assert err is not None
    assert "COMMIT_DF = True" not in code


def test_finalize_keeps_read_copy_for_non_mutation() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="покажи середнє значення",
        analysis_code="result = int(df['A'].mean())\n",
        op="read",
        commit_df=False,
    )
    assert err is None
    assert committed is False
    assert code.startswith("df = df.copy(deep=False)")
    assert "COMMIT_DF = True" not in code


def test_finalize_reclassifies_read_with_mutation_when_question_is_edit() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="зміни значення в колонці B",
        analysis_code="df.loc[df['A'] > 10, 'B'] = 0\nresult = 'done'\n",
        op="read",
        commit_df=False,
    )
    assert err is None
    assert committed is True
    assert "COMMIT_DF = True" in code


def test_finalize_degrades_invalid_edit_code_to_read() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="зміни значення в колонці B",
        analysis_code="result = 'done'\n",
        op="edit",
        commit_df=None,
    )
    assert err is None
    assert committed is False
    assert code.startswith("df = df.copy(deep=False)")
    assert "COMMIT_DF = True" not in code


def test_has_meaningful_mutation_true_on_committed_flag() -> None:
    pipeline = Pipeline()
    assert pipeline._has_meaningful_mutation({}, {"committed": True}) is True


def test_ensure_result_variable_autofixes_last_assignment() -> None:
    code = "tmp = 1\nfinal_value = (df['Ціна_UAH'] * df['Кількість']).sum()\n"
    fixed, was_fixed, has_result = _ensure_result_variable(code)
    assert was_fixed is True
    assert has_result is True
    assert "result = (df['Ціна_UAH'] * df['Кількість']).sum()" in fixed


def test_finalize_returns_error_when_read_has_no_assignment() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="Яка загальна вартість?",
        analysis_code="(df['Ціна_UAH'] * df['Кількість']).sum()\n",
        op="read",
        commit_df=False,
    )
    assert committed is False
    assert err is not None
    assert "missing_result_assignment" in err
    assert "WARNING: missing result assignment" in code


def test_detect_result_sum_of_product_columns() -> None:
    cols = _detect_result_sum_of_product_columns("result = (df['Ціна_UAH'] * df['Кількість']).sum()\n")
    assert cols == ("Ціна_UAH", "Кількість")


def test_detect_result_single_column_sum() -> None:
    col = _detect_result_single_column_sum("result = df['Кількість'].sum()\n")
    assert col == "Кількість"


def test_rewrite_sum_of_product_code_applies_valid_mask() -> None:
    code = "result = (df['Ціна_UAH'] * df['Кількість']).sum()\n"
    rewritten = _rewrite_sum_of_product_code(
        code,
        {"columns": ["ID", "Ціна_UAH", "Кількість", "Статус"]},
    )
    assert "_valid = pd.Series(True, index=df.index)" in rewritten
    assert "_id_like" in rewritten
    assert "result = float((_left * _right).sum())" in rewritten
    assert "pd.to_numeric" not in rewritten


def test_rewrite_single_column_sum_excludes_summary_rows() -> None:
    code = "result = df['Кількість'].sum()\n"
    profile = {"columns": ["ID", "Категорія", "Бренд", "Кількість", "Статус"]}
    rewritten = _rewrite_single_column_sum_code(code, profile)
    assert "single_sum_guard" in rewritten
    assert "result = float(_values.sum())" in rewritten

    df = pd.DataFrame(
        {
            "ID": [1001.0, 1002.0, np.nan],
            "Категорія": ["A", "B", np.nan],
            "Бренд": ["X", "Y", np.nan],
            "Кількість": [400, 467, 867],
            "Статус": ["В наявності", "В наявності", np.nan],
        }
    )
    env = {"df": df, "pd": pd, "np": np}
    exec(rewritten, env, env)
    assert float(env["result"]) == 867.0


def test_deterministic_answer_for_inventory_total_not_labeled_as_column() -> None:
    pipeline = Pipeline()
    answer = pipeline._deterministic_answer(
        "Яка загальна вартість усіх товарів на складі (Кількість × Ціна_UAH)?",
        "7485383.0",
        {"columns": ["Ціна_UAH", "Кількість"]},
    )
    assert answer is not None
    assert answer.startswith("Загальна вартість")
    assert "Кількість —" not in answer


def test_is_total_value_scalar_question_variant_phrase() -> None:
    profile = {"columns": ["Ціна_UAH", "Кількість", "Категорія"]}
    assert _is_total_value_scalar_question("Скільки всього коштує склад?", profile) is True
