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


def test_auto_detect_commit_df_filter_assignment() -> None:
    code = "df = df[df['A'] > 0]\nresult = int(df.shape[0])\n"
    assert _auto_detect_commit(code) is True


def test_auto_detect_commit_allows_df_copy_assignment_in_read_code() -> None:
    code = "df = df.copy(deep=False)\nresult = int(df['A'].mean())\n"
    assert _auto_detect_commit(code) is False


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


def test_finalize_allows_read_when_df_is_reassigned_to_filtered_subset() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="Яка максимальна ціна на товар Миша",
        analysis_code="df = df[df['Категорія'].astype(str).str.contains('Миша', case=False, na=False)]\nresult = df['Ціна_UAH'].max()\n",
        op="read",
        commit_df=False,
    )
    assert committed is False
    assert err is None
    assert code.startswith("df = df.copy(deep=False)\n_df_read = df.copy(deep=False)\n")
    assert "result = _df_read['Ціна_UAH'].max()" in code


def test_finalize_strips_think_sections_before_read_rebinding_guard() -> None:
    code, committed, err = _finalize_code_for_sandbox(
        question="Яка максимальна ціна на товар Миша",
        analysis_code=(
            "<think>\n"
            "I will filter then aggregate.\n"
            "</think>\n"
            "df = df[df['Категорія'].astype(str).str.contains('Миша', case=False, na=False)]\n"
            "result = df['Ціна_UAH'].max()\n"
        ),
        op="read",
        commit_df=False,
    )
    assert committed is False
    assert err is None
    assert "<think>" not in code
    assert "_df_read = _df_read[_df_read['Категорія'].astype(str).str.contains('Миша', case=False, na=False)]" in code


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


def test_detect_single_sum_does_not_match_product_expression() -> None:
    col = _detect_result_single_column_sum("result = df['Ціна_UAH'] * df['Кількість'].sum()\n")
    assert col is None


def test_detect_sum_of_product_does_not_match_right_side_sum_expression() -> None:
    cols = _detect_result_sum_of_product_columns("result = df['Ціна_UAH'] * df['Кількість'].sum()\n")
    assert cols is None


def test_rewrite_sum_of_product_code_applies_valid_mask() -> None:
    code = "result = (df['Ціна_UAH'] * df['Кількість']).sum()\n"
    rewritten = _rewrite_sum_of_product_code(
        code,
        {"columns": ["ID", "Ціна_UAH", "Кількість", "Статус"]},
    )
    assert "_valid = _left.notna() & _right.notna()" in rewritten
    assert "result = float((_left[_valid] * _right[_valid]).sum())" in rewritten
    assert "_meta_cols" not in rewritten
    assert "_id_like" not in rewritten
    assert "pd.to_numeric" in rewritten


def test_rewrite_sum_of_product_code_skips_for_non_sum_aggregation_intent() -> None:
    code = "result = (df['Ціна_UAH'] * df['Кількість']).sum()\n"
    rewritten = _rewrite_sum_of_product_code(
        code,
        {"columns": ["ID", "Категорія", "Ціна_UAH", "Кількість"]},
        question="Яка середня ціна ноутбуків?",
    )
    assert rewritten == code


def test_rewrite_sum_of_product_code_skips_when_code_has_filter() -> None:
    code = (
        "_m = df['Категорія'].astype(str).str.contains('Ноутбук', case=False, na=False)\n"
        "result = (df['Ціна_UAH'] * df['Кількість']).sum()\n"
    )
    rewritten = _rewrite_sum_of_product_code(
        code,
        {"columns": ["ID", "Категорія", "Ціна_UAH", "Кількість"]},
        question="Яка загальна вартість ноутбуків?",
    )
    assert rewritten == code


def test_rewrite_sum_of_product_code_does_not_exclude_rows_by_id_or_meta() -> None:
    code = "result = (df['Ціна_UAH'] * df['Кількість']).sum()\n"
    rewritten = _rewrite_sum_of_product_code(
        code,
        {"columns": ["ID", "Категорія", "Ціна_UAH", "Кількість", "Статус"]},
        question="Яка загальна вартість усіх товарів на складі?",
    )
    df = pd.DataFrame(
        {
            "ID": [1001.0, np.nan],
            "Категорія": ["Ноутбук", np.nan],
            "Ціна_UAH": [100.0, 200.0],
            "Кількість": [2, 3],
            "Статус": ["В наявності", np.nan],
        }
    )
    env = {"df": df, "pd": pd, "np": np}
    exec(rewritten, env, env)
    assert float(env["result"]) == 800.0


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


def test_rewrite_single_column_sum_skips_derived_column_assignment() -> None:
    code = (
        "df['Total_Cost'] = pd.to_numeric(df['Ціна_UAH'], errors='coerce') * "
        "pd.to_numeric(df['Кількість'], errors='coerce')\n"
        "result = df['Total_Cost'].sum()\n"
    )
    profile = {"columns": ["ID", "Ціна_UAH", "Кількість"]}
    rewritten = _rewrite_single_column_sum_code(code, profile)
    assert rewritten == code

    df = pd.DataFrame({"Ціна_UAH": [10.0, 20.0], "Кількість": [2, 3]})
    env = {"df": df, "pd": pd, "np": np}
    exec(rewritten, env, env)
    assert float(env["result"]) == 80.0


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


def test_is_total_value_scalar_question_false_for_average_intent() -> None:
    profile = {"columns": ["Ціна_UAH", "Кількість", "Категорія"]}
    assert _is_total_value_scalar_question("Середня ціна ноутбуків", profile) is False


def test_final_answer_returns_short_message_for_empty_result() -> None:
    pipeline = Pipeline()
    answer = pipeline._final_answer(
        question="Яка максимальна ціна серед мишей?",
        profile={"columns": ["Категорія", "Ціна_UAH"]},
        plan="",
        code="result = None\n",
        edit_expected=False,
        run_status="ok",
        stdout="",
        result_text="",
        result_meta={},
        mutation_summary={},
        mutation_flags={},
        error="",
    )
    assert answer == "За умовою запиту не знайдено значень."


def test_final_answer_nan_uses_filter_hint_instead_of_raw_nan() -> None:
    pipeline = Pipeline()
    answer = pipeline._final_answer(
        question="Яка максимальна ціна на товар Миша?",
        profile={"columns": ["Модель", "Ціна_UAH"]},
        plan="",
        code='filtered_df = df[df["Модель"] == "Миша"]\nresult = filtered_df["Ціна_UAH"].max()\n',
        edit_expected=False,
        run_status="ok",
        stdout="",
        result_text="nan",
        result_meta={},
        mutation_summary={},
        mutation_flags={},
        error="",
    )
    assert "Не знайдено рядків для фільтра" in answer
    assert "Модель = Миша" in answer


def test_stats_shortcut_does_not_replace_average_with_total_inventory() -> None:
    from pipelines.spreadsheet_analyst_pipeline import _stats_shortcut_code

    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Кількість"],
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Кількість": "int64"},
    }
    out = _stats_shortcut_code("Середня ціна ноутбуків", profile)
    assert out is not None
    code, _plan = out
    assert ".mean()" in code or "result['mean']" in code
    assert "(df['Ціна_UAH'] * df['Кількість']).sum()" not in code


def test_lookup_shortcut_status_semantic_uses_contains_pattern() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Категорія", "Статус", "Кількість"],
        "dtypes": {"Категорія": "str", "Статус": "str", "Кількість": "int64"},
    }
    slots = {
        "mode": "lookup",
        "filters": [
            {"column": "Категорія", "op": "contains", "value": "Миша"},
            {"column": "Статус", "op": "eq", "value": "на складі"},
        ],
        "output_columns": ["Кількість"],
    }
    out = pipeline._lookup_shortcut_code_from_slots("Яка кількість мишей на складі", profile, slots)
    assert out is not None
    code, _plan = out
    assert "_pat1" in code
    assert "склад" in code
    assert ".str.contains(_pat" in code


def test_final_answer_success_path_does_not_call_llm_and_formats_result_text() -> None:
    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            raise AssertionError("LLM should not be called on strict success path")

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeLLM:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    pipeline = Pipeline()
    pipeline._llm = _FakeLLM()

    answer = pipeline._final_answer(
        question="Виручка по категоріях (Ціна × Кількість)",
        profile={"columns": ["Категорія", "Виручка_UAH"]},
        plan="retrieval_intent:groupby_agg",
        code="result = _work.groupby(_group_col)['_metric'].sum().reset_index(name='Виручка_UAH')\n",
        edit_expected=False,
        run_status="ok",
        stdout="",
        result_text='[{"Категорія":"Ноутбук","Виручка_UAH":1000.0}]',
        result_meta={},
        mutation_summary={},
        mutation_flags={},
        error="",
    )
    assert "| Категорія | Виручка_UAH |" in answer
    assert "| Ноутбук | 1000.0 |" in answer
    assert pipeline._llm.chat.completions.calls == 0
