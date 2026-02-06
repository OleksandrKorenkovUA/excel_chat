from pipelines.spreadsheet_analyst_pipeline import _auto_detect_commit, _finalize_code_for_sandbox


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


def test_finalize_auto_adds_commit_for_read_with_mutation() -> None:
    code, committed = _finalize_code_for_sandbox(
        question="покажи середнє значення",
        analysis_code="df = df.drop(columns=['A'])\nresult = 'ok'\n",
        op="read",
        commit_df=False,
    )
    assert committed is True
    assert "COMMIT_DF = True" in code


def test_finalize_keeps_read_copy_for_non_mutation() -> None:
    code, committed = _finalize_code_for_sandbox(
        question="покажи середнє значення",
        analysis_code="result = int(df['A'].mean())\n",
        op="read",
        commit_df=False,
    )
    assert committed is False
    assert code.startswith("df = df.copy()")
    assert "COMMIT_DF = True" not in code
