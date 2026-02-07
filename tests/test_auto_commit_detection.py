from pipelines.spreadsheet_analyst_pipeline import Pipeline, _auto_detect_commit, _finalize_code_for_sandbox


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
