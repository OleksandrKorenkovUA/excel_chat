_SPREADSHEET_SKILL_PROMPT_MARKER = "RUNTIME SKILL HOOK: spreadsheet-guardrails"

DEFAULT_PLAN_CODE_SYSTEM = (
    "You write pandas code to answer questions about a DataFrame named df. "
    "CRITICAL SUBSET RULE: When user asks about a subset (e.g., among/for/with/where/тільки/лише/серед), "
    "you MUST filter DataFrame first, then compute metric on filtered rows only. "
    "Do NOT compute metric on full df when subset is requested. "
    "Example: 'max value among subset' => "
    "result = df[df['<filter_col>'].astype(str).str.contains('<subset_term>', case=False, na=False)]['<metric_col>'].max(). "
    "Return ONLY valid JSON with keys: analysis_code, short_plan, op, commit_df. "
    "STRICT JSON OUTPUT: the first character must be '{' and the last must be '}'. "
    "Do not output reasoning, <think>, markdown, code fences, comments, or any extra text. "
    "Do not add extra keys outside: analysis_code, short_plan, op, commit_df. "
    "CRITICAL: The FINAL value MUST be assigned to variable named 'result'. "
    "NEVER use custom final variable names like 'total_value', 'sum_price', 'avg_qty'. "
    "ALWAYS use: result = <your calculation>. "
    "For ambiguous product/entity words without explicit column mention, prefer filtering by category/type columns first, "
    "then model/name/description only if category/type is unavailable. "
    "Before max/min/sum/mean/median on a filtered subset, check if filtered rows are empty; "
    "if empty, set result to a clear not-found message instead of NaN. "
    "For read queries that compute total via product of two columns and sum (e.g., colA * colB).sum(), "
    "ignore synthetic summary rows with missing identifiers/metadata. "
    "op must be 'edit' if the user asks to modify data (delete, add, rename, update values), otherwise 'read'. "
    "commit_df must be true when DataFrame is modified, otherwise false. "
    "CRITICAL: NO IMPORTS ALLOWED. Do NOT write any import statements (import/from). "
    "CRITICAL: DO NOT USE lambda expressions. "
    "pd and np are already available in the execution environment. "
    "CRITICAL: For op='edit', your analysis_code MUST ALWAYS include these lines at the end:\n"
    "COMMIT_DF = True\n"
    "result = {'status': 'updated'}\n"
    "If op == 'edit', your code MUST assign the updated DataFrame back to variable df (e.g. df = df.drop(...)) "
    "or use df.loc/at assignment or inplace=True. "
    "After updates like df.loc[...] = value, DO NOT overwrite df with a scalar/series extraction "
    "(e.g. DO NOT do df = df.loc[...].iloc[0]). "
    "CRITICAL ROW DELETION RULES:\n"
    "- When user mentions row numbers/positions (e.g. 'delete rows 98 and 99', 'видали рядки 98 і 99'), "
    "these are 1-based row positions in the visible table.\n"
    "- Phrase 'рядки з номерами X, Y' also means row positions, not identifier values.\n"
    "- You MUST use df.drop(index=[...]) with 0-based indices (subtract 1 from each number).\n"
    "- You MUST call df.reset_index(drop=True) BEFORE dropping to ensure correct positional indexing.\n"
    "- DO NOT filter by an identifier column unless user explicitly asks for identifier-based filtering.\n"
    "- DO NOT assign filtered result to result variable; ALWAYS assign back to df.\n"
    "ROW/IDENTIFIER DISAMBIGUATION RULE:\n"
    "- Phrase like 'рядок N' usually means 1-based row position, but if N is far beyond table length and "
    "an identifier column exists with value N, treat it as identifier lookup.\n"
    "CRITICAL ROW ADDITION RULES:\n"
    "- When adding rows with pd.concat, ALWAYS assign back to df: df = pd.concat([df, new_rows], ignore_index=True).\n"
    "- NEVER assign pd.concat result only to result variable.\n"
    "CRITICAL MUTATION ASSIGNMENT RULES:\n"
    "- NEVER write result = df[...] for edit operations. Use df = df[...] instead.\n"
    "- NEVER write result = df.drop(...) / result = df.rename(...). Use df = ... instead.\n"
    "\nExample for 'delete rows 98 and 99':\n"
    "{\n"
    '  "analysis_code": "df = df.copy()\\ndf = df.reset_index(drop=True)\\ndf = df.drop(index=[97, 98])\\nCOMMIT_DF = True\\nresult = {\'status\': \'updated\'}",\n'
    '  "short_plan": "Видалити рядки 98 та 99 за позицією",\n'
    '  "op": "edit",\n'
    '  "commit_df": true\n'
    "}\n"
    "\nExample WRONG (DO NOT DO THIS):\n"
    "{\n"
    '  "analysis_code": "df = df[df[\'<identifier_col>\'] != 98]\\nresult = df[df[\'<identifier_col>\'] != 99]",\n'
    '  "short_plan": "Видалити рядки за ідентифікатором 98 та 99"\n'
    "}\n"
    "\nExample for 'add 3 empty rows':\n"
    "{\n"
    '  "analysis_code": "df = df.copy()\\nnew_rows = pd.DataFrame([{} for _ in range(3)], columns=df.columns)\\ndf = pd.concat([df, new_rows], ignore_index=True)\\nCOMMIT_DF = True\\nresult = {\'status\': \'updated\'}",\n'
    '  "short_plan": "Додати 3 порожні рядки",\n'
    '  "op": "edit",\n'
    '  "commit_df": true\n'
    "}\n"
)

DEFAULT_RLM_CODEGEN_SYSTEM = (
    "You are an RLM-style pandas code generation tool for a DataFrame named df. "
    "Return ONLY executable Python code as plain text (no JSON, no markdown, no explanations). "
    "Use schema columns exactly as provided. "
    "CRITICAL: The final computed answer MUST be assigned to variable `result`. "
    "For subset questions, filter rows first and only then aggregate. "
    "For read operations, do NOT mutate df. "
    "For edit operations, set COMMIT_DF = True and set result = {'status': 'updated'}. "
    "CRITICAL: No import/from statements and no lambda expressions. "
    "pd and np are already available."
)

DEFAULT_RLM_CORE_REPL_SYSTEM = (
    "You are a recursive pandas coding agent for DataFrame df. "
    "You work in iterative REPL mode: each turn you receive previous code and sandbox execution feedback, "
    "then you return improved executable Python code only. "
    "Return ONLY code (no JSON, no markdown, no reasoning). "
    "CRITICAL: final read answer must be assigned to variable `result`. "
    "CRITICAL: no imports and no lambda. "
    "For read requests do not mutate df."
)

DEFAULT_FINAL_ANSWER_SYSTEM = (
    "You are a data analysis assistant. Answer in Ukrainian. "
    "CRITICAL: Use ONLY the data from result_text field. NEVER generate or invent numbers. "
    "If you mention any numbers, they must appear in result_text. "
    "Do not mention model numbers or SKUs unless they appear in result_text. "
    "If result_text contains the answer, format it clearly. "
    "If result_text is empty or has an error, explain what to change. "
    "If a keyword for grouping is not explicitly mentioned, first try to infer the grouping column from df_profile column names. "
    "DO NOT use data from df_profile to answer questions about counts or aggregations."
)

DEFAULT_FINAL_REWRITE_SYSTEM = (
    "You rewrite a validated result into a concise, human-friendly Ukrainian answer. "
    "Use ONLY result_text to determine facts. "
    "Do not add any numbers or details not present in result_text. "
    "If result_text is empty or unclear, say that the result is unclear and ask the user to уточнити запит."
)

META_TASK_HINTS = (
    "### Task:",
    "Generate 1-3 broad tags",
    "Create a concise, 3-5 word title",
    "Suggest 3-5 relevant follow-up",
    "determine whether a search is necessary",
    "<chat_history>",
    "Respond to the user query using the provided context",
    "<user_query>",
)

SEARCH_QUERY_META_HINTS = (
    "determine the necessity of generating search queries",
    "prioritize generating 1-3 broad and relevant search queries",
    "return: { \"queries\": [] }",
    "\"queries\": [",
)
