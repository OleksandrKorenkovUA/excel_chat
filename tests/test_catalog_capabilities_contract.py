import json
from pathlib import Path


CATALOG_PATHS = [
    Path("pipelines/catalog.json"),
    Path("sandbox_service/catalog.json"),
]

ALLOWED_CAPABILITY_KEYS = {
    "filters",
    "prefilter_before_groupby",
    "metric_types",
    "requires_numeric_target",
}
ALLOWED_FILTER_OPS = {
    "equals",
    "contains",
    "gt",
    "lt",
    "gte",
    "lte",
    "startswith",
    "endswith",
}
ALLOWED_METRIC_TYPES = {
    "count",
    "numeric",
    "product",
}

# Incremental rollout guard: legacy intents are allowed to miss capabilities,
# but new/changed intents must include the capabilities contract.
LEGACY_INTENTS_WITHOUT_CAPABILITIES = {
    "stats_shape",
    "head_tail",
    "stats_nulls",
    "unique_nunique",
    "stats_aggregation",
    "filter_equals",
    "sort_by_column",
    "select_columns",
    "drop_columns",
    "rename_column",
    "drop_rows_by_position",
    "drop_duplicates",
    "export_csv",
    "stats_describe",
    "filter_contains",
    "filtered_metric_aggregation",
    "keyword_search_count",
    "filter_range_numeric",
}


def _load_catalog(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_catalog_capabilities_contract() -> None:
    for path in CATALOG_PATHS:
        catalog = _load_catalog(path)
        intents = catalog.get("intents") or []
        assert intents, f"{path} has no intents"

        ids: list[str] = []
        missing_caps: set[str] = set()
        for intent in intents:
            intent_id = str(intent.get("id") or "").strip()
            assert intent_id, f"{path} has intent without id"
            ids.append(intent_id)

            caps = intent.get("capabilities")
            if caps is None:
                missing_caps.add(intent_id)
                continue

            assert isinstance(caps, dict), f"{path} intent '{intent_id}' capabilities must be object"
            keys = set(caps.keys())
            unknown = keys - ALLOWED_CAPABILITY_KEYS
            missing = ALLOWED_CAPABILITY_KEYS - keys
            assert not unknown, f"{path} intent '{intent_id}' has unknown capability keys: {sorted(unknown)}"
            assert not missing, f"{path} intent '{intent_id}' missing capability keys: {sorted(missing)}"

            filters = caps.get("filters")
            assert isinstance(filters, list), f"{path} intent '{intent_id}' capabilities.filters must be list"
            filter_values = {str(v).strip().lower() for v in filters if str(v or "").strip()}
            assert filter_values <= ALLOWED_FILTER_OPS, (
                f"{path} intent '{intent_id}' has invalid filters ops: {sorted(filter_values - ALLOWED_FILTER_OPS)}"
            )

            metric_types = caps.get("metric_types")
            assert isinstance(metric_types, list), (
                f"{path} intent '{intent_id}' capabilities.metric_types must be list"
            )
            metric_values = {str(v).strip().lower() for v in metric_types if str(v or "").strip()}
            assert metric_values <= ALLOWED_METRIC_TYPES, (
                f"{path} intent '{intent_id}' has invalid metric_types: {sorted(metric_values - ALLOWED_METRIC_TYPES)}"
            )

            assert isinstance(caps.get("prefilter_before_groupby"), bool), (
                f"{path} intent '{intent_id}' capabilities.prefilter_before_groupby must be bool"
            )
            assert isinstance(caps.get("requires_numeric_target"), bool), (
                f"{path} intent '{intent_id}' capabilities.requires_numeric_target must be bool"
            )

        assert len(ids) == len(set(ids)), f"{path} has duplicate intent ids"
        unexpected_missing = missing_caps - LEGACY_INTENTS_WITHOUT_CAPABILITIES
        assert not unexpected_missing, (
            f"{path} has intents without capabilities outside legacy allowlist: {sorted(unexpected_missing)}"
        )


def test_catalog_capabilities_sync_between_pipeline_and_runtime() -> None:
    pipeline_catalog = _load_catalog(Path("pipelines/catalog.json"))
    runtime_catalog = _load_catalog(Path("sandbox_service/catalog.json"))

    pipeline_intents = {str(i.get("id") or ""): i for i in (pipeline_catalog.get("intents") or [])}
    runtime_intents = {str(i.get("id") or ""): i for i in (runtime_catalog.get("intents") or [])}

    assert set(pipeline_intents.keys()) == set(runtime_intents.keys()), "Intent ids differ between catalogs"

    for intent_id in sorted(pipeline_intents.keys()):
        pipeline_caps = pipeline_intents[intent_id].get("capabilities")
        runtime_caps = runtime_intents[intent_id].get("capabilities")
        assert pipeline_caps == runtime_caps, f"Capabilities mismatch for intent '{intent_id}'"
