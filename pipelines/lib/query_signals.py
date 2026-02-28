import re


ROUTER_FILTER_CONTEXT_RE = re.compile(
    r"\b(серед|among|within|where|де|тільки|only|лише|having)\b"
    r"|(?:що|які)\s+мають"
    r"|(?:with|for)\s+[a-zа-яіїєґ0-9_]{2,}",
    re.I,
)

EXPLICIT_SUBSET_FILTER_WORD_RE = re.compile(
    r"\b(серед|among|within|where|де|тільки|only|лише|having)\b"
    r"|(?:що|які)\s+мають",
    re.I,
)

AVAILABILITY_FILTER_CUE_RE = re.compile(
    r"\b(на\s+склад\w*|в\s+наявн\w*|наявн\w*|в\s+запас\w*|запас\w*|залишк\w*|in\s*stock|available|inventory|warehouse|доступн\w*)\b",
    re.I,
)

ROUTER_METRIC_CUE_RE = re.compile(
    r"\b("
    r"max(?:imum)?|minimum|min|mean|average|avg|median|sum|total|count|"
    r"макс\w*|мін\w*|середн\w*|сума|підсум\w*|кільк\w*|скільк\w*"
    r")\b",
    re.I,
)

ROUTER_ENTITY_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")

GROUPING_CUE_RE = re.compile(
    r"\b("
    r"group\s*by|"
    r"by\s+\w+|"
    r"by\s+(?:each|every)\s+\w+|"
    r"for\s+(?:each|every)\s+\w+|"
    r"by\s+(?:category|categories|brand|brands|model|models|type|types)|"
    r"per\s+\w+|"
    r"по\s+(?:кож\w+\s+)?(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*|статус\w*)|"
    r"за\s+(?:кож\w+\s+)?(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*|статус\w*)"
    r")\b",
    re.I,
)

METRIC_CONTEXT_RE = re.compile(
    r"\b(value|price|amount|total|sum|qty|count|number|rows?|records?|items?|products?|sales?|revenue|profit)\b"
    r"|\b(значенн\w*|цін\w*|варт\w*|сум\w*|кільк\w*|рядк\w*|запис\w*|елемент\w*|товар\w*)\b",
    re.I,
)

METRIC_PATTERNS = {
    "mean": r"\b(mean|average|avg|середн\w*)\b",
    "min": r"\b(min|мін(ім(ум|ал\w*)?)?)\b|\bminimum\b(?=\s+\S+)",
    "max": r"\b(max|макс(имум|имал\w*)?)\b|\bmaximum\b(?=\s+\S+)",
    "sum": r"\b(sum|сума|підсум\w*)\b",
    "median": r"\b(median|медіан\w*)\b",
}

COUNT_CONTEXT_RE = re.compile(
    r"\b(items?|products?|rows?|records?|entries|values?|brands?|categories|orders?|customers?)\b"
    r"|\b(товар\w*|рядк\w*|запис\w*|елемент\w*|бренд\w*|категор\w*|замовлен\w*|клієнт\w*)\b",
    re.I,
)

COUNT_WORD_RE = re.compile(r"\b(count|кільк\w*)\b", re.I)
COUNT_NUMBER_OF_RE = re.compile(r"\bnumber\s+of\s+([a-z]+)\b", re.I)
COUNT_QTY_RE = re.compile(r"\bqty\b", re.I)


def has_router_metric_cue(text: str) -> bool:
    return bool(ROUTER_METRIC_CUE_RE.search(text or ""))


def has_router_filter_context_cue(text: str) -> bool:
    return bool(ROUTER_FILTER_CONTEXT_RE.search(text or ""))


def has_explicit_subset_filter_words(text: str) -> bool:
    return bool(EXPLICIT_SUBSET_FILTER_WORD_RE.search(text or ""))


def has_availability_filter_cue(text: str) -> bool:
    return bool(AVAILABILITY_FILTER_CUE_RE.search(text or ""))


def has_grouping_cue(text: str) -> bool:
    return bool(GROUPING_CUE_RE.search(text or ""))


def has_router_entity_token(text: str) -> bool:
    return bool(ROUTER_ENTITY_TOKEN_RE.search(text or ""))
