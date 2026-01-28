from typing import Any, Dict, Optional, Tuple


class ShortcutRouterConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class ShortcutRouter:
    def __init__(self, config: Optional[ShortcutRouterConfig] = None, llm_json: Any = None) -> None:
        self.config = config or ShortcutRouterConfig()
        self._llm_json = llm_json
        self._disabled_reason = "stub"

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def shortcut_to_sandbox_code(self, query: str, profile: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        return None
