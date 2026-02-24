from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

THESYS_API_BASE = "https://api.thesys.dev/v1/embed"


def _register_thesys_model_costs() -> None:
    import json

    import litellm
    from importlib.resources import files

    from litellm.utils import _invalidate_model_cost_lowercase_map

    try:
        content = (
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        all_models = json.loads(content)
        new_entries = {
            k: v
            for k, v in all_models.items()
            if k.startswith("thesys/") and k not in litellm.model_cost
        }
        if new_entries:
            litellm.model_cost.update(new_entries)
            _invalidate_model_cost_lowercase_map()
    except Exception:
        pass


_register_thesys_model_costs()


class ThesysChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "thesys"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("THESYS_API_BASE") or THESYS_API_BASE
        dynamic_api_key = api_key or get_secret_str("THESYS_API_KEY")
        return api_base, dynamic_api_key

