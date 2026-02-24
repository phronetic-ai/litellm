from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

THESYS_API_BASE = "https://api.thesys.dev/v1/embed"


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

