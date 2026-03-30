import importlib.util
import os


def _contains_token(value: str | None, token: str) -> bool:
    if not value:
        return False
    return token.lower() in value.lower()


def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn_implementation(model_source: str | None = None) -> str:
    if _contains_token(model_source, "Qwen3.5") or _contains_token(model_source, "Qwen3_5"):
        return "eager"

    if has_flash_attn():
        return "flash_attention_2"

    if os.getenv("COLETTE_REQUIRE_FLASH_ATTN") == "1":
        raise ImportError(
            "COLETTE_REQUIRE_FLASH_ATTN=1 but flash_attn is not installed in the active environment"
        )

    return "eager"