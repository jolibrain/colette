import importlib.util
import os


def has_flash_attn() -> bool:
    if importlib.util.find_spec("flash_attn") is None:
        return False

    # A broken flash-attn binary can exist in site-packages but fail to import.
    # Treat that as unavailable so we safely fall back to eager attention.
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False

    return True


def _contains_token(value: str | None, token: str) -> bool:
    if not value:
        return False
    return token.lower() in value.lower()


def resolve_attn_implementation(model_source: str | None = None) -> str:
    # Qwen3.5 uses 3D mrope position IDs for multimodal tokens. Transformers
    # 5.x dispatches those through flash-attn's varlen kernel, which crashes
    # on current hardware (illegal memory access in flash_attn_gpu.varlen_fwd).
    # PyTorch's native sdpa handles those position IDs correctly and is still
    # hardware-accelerated — it is NOT the same as eager mode.
    if _contains_token(model_source, "Qwen3.5") or _contains_token(model_source, "Qwen3_5"):
        return "sdpa"

    if has_flash_attn():
        return "flash_attention_2"

    if os.getenv("COLETTE_REQUIRE_FLASH_ATTN") == "1":
        raise ImportError(
            "COLETTE_REQUIRE_FLASH_ATTN=1 but flash_attn is not installed in the active environment"
        )

    return "eager"