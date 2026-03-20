import json
from types import SimpleNamespace

from tools.image_generation_tool import (
    check_image_generation_requirements,
    image_generate_tool,
)


def test_check_image_generation_requirements_accepts_venice(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setenv("VENICE_API_KEY", "venice-key")
    monkeypatch.delenv("VENICE_BASE_URL", raising=False)
    monkeypatch.delenv("VENICE_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("IMAGE_GENERATION_PROVIDER", raising=False)
    monkeypatch.setattr(
        "tools.image_generation_tool._import_openai_client",
        lambda: object,
    )

    assert check_image_generation_requirements() is True


def test_image_generate_tool_uses_venice_when_explicitly_selected(monkeypatch):
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "venice")
    monkeypatch.setenv("VENICE_API_KEY", "venice-key")
    monkeypatch.setenv("FAL_KEY", "fal-key")
    monkeypatch.delenv("VENICE_BASE_URL", raising=False)
    monkeypatch.delenv("VENICE_IMAGE_MODEL", raising=False)

    class _FakeImages:
        def generate(self, **kwargs):
            assert kwargs["model"] == "nano-banana-pro"
            assert kwargs["size"] == "1536x1024"
            assert kwargs["n"] == 1
            return SimpleNamespace(
                data=[SimpleNamespace(url="https://img.venice.example/generated.png")]
            )

    class _FakeClient:
        def __init__(self, api_key, base_url):
            assert api_key == "venice-key"
            assert base_url == "https://api.venice.ai/api/v1"
            self.images = _FakeImages()

    monkeypatch.setattr(
        "tools.image_generation_tool._import_openai_client",
        lambda: _FakeClient,
    )

    result = json.loads(image_generate_tool("A neon city skyline at dusk"))

    assert result["success"] is True
    assert result["image"] == "https://img.venice.example/generated.png"
