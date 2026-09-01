"""Pure tests for agent.azure_endpoint. No network, no env vars needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import azure_endpoint  # noqa: E402


def test_foundry_project_url_reduces_to_scheme_and_host():
    raw = "https://my-resource.services.ai.azure.com/api/projects/my-project"
    assert azure_endpoint(raw) == "https://my-resource.services.ai.azure.com"


def test_bare_azure_openai_url_is_unchanged():
    assert azure_endpoint("https://x.openai.azure.com/") == "https://x.openai.azure.com"
